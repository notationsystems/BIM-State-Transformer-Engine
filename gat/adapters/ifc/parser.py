"""Parser for the STEP Physical File structure (ISO 10303-21).

Purely syntactic: produces :class:`RawInstance` records with a recursive
argument model and no schema knowledge.  Unknown entity types survive as
opaque instances — the adapter is tolerant by design, and everything it
does not understand round-trips verbatim through the canonical writer.

Value model for instance arguments:

* ``int``, ``float``, ``str``
* ``EnumVal(name)`` — ``.T.``, ``.METRE.``
* ``Ref(step_id)`` — ``#123``
* ``None`` — ``$``
* ``OMITTED`` — ``*`` (a derived attribute placeholder)
* ``tuple`` — nested aggregate ``( ... )``
* ``Typed(name, args)`` — inline typed value, e.g. ``IFCREAL(320.)``
"""

from __future__ import annotations

from dataclasses import dataclass

from gat.adapters.ifc.lexer import TokKind, Token, tokenize
from gat.errors import SpfParseError


@dataclass(frozen=True)
class EnumVal:
    name: str


@dataclass(frozen=True)
class Ref:
    step_id: int


@dataclass(frozen=True)
class Typed:
    name: str
    args: tuple


class _Omitted:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "OMITTED"


OMITTED = _Omitted()


@dataclass(frozen=True)
class RawInstance:
    step_id: int
    type_name: str
    args: tuple


@dataclass
class IfcFile:
    header: dict[str, tuple]
    instances: dict[int, RawInstance]
    schema: str

    def by_type(self, type_name: str) -> tuple[RawInstance, ...]:
        upper = type_name.upper()
        return tuple(
            inst
            for sid in sorted(self.instances)
            if (inst := self.instances[sid]).type_name == upper
        )

    def deref(self, ref: Ref) -> RawInstance:
        inst = self.instances.get(ref.step_id)
        if inst is None:
            raise SpfParseError(f"dangling reference #{ref.step_id}")
        return inst

    def max_step_id(self) -> int:
        return max(self.instances) if self.instances else 0


class _Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        if self.pos >= len(self.tokens):
            raise SpfParseError("unexpected end of file")
        return self.tokens[self.pos]

    def next(self) -> Token:
        tok = self.peek()
        self.pos += 1
        return tok

    def expect(self, kind: TokKind) -> Token:
        tok = self.next()
        if tok.kind is not kind:
            raise SpfParseError(
                f"expected {kind.value}, got {tok.kind.value} ({tok.value!r})",
                tok.line,
                tok.col,
            )
        return tok

    def expect_keyword(self, name: str) -> None:
        tok = self.expect(TokKind.KEYWORD)
        if str(tok.value).upper() != name:
            raise SpfParseError(f"expected {name}, got {tok.value!r}", tok.line, tok.col)

    def at_keyword(self, name: str) -> bool:
        tok = self.peek()
        return tok.kind is TokKind.KEYWORD and str(tok.value).upper() == name

    # -- grammar -----------------------------------------------------------

    def parse_file(self) -> IfcFile:
        self.expect_keyword("ISO-10303-21")
        self.expect(TokKind.SEMI)
        self.expect_keyword("HEADER")
        self.expect(TokKind.SEMI)

        header: dict[str, tuple] = {}
        while not self.at_keyword("ENDSEC"):
            tok = self.expect(TokKind.KEYWORD)
            args = self.parse_arg_list()
            self.expect(TokKind.SEMI)
            header[str(tok.value).upper()] = args
        self.expect_keyword("ENDSEC")
        self.expect(TokKind.SEMI)

        self.expect_keyword("DATA")
        self.expect(TokKind.SEMI)
        instances: dict[int, RawInstance] = {}
        while not self.at_keyword("ENDSEC"):
            ref_tok = self.expect(TokKind.REF)
            self.expect(TokKind.EQ)
            type_tok = self.expect(TokKind.KEYWORD)
            args = self.parse_arg_list()
            self.expect(TokKind.SEMI)
            step_id = int(ref_tok.value)
            if step_id in instances:
                raise SpfParseError(
                    f"duplicate instance #{step_id}", ref_tok.line, ref_tok.col
                )
            instances[step_id] = RawInstance(
                step_id, str(type_tok.value).upper(), args
            )
        self.expect_keyword("ENDSEC")
        self.expect(TokKind.SEMI)
        self.expect_keyword("END-ISO-10303-21")
        self.expect(TokKind.SEMI)

        schema = ""
        fs = header.get("FILE_SCHEMA")
        if fs and fs and isinstance(fs[0], tuple) and fs[0]:
            schema = str(fs[0][0])
        return IfcFile(header, instances, schema)

    def parse_arg_list(self) -> tuple:
        self.expect(TokKind.LPAREN)
        args: list = []
        if self.peek().kind is TokKind.RPAREN:
            self.next()
            return tuple(args)
        while True:
            args.append(self.parse_value())
            tok = self.next()
            if tok.kind is TokKind.RPAREN:
                return tuple(args)
            if tok.kind is not TokKind.COMMA:
                raise SpfParseError(
                    f"expected ',' or ')', got {tok.value!r}", tok.line, tok.col
                )

    def parse_value(self):
        tok = self.peek()
        if tok.kind is TokKind.INT:
            self.next()
            return int(tok.value)
        if tok.kind is TokKind.REAL:
            self.next()
            return float(tok.value)
        if tok.kind is TokKind.STRING:
            self.next()
            return str(tok.value)
        if tok.kind is TokKind.ENUM:
            self.next()
            return EnumVal(str(tok.value))
        if tok.kind is TokKind.REF:
            self.next()
            return Ref(int(tok.value))
        if tok.kind is TokKind.DOLLAR:
            self.next()
            return None
        if tok.kind is TokKind.STAR:
            self.next()
            return OMITTED
        if tok.kind is TokKind.LPAREN:
            return self.parse_arg_list()
        if tok.kind is TokKind.KEYWORD:
            self.next()
            args = self.parse_arg_list()
            return Typed(str(tok.value).upper(), args)
        raise SpfParseError(f"unexpected token {tok.value!r}", tok.line, tok.col)


def parse_ifc(text: str) -> IfcFile:
    return _Parser(tokenize(text)).parse_file()


def parse_ifc_file(path: str) -> IfcFile:
    with open(path, "r", encoding="utf-8") as fh:
        return parse_ifc(fh.read())
