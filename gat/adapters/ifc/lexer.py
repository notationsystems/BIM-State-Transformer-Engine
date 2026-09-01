"""Tokenizer for STEP Physical File (ISO 10303-21) text — the IFC subset.

Hand-rolled scanner producing ``(kind, value, line, col)`` tokens.  It is
schema-agnostic: it knows the token grammar, not IFC.  Errors carry source
locations.  Codepage escapes (``\\X\\...``) are passed through verbatim —
decoding them is out of scope for v0.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from gat.errors import SpfParseError


class TokKind(Enum):
    KEYWORD = "keyword"      # IFCWALL, ISO-10303-21, HEADER, ...
    INT = "int"
    REAL = "real"
    STRING = "string"
    ENUM = "enum"            # .T., .METRE.
    REF = "ref"              # #123
    DOLLAR = "dollar"        # $
    STAR = "star"            # *
    LPAREN = "lparen"
    RPAREN = "rparen"
    COMMA = "comma"
    SEMI = "semi"
    EQ = "eq"


@dataclass(frozen=True)
class Token:
    kind: TokKind
    value: str | int | float
    line: int
    col: int


_KEYWORD_START = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz")
_KEYWORD_BODY = _KEYWORD_START | set("0123456789-")
_DIGITS = set("0123456789")


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(text)

    def error(msg: str) -> SpfParseError:
        return SpfParseError(msg, line, col)

    while i < n:
        ch = text[i]

        if ch == "\n":
            i += 1
            line += 1
            col = 1
            continue
        if ch in " \t\r":
            i += 1
            col += 1
            continue

        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            if end < 0:
                raise error("unterminated comment")
            for c in text[i : end + 2]:
                if c == "\n":
                    line += 1
                    col = 1
                else:
                    col += 1
            i = end + 2
            continue

        start_line, start_col = line, col

        if ch == "'":
            j = i + 1
            buf = []
            while True:
                if j >= n:
                    raise SpfParseError("unterminated string", start_line, start_col)
                c = text[j]
                if c == "'":
                    if j + 1 < n and text[j + 1] == "'":
                        buf.append("'")
                        j += 2
                        continue
                    break
                if c == "\n":
                    raise SpfParseError("newline in string", start_line, start_col)
                buf.append(c)
                j += 1
            tokens.append(Token(TokKind.STRING, "".join(buf), start_line, start_col))
            col += j + 1 - i
            i = j + 1
            continue

        if ch == ".":
            j = i + 1
            while j < n and text[j] != "." and text[j] not in "\n":
                j += 1
            if j >= n or text[j] != ".":
                raise SpfParseError("unterminated enum literal", start_line, start_col)
            tokens.append(Token(TokKind.ENUM, text[i + 1 : j], start_line, start_col))
            col += j + 1 - i
            i = j + 1
            continue

        if ch == "#":
            j = i + 1
            while j < n and text[j] in _DIGITS:
                j += 1
            if j == i + 1:
                raise error("bare '#' without instance id")
            tokens.append(Token(TokKind.REF, int(text[i + 1 : j]), start_line, start_col))
            col += j - i
            i = j
            continue

        if ch in _DIGITS or (ch in "+-" and i + 1 < n and (text[i + 1] in _DIGITS or text[i + 1] == ".")):
            j = i
            if text[j] in "+-":
                j += 1
            while j < n and text[j] in _DIGITS:
                j += 1
            is_real = False
            if j < n and text[j] == ".":
                is_real = True
                j += 1
                while j < n and text[j] in _DIGITS:
                    j += 1
            if j < n and text[j] in "eE":
                is_real = True
                j += 1
                if j < n and text[j] in "+-":
                    j += 1
                if j >= n or text[j] not in _DIGITS:
                    raise SpfParseError("malformed real exponent", start_line, start_col)
                while j < n and text[j] in _DIGITS:
                    j += 1
            literal = text[i:j]
            if is_real:
                tokens.append(Token(TokKind.REAL, float(literal), start_line, start_col))
            else:
                tokens.append(Token(TokKind.INT, int(literal), start_line, start_col))
            col += j - i
            i = j
            continue

        if ch in _KEYWORD_START:
            j = i + 1
            while j < n and text[j] in _KEYWORD_BODY:
                j += 1
            tokens.append(Token(TokKind.KEYWORD, text[i:j], start_line, start_col))
            col += j - i
            i = j
            continue

        simple = {
            "$": TokKind.DOLLAR,
            "*": TokKind.STAR,
            "(": TokKind.LPAREN,
            ")": TokKind.RPAREN,
            ",": TokKind.COMMA,
            ";": TokKind.SEMI,
            "=": TokKind.EQ,
        }
        if ch in simple:
            tokens.append(Token(simple[ch], ch, start_line, start_col))
            i += 1
            col += 1
            continue

        raise error(f"unexpected character {ch!r}")

    return tokens
