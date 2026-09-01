"""Tests for gat/adapters/ifc/lexer.py and parser.py — the SPF layer.

Lexer: token kinds and (line, col) positions for strings with '' escapes,
enums, #refs, real literals, and comments; SpfParseError with locations.
Parser: value model (nested aggregates, Typed, $, *), header schema
extraction, by_type ordering, deref/duplicate errors, opaque unknowns.
"""

from __future__ import annotations

import unittest

from gat.adapters.ifc.lexer import TokKind, tokenize
from gat.adapters.ifc.parser import (
    OMITTED,
    EnumVal,
    Ref,
    Typed,
    parse_ifc,
)
from gat.errors import SpfParseError


def spf(data_lines: list[str], header: str = "FILE_SCHEMA(('IFC4'));") -> str:
    """Wrap DATA-section lines in a minimal valid SPF file."""
    return (
        "ISO-10303-21;\nHEADER;\n"
        + header
        + "\nENDSEC;\nDATA;\n"
        + "\n".join(data_lines)
        + "\nENDSEC;\nEND-ISO-10303-21;\n"
    )


class TestLexerTokens(unittest.TestCase):
    def test_kinds_values_and_positions(self):
        # Layout (1-based cols):
        # line 1: 'it''s' #7          -- string cols 1-7 (7 chars), ref at col 9
        # line 2:   .LENGTHUNIT. 1.E-2 -- enum cols 3-14, real at col 16
        # line 3: -0.5 3. /* c        -- reals at cols 1 and 6, comment opens
        # line 4: c */ 42             -- comment closes at col 4, int at col 6
        text = "'it''s' #7\n  .LENGTHUNIT. 1.E-2\n-0.5 3. /* c\nc */ 42\n"
        toks = tokenize(text)
        expected = [
            (TokKind.STRING, "it's", 1, 1),
            (TokKind.REF, 7, 1, 9),
            (TokKind.ENUM, "LENGTHUNIT", 2, 3),
            (TokKind.REAL, 0.01, 2, 16),  # 1.E-2 == 1e-2
            (TokKind.REAL, -0.5, 3, 1),
            (TokKind.REAL, 3.0, 3, 6),  # trailing-dot real
            (TokKind.INT, 42, 4, 6),
        ]
        self.assertEqual(
            [(t.kind, t.value, t.line, t.col) for t in toks], expected
        )

    def test_string_double_quote_escape_only_escapes_once(self):
        toks = tokenize("''''")  # a string containing exactly one quote
        self.assertEqual(len(toks), 1)
        self.assertEqual(toks[0].kind, TokKind.STRING)
        self.assertEqual(toks[0].value, "'")

    def test_empty_string(self):
        toks = tokenize("''")
        self.assertEqual((toks[0].kind, toks[0].value), (TokKind.STRING, ""))

    def test_punctuation_kinds(self):
        toks = tokenize("$ * ( ) , ; =")
        self.assertEqual(
            [t.kind for t in toks],
            [
                TokKind.DOLLAR,
                TokKind.STAR,
                TokKind.LPAREN,
                TokKind.RPAREN,
                TokKind.COMMA,
                TokKind.SEMI,
                TokKind.EQ,
            ],
        )

    def test_real_forms(self):
        toks = tokenize("1.E-2 -0.5 3. +2.5e3 320.")
        self.assertEqual([t.kind for t in toks], [TokKind.REAL] * 5)
        self.assertEqual([t.value for t in toks], [0.01, -0.5, 3.0, 2500.0, 320.0])

    def test_signed_int_stays_int(self):
        toks = tokenize("-12 +4")
        self.assertEqual(
            [(t.kind, t.value) for t in toks],
            [(TokKind.INT, -12), (TokKind.INT, 4)],
        )

    def test_keyword_token(self):
        toks = tokenize("IFCWALL ISO-10303-21")
        self.assertEqual(
            [(t.kind, t.value) for t in toks],
            [(TokKind.KEYWORD, "IFCWALL"), (TokKind.KEYWORD, "ISO-10303-21")],
        )

    def test_comment_between_tokens_is_skipped(self):
        toks = tokenize("1/* anything ; #5 'x' */2")
        self.assertEqual([(t.kind, t.value) for t in toks], [(TokKind.INT, 1), (TokKind.INT, 2)])


class TestLexerErrors(unittest.TestCase):
    def test_unterminated_string_carries_start_position(self):
        # String opens on line 2, col 2 and never closes.
        with self.assertRaises(SpfParseError) as ctx:
            tokenize("ab\n 'unclosed")
        self.assertEqual(ctx.exception.line, 2)
        self.assertEqual(ctx.exception.col, 2)
        self.assertIn("unterminated string", str(ctx.exception))

    def test_newline_in_string_carries_start_position(self):
        with self.assertRaises(SpfParseError) as ctx:
            tokenize("'ab\ncd'")
        self.assertEqual((ctx.exception.line, ctx.exception.col), (1, 1))

    def test_malformed_real_exponent(self):
        with self.assertRaises(SpfParseError) as ctx:
            tokenize("  1.5E+ ")
        self.assertEqual((ctx.exception.line, ctx.exception.col), (1, 3))
        self.assertIn("exponent", str(ctx.exception))

    def test_unterminated_enum(self):
        with self.assertRaises(SpfParseError) as ctx:
            tokenize(".NOTCLOSED")
        self.assertIn("enum", str(ctx.exception))

    def test_bare_hash(self):
        with self.assertRaises(SpfParseError):
            tokenize("# ")

    def test_unterminated_comment(self):
        with self.assertRaises(SpfParseError):
            tokenize("/* never closed")

    def test_unexpected_character(self):
        with self.assertRaises(SpfParseError) as ctx:
            tokenize("\n\n  @")
        self.assertEqual((ctx.exception.line, ctx.exception.col), (3, 3))


class TestParserValues(unittest.TestCase):
    def parse_single(self, args_text: str):
        f = parse_ifc(spf([f"#1=FOO{args_text};"]))
        return f.instances[1]

    def test_nested_aggregates(self):
        inst = self.parse_single("(((1.,2.),(3.,4.)),(5,6))")
        self.assertEqual(inst.args, (((1.0, 2.0), (3.0, 4.0)), (5, 6)))

    def test_typed_value(self):
        inst = self.parse_single("('UnitCost',$,IFCREAL(320.),$)")
        self.assertEqual(inst.args[2], Typed("IFCREAL", (320.0,)))

    def test_typed_value_name_uppercased(self):
        inst = self.parse_single("(ifcreal(1.5))")
        self.assertEqual(inst.args[0], Typed("IFCREAL", (1.5,)))

    def test_dollar_is_none(self):
        inst = self.parse_single("($)")
        self.assertEqual(inst.args, (None,))

    def test_star_is_omitted_singleton(self):
        inst = self.parse_single("(*,$)")
        self.assertIs(inst.args[0], OMITTED)

    def test_enum_and_ref_values(self):
        inst = self.parse_single("(.LENGTHUNIT.,#42)")
        self.assertEqual(inst.args, (EnumVal("LENGTHUNIT"), Ref(42)))

    def test_empty_arg_list(self):
        inst = self.parse_single("()")
        self.assertEqual(inst.args, ())

    def test_type_name_uppercased_and_string_kept(self):
        f = parse_ifc(spf(["#3=ifcWall('it''s a wall');"]))
        inst = f.instances[3]
        self.assertEqual(inst.type_name, "IFCWALL")
        self.assertEqual(inst.args, ("it's a wall",))


class TestParserFileStructure(unittest.TestCase):
    def test_file_schema_extraction(self):
        f = parse_ifc(spf(["#1=FOO();"]))
        self.assertEqual(f.schema, "IFC4")
        self.assertIn("FILE_SCHEMA", f.header)
        self.assertEqual(f.header["FILE_SCHEMA"], (("IFC4",),))

    def test_by_type_ordered_by_step_id(self):
        # Instances appear out of order in the file; by_type sorts by id.
        f = parse_ifc(spf(["#5=BAR();", "#9=FOO();", "#2=FOO();", "#1=FOO();"]))
        self.assertEqual([i.step_id for i in f.by_type("FOO")], [1, 2, 9])
        # by_type uppercases its query.
        self.assertEqual([i.step_id for i in f.by_type("foo")], [1, 2, 9])
        self.assertEqual(f.by_type("MISSING"), ())

    def test_deref_dangling_reference_raises(self):
        f = parse_ifc(spf(["#1=FOO(#999);"]))  # target never defined
        with self.assertRaises(SpfParseError) as ctx:
            f.deref(Ref(999))
        self.assertIn("#999", str(ctx.exception))

    def test_deref_resolves_existing(self):
        f = parse_ifc(spf(["#1=FOO(#2);", "#2=BAR();"]))
        self.assertEqual(f.deref(Ref(2)).type_name, "BAR")

    def test_duplicate_instance_id_raises(self):
        with self.assertRaises(SpfParseError) as ctx:
            parse_ifc(spf(["#1=FOO();", "#1=BAR();"]))
        self.assertIn("duplicate instance #1", str(ctx.exception))

    def test_unknown_entity_types_survive_opaque(self):
        f = parse_ifc(spf(["#7=MYSTERYTHING('abc',#7,(1,2),.X.,*,$);"]))
        inst = f.instances[7]
        self.assertEqual(inst.type_name, "MYSTERYTHING")
        self.assertEqual(inst.args[0], "abc")
        self.assertEqual(inst.args[1], Ref(7))
        self.assertEqual(inst.args[2], (1, 2))
        self.assertEqual(inst.args[3], EnumVal("X"))
        self.assertIs(inst.args[4], OMITTED)
        self.assertIsNone(inst.args[5])

    def test_max_step_id(self):
        f = parse_ifc(spf(["#3=FOO();", "#17=BAR();"]))
        self.assertEqual(f.max_step_id(), 17)


class TestParserErrors(unittest.TestCase):
    def test_malformed_arg_separator_carries_position(self):
        # "#1=FOO(1 2);" — the INT 2 is where ',' or ')' was expected.
        # It sits on line 6 of the wrapped file (5 wrapper lines above),
        # col 10 (t: "#1=FOO(1 2);"  -> '2' is the 10th character).
        with self.assertRaises(SpfParseError) as ctx:
            parse_ifc(spf(["#1=FOO(1 2);"]))
        self.assertEqual((ctx.exception.line, ctx.exception.col), (6, 10))

    def test_missing_iso_prologue(self):
        with self.assertRaises(SpfParseError):
            parse_ifc("HEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n")

    def test_truncated_file(self):
        with self.assertRaises(SpfParseError):
            parse_ifc("ISO-10303-21;\nHEADER;\n")

    def test_data_section_requires_instances(self):
        # A bare value where '#id=' is expected.
        with self.assertRaises(SpfParseError):
            parse_ifc(spf(["FOO();"]))


if __name__ == "__main__":
    unittest.main()
