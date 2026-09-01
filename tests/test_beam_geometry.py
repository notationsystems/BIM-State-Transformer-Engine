"""Identity-bound IFC beam geometry derivation."""

from __future__ import annotations

import hashlib
import unittest

from gat.adapters.ifc.beam_geometry import (
    BEAM_GEOMETRY_METHOD,
    BeamGeometryStatus,
    derive_all_beam_geometry,
    derive_beam_geometry,
)
from gat.adapters.ifc.parser import parse_ifc


RECTANGULAR_BEAM = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('beam geometry test'),'2;1');
FILE_NAME('beam.ifc','2026-09-01T00:00:00',(),(),'', '', '');
FILE_SCHEMA(('IFC4'));
ENDSEC;
DATA;
#1=IFCPROJECT('PROJECT-GID',$,'Project',$,$,$,$,$,#10);
#10=IFCUNITASSIGNMENT((#11,#12));
#11=IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.);
#12=IFCSIUNIT(*,.PLANEANGLEUNIT.,$,.RADIAN.);
#100=IFCBEAM('RECT-BEAM-GID',$,'Rect Beam',$,$,$,#101,$);
#101=IFCPRODUCTDEFINITIONSHAPE($,$,(#102,#103));
#102=IFCSHAPEREPRESENTATION($,'Axis','Curve2D',(#104));
#103=IFCSHAPEREPRESENTATION($,'Body','SweptSolid',(#107));
#104=IFCPOLYLINE((#105,#106));
#105=IFCCARTESIANPOINT((0.,0.));
#106=IFCCARTESIANPOINT((3.,4.));
#107=IFCEXTRUDEDAREASOLID(#108,$,#109,5.);
#108=IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,$,#110);
#109=IFCDIRECTION((0.,0.,1.));
#110=IFCPOLYLINE((#111,#112,#113,#114,#111));
#111=IFCCARTESIANPOINT((-0.1,-0.2));
#112=IFCCARTESIANPOINT((0.1,-0.2));
#113=IFCCARTESIANPOINT((0.1,0.2));
#114=IFCCARTESIANPOINT((-0.1,0.2));
ENDSEC;
END-ISO-10303-21;
"""


class BeamGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = RECTANGULAR_BEAM.encode("utf-8")
        self.digest = hashlib.sha256(self.source).hexdigest()
        self.file = parse_ifc(RECTANGULAR_BEAM)
        self.beam = self.file.by_type("IFCBEAM")[0]

    def test_axis_and_rectangle_section_properties_match_closed_form(self) -> None:
        result = derive_beam_geometry(
            self.file,
            self.beam,
            source_ifc_sha256=self.digest,
        )

        self.assertEqual(result.status, BeamGeometryStatus.COMPLETE)
        self.assertAlmostEqual(result.axis_length.value, 5.0)
        self.assertAlmostEqual(result.cross_section_area.value, 0.08)
        self.assertAlmostEqual(result.section_modulus_major.value, 0.2 * 0.4**2 / 6.0)
        self.assertAlmostEqual(result.section_modulus_minor.value, 0.4 * 0.2**2 / 6.0)
        document = result.to_dict()
        self.assertEqual(document["method"], BEAM_GEOMETRY_METHOD)
        self.assertEqual(document["subject"]["global_id"], "RECT-BEAM-GID")
        self.assertEqual(document["provenance"]["source_ifc_sha256"], self.digest)
        self.assertGreater(len(document["provenance"]["source_refs"]), 4)
        self.assertIn(
            "no as-built",
            document["provenance"]["uncertainty_scope"],
        )

    def test_unsupported_body_is_explicit_length_only_not_bounding_box(self) -> None:
        text = RECTANGULAR_BEAM.replace("'SweptSolid'", "'SurfaceModel'")
        source = text.encode("utf-8")
        file = parse_ifc(text)
        result = derive_beam_geometry(
            file,
            file.by_type("IFCBEAM")[0],
            source_ifc_sha256=hashlib.sha256(source).hexdigest(),
        )

        self.assertEqual(result.status, BeamGeometryStatus.LENGTH_ONLY)
        self.assertAlmostEqual(result.axis_length.value, 5.0)
        self.assertIsNone(result.section_modulus_major)
        self.assertIn("SurfaceModel", result.issues[0])

    def test_all_beam_derivation_is_ordered_and_digest_deterministic(self) -> None:
        first = derive_all_beam_geometry(
            self.file,
            source_ifc_sha256=self.digest,
        )
        second = derive_all_beam_geometry(
            self.file,
            source_ifc_sha256=self.digest,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(first[0].digest(), second[0].digest())

    def test_malformed_beam_isolated_without_hiding_valid_results(self) -> None:
        text = RECTANGULAR_BEAM.replace(
            "ENDSEC;\nEND-ISO-10303-21;",
            (
                "#200=IFCBEAM('BLOCKED-BEAM-GID',$,'Blocked Beam',$,$,$,$,$);\n"
                "ENDSEC;\nEND-ISO-10303-21;"
            ),
        )
        source = text.encode("utf-8")
        results = derive_all_beam_geometry(
            parse_ifc(text),
            source_ifc_sha256=hashlib.sha256(source).hexdigest(),
        )

        self.assertEqual(
            [result.status for result in results],
            [BeamGeometryStatus.COMPLETE, BeamGeometryStatus.BLOCKED],
        )
        self.assertEqual(results[0].beam_global_id, "RECT-BEAM-GID")
        self.assertEqual(results[1].beam_global_id, "BLOCKED-BEAM-GID")
        self.assertIn("Representation", results[1].issues[0])


if __name__ == "__main__":
    unittest.main()
