"""Conformance tests for the optional OpenUSD state carrier."""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from io import StringIO
import tempfile
import unittest

import numpy as np

import gat.demo
from gat.causal import AssessmentRecord
from gat.adapters.openusd import (
    OPENUSD_CARRIER_VERSION,
    OpenUsdReadLimits,
    generate_openusd_keypair,
    migrate_openusd,
    openusd_available,
    read_openusd,
    write_openusd,
)
from gat.engine.transform import ObserveQuantity, ShiftParameter
from gat.engine.dynamics import EvolveLinearGaussian
from gat.demo.openusd_portability import run_demo
from gat.errors import OpenUsdError, SnapshotError
from gat.ledger import ExecutionLedger, replay_ledger
from gat.session import GatSession
from gat.state_snapshot import computational_equivalence


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")

if openusd_available():
    from pxr import Sdf, Usd


@unittest.skipUnless(openusd_available(), "optional usd-core runtime is not installed")
class OpenUsdCarrierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = GatSession.load_ifc(MODEL)
        volume = self.session.var("Office-A", "Volume")
        self.session.run(ObserveQuantity.single(volume, 59.4, noise_sigma=0.05))

    def test_usda_roundtrip_preserves_exact_computational_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            trace = tuple(self.session.trace.events)
            digest = write_openusd(self.session.world, path, trace)
            restored = read_openusd(path)
        report = computational_equivalence(self.session.world, restored.world)
        self.assertTrue(report.passed, report.render())
        self.assertEqual(restored.trace_events, trace)
        self.assertEqual(restored.snapshot_digest, digest)

    def test_usdc_binary_carrier_roundtrips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usdc")
            write_openusd(self.session.world, path)
            restored = read_openusd(path)
        self.assertTrue(
            computational_equivalence(self.session.world, restored.world).passed
        )

    def test_stage_exposes_native_state_and_separate_derived_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            write_openusd(self.session.world, path)
            stage = Usd.Stage.Open(path)
            root = stage.GetDefaultPrim()
            state = stage.GetPrimAtPath(root.GetPath().AppendChild("State"))
            view = stage.GetPrimAtPath(root.GetPath().AppendChild("View"))
            belief = stage.GetPrimAtPath(state.GetPath().AppendChild("Belief"))
            raw_targets = belief.GetRelationship("gat:rawVariables").GetTargets()
            covariance = belief.GetAttribute("gat:rawCovariance")

            self.assertEqual(root.GetName(), "GAT")
            self.assertTrue(state.GetAttribute("gat:authoritative").Get())
            self.assertFalse(view.GetAttribute("gat:authoritative").Get())
            self.assertEqual(len(raw_targets), self.session.world.binding.n_raw)
            self.assertEqual(str(covariance.GetTypeName()), "double[]")
            self.assertEqual(
                len(covariance.Get()), self.session.world.binding.n_raw**2
            )
            view_entities = stage.GetPrimAtPath(
                view.GetPath().AppendChild("Entities")
            )
            self.assertGreater(len(view_entities.GetChildren()), 0)
            first = view_entities.GetChildren()[0]
            self.assertEqual(
                len(first.GetRelationship("gat:source").GetTargets()), 1
            )

    def test_stage_exposes_and_roundtrips_authoritative_execution_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            self.session.export_openusd(path)
            stage = Usd.Stage.Open(path)
            ledger_prim = stage.GetPrimAtPath("/GAT/State/Ledger")
            events = stage.GetPrimAtPath("/GAT/State/Ledger/Events")
            restored = read_openusd(path)

        self.assertTrue(ledger_prim.GetAttribute("gat:authoritative").Get())
        self.assertEqual(
            ledger_prim.GetAttribute("gat:ledgerHead").Get(), self.session.ledger.head
        )
        self.assertEqual(len(events.GetChildren()), len(self.session.ledger.events))
        self.assertIsNotNone(restored.ledger)
        assert restored.ledger is not None
        self.assertEqual(restored.ledger.to_dict(), self.session.ledger.to_dict())
        checkpoint = GatSession.load_ifc(MODEL).world
        replayed = replay_ledger(checkpoint, restored.ledger)
        self.assertEqual(replayed.world.digest(), self.session.world.digest())

    def test_joint_off_diagonal_covariance_survives(self) -> None:
        covariance = self.session.world.belief.sigma
        off_diagonal = covariance - np.diag(np.diag(covariance))
        self.assertGreater(np.abs(off_diagonal).max(), 0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usdc")
            write_openusd(self.session.world, path)
            restored = read_openusd(path).world
        self.assertTrue(np.array_equal(restored.belief.sigma, covariance))

    def test_usda_export_is_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "first.usda")
            second = os.path.join(directory, "second.usda")
            write_openusd(self.session.world, first, self.session.trace.events)
            write_openusd(self.session.world, second, self.session.trace.events)
            with open(first, "rb") as stream:
                first_bytes = stream.read()
            with open(second, "rb") as stream:
                second_bytes = stream.read()
        self.assertEqual(first_bytes, second_bytes)

    def test_derived_view_edits_do_not_change_authoritative_state(self) -> None:
        key = generate_openusd_keypair("view-independent")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            write_openusd(self.session.world, path, signing_key=key)
            stage = Usd.Stage.Open(path)
            view = stage.GetPrimAtPath("/GAT/View")
            view.CreateAttribute(
                "review:note", Sdf.ValueTypeNames.String, custom=True
            ).Set("derived presentation may change")
            stage.GetRootLayer().Save()
            restored = read_openusd(
                path,
                trusted_public_keys={key.key_id: key.public_key},
                require_signature=True,
            )
        self.assertTrue(restored.signature.verified)
        report = computational_equivalence(self.session.world, restored.world)
        self.assertTrue(report.passed, report.render())

    def test_authoritative_belief_edit_without_new_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            write_openusd(self.session.world, path)
            stage = Usd.Stage.Open(path)
            attribute = stage.GetPrimAtPath("/GAT/State/Belief").GetAttribute(
                "gat:rawMean"
            )
            values = list(attribute.Get())
            values[0] += 0.1
            attribute.Set(values)
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(SnapshotError, "integrity digest mismatch"):
                read_openusd(path)

    def test_namespace_renames_preserve_identity_and_relationships(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            write_openusd(self.session.world, path)
            stage = Usd.Stage.Open(path)
            entities = stage.GetPrimAtPath("/GAT/State/Entities")
            entity = next(
                prim
                for prim in entities.GetChildren()
                if stage.GetPrimAtPath(
                    prim.GetPath().AppendChild("Quantities")
                ).GetChildren()
            )
            editor = Usd.NamespaceEditor(stage)
            self.assertTrue(editor.RenamePrim(entity, "RenamedEntity"))
            self.assertTrue(editor.ApplyEdits())

            renamed = stage.GetPrimAtPath(
                entities.GetPath().AppendChild("RenamedEntity")
            )
            quantity = stage.GetPrimAtPath(
                renamed.GetPath().AppendChild("Quantities")
            ).GetChildren()[0]
            editor = Usd.NamespaceEditor(stage)
            self.assertTrue(editor.RenamePrim(quantity, "RenamedQuantity"))
            self.assertTrue(editor.ApplyEdits())
            stage.GetRootLayer().Save()
            restored = read_openusd(path).world
        report = computational_equivalence(self.session.world, restored)
        self.assertTrue(report.passed, report.render())

    def test_reference_composition_can_rehouse_complete_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.usdc")
            composed = os.path.join(directory, "composed.usda")
            write_openusd(self.session.world, source)
            stage = Usd.Stage.CreateNew(composed)
            root = stage.DefinePrim("/PortableBelief", "Scope")
            root.GetReferences().AddReference(source, "/GAT")
            stage.SetDefaultPrim(root)
            stage.OverridePrim("/PortableBelief/View").CreateAttribute(
                "review:layer", Sdf.ValueTypeNames.String, custom=True
            ).Set("stronger presentation opinion")
            stage.GetRootLayer().Save()
            restored = read_openusd(composed).world
        report = computational_equivalence(self.session.world, restored)
        self.assertTrue(report.passed, report.render())

    def test_continuation_matches_uninterrupted_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.usdc")
            self.session.export_openusd(path)
            resumed = GatSession.load_openusd(path)
        self.session.run(
            ShiftParameter(self.session.var("Level 1", "ClearHeight"), 0.10)
        )
        resumed.run(ShiftParameter(resumed.var("Level 1", "ClearHeight"), 0.10))
        report = computational_equivalence(self.session.world, resumed.world)
        self.assertTrue(report.passed, report.render())
        self.assertEqual(self.session.ledger.head, resumed.ledger.head)
        self.assertEqual(self.session.ledger.to_dict(), resumed.ledger.to_dict())

    def test_geometry_can_be_omitted_without_weakening_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            write_openusd(self.session.world, path, include_geometry=False)
            stage = Usd.Stage.Open(path)
            self.assertFalse(stage.GetPrimAtPath("/GAT/View"))
            restored = read_openusd(path).world
        self.assertTrue(
            computational_equivalence(self.session.world, restored).passed
        )

    def test_non_usd_extension_is_rejected(self) -> None:
        with self.assertRaisesRegex(OpenUsdError, "must end"):
            write_openusd(self.session.world, "state.json")

    def test_separate_process_openusd_continuation_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                digest = run_demo(directory)
            self.assertEqual(len(digest), 64)
            self.assertTrue(
                os.path.exists(os.path.join(directory, "state_t2_resumed.usdc"))
            )

    def test_signed_carrier_verifies_under_explicit_trust_store(self) -> None:
        key = generate_openusd_keypair("survey-authority")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "signed.usdc")
            write_openusd(self.session.world, path, signing_key=key)
            untrusted = read_openusd(path)
            restored = read_openusd(
                path,
                trusted_public_keys={key.key_id: key.public_key},
                require_signature=True,
            )
            resumed = GatSession.load_openusd(
                path,
                trusted_public_keys={key.key_id: key.public_key},
                require_signature=True,
            )
        self.assertTrue(restored.signature.present)
        self.assertTrue(restored.signature.verified)
        self.assertTrue(untrusted.signature.present)
        self.assertFalse(untrusted.signature.verified)
        self.assertEqual(restored.signature.key_id, key.key_id)
        self.assertIn("signature verified", resumed.trace.events[-1].detail)
        self.assertTrue(
            computational_equivalence(self.session.world, restored.world).passed
        )

    def test_unsigned_or_unknown_key_fails_closed_when_trust_is_required(self) -> None:
        key = generate_openusd_keypair("known")
        with tempfile.TemporaryDirectory() as directory:
            unsigned = os.path.join(directory, "unsigned.usdc")
            signed = os.path.join(directory, "signed.usdc")
            write_openusd(self.session.world, unsigned)
            write_openusd(self.session.world, signed, signing_key=key)
            with self.assertRaisesRegex(OpenUsdError, "requires a signed"):
                read_openusd(unsigned, require_signature=True)
            with self.assertRaisesRegex(OpenUsdError, "no trusted public key"):
                read_openusd(
                    signed,
                    trusted_public_keys={},
                    require_signature=True,
                )

    def test_invalid_signature_is_rejected_when_key_is_trusted(self) -> None:
        key = generate_openusd_keypair("trusted")
        other = generate_openusd_keypair("other")
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "signed.usda")
            write_openusd(self.session.world, path, signing_key=key)
            with self.assertRaisesRegex(OpenUsdError, "signature verification failed"):
                read_openusd(
                    path,
                    trusted_public_keys={key.key_id: other.public_key},
                    require_signature=True,
                )

    def test_resource_limits_fail_before_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usdc")
            write_openusd(self.session.world, path, ledger=self.session.ledger)
            cases = (
                (OpenUsdReadLimits(max_file_bytes=1), "root layer exceeds"),
                (OpenUsdReadLimits(max_composed_prims=1), "prim count exceeds"),
                (OpenUsdReadLimits(max_entities=1), "entities count exceeds"),
                (OpenUsdReadLimits(max_raw_variables=1), "raw variable count exceeds"),
                (OpenUsdReadLimits(max_json_chars=1), "JSON budget exceeds"),
                (OpenUsdReadLimits(max_ledger_events=1), "ledger event count exceeds"),
            )
            for limits, message in cases:
                with self.subTest(message=message):
                    with self.assertRaisesRegex(OpenUsdError, message):
                        read_openusd(path, limits=limits)

    def test_read_limits_require_positive_integer_budgets(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_entities"):
            OpenUsdReadLimits(max_entities=0)

    def test_version_one_carrier_migrates_to_signed_current_version(self) -> None:
        key = generate_openusd_keypair("migration-authority")
        with tempfile.TemporaryDirectory() as directory:
            legacy = os.path.join(directory, "legacy-v1.usda")
            current = os.path.join(directory, "current-v2.usdc")
            write_openusd(self.session.world, legacy)
            stage = Usd.Stage.Open(legacy)
            stage.GetDefaultPrim().GetAttribute("gat:carrierVersion").Set(1)
            stage.GetRootLayer().Save()

            report = migrate_openusd(legacy, current, signing_key=key)
            restored = read_openusd(
                current,
                trusted_public_keys={key.key_id: key.public_key},
                require_signature=True,
            )
        self.assertEqual(report.source_version, 1)
        self.assertEqual(report.target_version, OPENUSD_CARRIER_VERSION)
        self.assertEqual(report.destination_signed_by, key.key_id)
        self.assertTrue(restored.signature.verified)
        self.assertIsNotNone(restored.ledger)
        assert restored.ledger is not None
        self.assertEqual(report.ledger_head, restored.ledger.head)
        self.assertTrue(
            computational_equivalence(self.session.world, restored.world).passed
        )

    def test_signed_migration_cannot_strip_or_rebless_unverified_source(self) -> None:
        source_key = generate_openusd_keypair("source")
        destination_key = generate_openusd_keypair("destination")
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.usdc")
            destination = os.path.join(directory, "destination.usdc")
            write_openusd(
                self.session.world,
                source,
                signing_key=source_key,
                ledger=self.session.ledger,
            )
            with self.assertRaisesRegex(OpenUsdError, "must verify"):
                migrate_openusd(source, destination, signing_key=destination_key)
            with self.assertRaisesRegex(OpenUsdError, "destination signing key"):
                migrate_openusd(
                    source,
                    destination,
                    trusted_public_keys={source_key.key_id: source_key.public_key},
                )
            report = migrate_openusd(
                source,
                destination,
                trusted_public_keys={source_key.key_id: source_key.public_key},
                signing_key=destination_key,
            )
            migrated = read_openusd(
                destination,
                trusted_public_keys={destination_key.key_id: destination_key.public_key},
                require_signature=True,
            )
            self.assertEqual(report.destination_signed_by, destination_key.key_id)
            self.assertTrue(migrated.signature.verified)
            self.assertIsNotNone(migrated.ledger)
            assert migrated.ledger is not None
            self.assertEqual(migrated.ledger.head, self.session.ledger.head)

    def test_legacy_v2_remains_readable_and_migrates_with_explicit_genesis(self) -> None:
        key = generate_openusd_keypair("v2-migration")
        with tempfile.TemporaryDirectory() as directory:
            legacy = os.path.join(directory, "legacy-v2.usda")
            current = os.path.join(directory, "current-v3.usdc")
            write_openusd(self.session.world, legacy)
            stage = Usd.Stage.Open(legacy)
            stage.GetDefaultPrim().GetAttribute("gat:carrierVersion").Set(2)
            stage.GetRootLayer().Save()

            legacy_loaded = read_openusd(legacy)
            report = migrate_openusd(legacy, current, signing_key=key)
            migrated = read_openusd(
                current,
                trusted_public_keys={key.key_id: key.public_key},
                require_signature=True,
            )
        self.assertIsNone(legacy_loaded.ledger)
        self.assertEqual(report.source_version, 2)
        self.assertIsNotNone(migrated.ledger)
        assert migrated.ledger is not None
        self.assertEqual(len(migrated.ledger.events), 1)
        self.assertEqual(
            migrated.ledger.events[0].provenance["checkpoint"],
            "openusd-carrier-migration",
        )

    def test_ledger_edit_without_rehash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "state.usda")
            self.session.export_openusd(path)
            stage = Usd.Stage.Open(path)
            event = stage.GetPrimAtPath("/GAT/State/Ledger/Events").GetChildren()[-1]
            event.GetAttribute("gat:provenance").Set('{"sensor":"forged"}')
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(OpenUsdError, "invalid embedded execution ledger"):
                read_openusd(path)

    def test_valid_ledger_for_another_world_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            carrier = os.path.join(directory, "carrier.usda")
            other = os.path.join(directory, "other.usda")
            write_openusd(self.session.world, carrier)
            changed = GatSession.load_ifc(MODEL)
            changed.run(
                ShiftParameter(changed.var("Level 1", "ClearHeight"), 0.01)
            )
            write_openusd(changed.world, other, ledger=changed.ledger)

            carrier_stage = Usd.Stage.Open(carrier)
            other_stage = Usd.Stage.Open(other)
            Sdf.CopySpec(
                other_stage.GetRootLayer(),
                Sdf.Path("/GAT/State/Ledger"),
                carrier_stage.GetRootLayer(),
                Sdf.Path("/GAT/State/Ledger"),
            )
            carrier_stage.GetRootLayer().Save()
            with self.assertRaisesRegex(OpenUsdError, "does not describe"):
                read_openusd(carrier)

    def test_signature_binds_ledger_head_not_only_snapshot(self) -> None:
        key = generate_openusd_keypair("ledger-authority")
        with tempfile.TemporaryDirectory() as directory:
            signed = os.path.join(directory, "signed.usda")
            alternate = os.path.join(directory, "alternate.usda")
            self.session.export_openusd(signed, signing_key=key)
            alternate_ledger = ExecutionLedger.genesis(self.session.world)
            write_openusd(
                self.session.world,
                alternate,
                ledger=alternate_ledger,
            )

            signed_stage = Usd.Stage.Open(signed)
            alternate_stage = Usd.Stage.Open(alternate)
            Sdf.CopySpec(
                alternate_stage.GetRootLayer(),
                Sdf.Path("/GAT/State/Ledger"),
                signed_stage.GetRootLayer(),
                Sdf.Path("/GAT/State/Ledger"),
            )
            signed_stage.GetRootLayer().Save()

            # The replacement ledger is internally valid and reaches the same
            # state, but it was not the history signed by this publisher.
            untrusted = read_openusd(signed)
            self.assertEqual(untrusted.ledger.head, alternate_ledger.head)
            with self.assertRaisesRegex(OpenUsdError, "signature verification failed"):
                read_openusd(
                    signed,
                    trusted_public_keys={key.key_id: key.public_key},
                    require_signature=True,
                )

    def test_signed_carrier_preserves_non_mutating_causal_events(self) -> None:
        key = generate_openusd_keypair("causal-history-authority")
        digest = self.session.world.digest()
        self.session.record_assessment(
            AssessmentRecord(
                digest,
                "assessment-42",
                "clearance-assurance",
                "route 42",
                "UNRESOLVED",
                "dependence-safe-clearance-bounds-v1",
                details={"p_any_violation_upper": 0.12},
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "causal.usdc")
            self.session.export_openusd(path, signing_key=key)
            restored = read_openusd(
                path,
                trusted_public_keys={key.key_id: key.public_key},
                require_signature=True,
            )
        self.assertTrue(restored.signature.verified)
        self.assertIsNotNone(restored.ledger)
        assert restored.ledger is not None
        self.assertEqual(restored.ledger.to_dict(), self.session.ledger.to_dict())
        self.assertEqual(restored.ledger.events[-1].kind, "assessment")

    def test_temporal_process_continuation_preserves_identical_history(self) -> None:
        process = EvolveLinearGaussian(
            (self.session.var("Level 1", "ClearHeight"),),
            np.array([[1.0]]),
            np.array([0.001]),
            np.array([[1.0e-6]]),
            60.0,
            "building-drift-v1",
            "a" * 64,
        )
        self.session.run(process)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "temporal.usdc")
            self.session.export_openusd(path)
            resumed = GatSession.load_openusd(path)
        self.session.run(process)
        resumed.run(process)
        self.assertEqual(resumed.world.digest(), self.session.world.digest())
        self.assertEqual(resumed.ledger.to_dict(), self.session.ledger.to_dict())

    def test_derived_variant_selection_preserves_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.usdc")
            composed = os.path.join(directory, "variants.usda")
            write_openusd(self.session.world, source)
            stage = Usd.Stage.CreateNew(composed)
            root = stage.DefinePrim("/PortableBelief", "Scope")
            root.GetReferences().AddReference(source, "/GAT")
            stage.SetDefaultPrim(root)
            variants = root.GetVariantSets().AddVariantSet("reviewState")
            for selection in ("draft", "approved"):
                variants.AddVariant(selection)
                variants.SetVariantSelection(selection)
                with variants.GetVariantEditContext():
                    stage.OverridePrim("/PortableBelief/View").CreateAttribute(
                        "review:status", Sdf.ValueTypeNames.String, custom=True
                    ).Set(selection)
            for selection in ("draft", "approved"):
                variants.SetVariantSelection(selection)
                restored = read_openusd(composed).world
                report = computational_equivalence(self.session.world, restored)
                self.assertTrue(report.passed, report.render())

    def test_authoritative_variant_opinion_is_integrity_checked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "source.usdc")
            composed = os.path.join(directory, "variants.usda")
            write_openusd(self.session.world, source)
            stage = Usd.Stage.CreateNew(composed)
            root = stage.DefinePrim("/PortableBelief", "Scope")
            root.GetReferences().AddReference(source, "/GAT")
            stage.SetDefaultPrim(root)
            variants = root.GetVariantSets().AddVariantSet("beliefState")
            variants.AddVariant("tampered")
            variants.SetVariantSelection("tampered")
            with variants.GetVariantEditContext():
                belief = stage.OverridePrim("/PortableBelief/State/Belief")
                attribute = belief.GetAttribute("gat:rawMean")
                values = list(attribute.Get())
                values[0] += 0.1
                attribute.Set(values)
            stage.GetRootLayer().Save()
            with self.assertRaisesRegex(SnapshotError, "integrity digest mismatch"):
                read_openusd(composed)


if __name__ == "__main__":
    unittest.main()
