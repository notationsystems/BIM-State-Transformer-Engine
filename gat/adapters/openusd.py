"""OpenUSD carrier for restartable GAT computational state.

The stage has two deliberately different branches beneath its default prim::

    /GAT/State   authoritative snapshot + causal execution ledger
    /GAT/View    disposable geometry derived from the reconstructed world

``State`` is not a JSON blob.  Entity and quantity identities are prims,
topology and belief indexing are USD relationships, and the raw mean and full
dense covariance are native numeric arrays.  Closed expression and constraint
records remain canonical JSON values on their owning prims because they are
GAT IR syntax, not executable USD behavior. Carrier v3 adds an inspectable
``State/Ledger/Events`` branch and binds its chain head into the optional
Ed25519 carrier signature.

The optional Pixar ``pxr`` runtime is imported lazily so the core package keeps
its NumPy-only dependency contract.  Install ``gat-bim[openusd]`` to use this
adapter.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

from gat.engine.executor import World
from gat.errors import LedgerError, OpenUsdError, SnapshotError
from gat.geometry.stateio import derive_scene
from gat.ledger import ExecutionLedger
from gat.state_snapshot import (
    SnapshotLoadResult,
    capture_snapshot,
    reconstruct_snapshot,
)
from gat.trace import TraceEvent


OPENUSD_CARRIER_FORMAT = "gat-openusd-state-carrier"
OPENUSD_CARRIER_VERSION = 3
OPENUSD_SUPPORTED_VERSIONS = (1, 2, 3)
OPENUSD_SIGNATURE_ALGORITHM = "ed25519"
_SIGNATURE_DOMAINS = {
    2: "gat-openusd-provenance-v1",
    3: "gat-openusd-provenance-v2",
}
_SUPPORTED_EXTENSIONS = {".usd", ".usda", ".usdc"}


@dataclass(frozen=True)
class OpenUsdReadLimits:
    """Fail-closed budgets for composed OpenUSD state reconstruction."""

    max_file_bytes: int = 512 * 1024 * 1024
    max_composed_prims: int = 250_000
    max_entities: int = 100_000
    max_quantities: int = 1_000_000
    max_relationships: int = 2_000_000
    max_constraints: int = 1_000_000
    max_raw_variables: int = 4_096
    max_covariance_values: int = 4_096**2
    max_json_chars: int = 64 * 1024 * 1024
    max_trace_events: int = 1_000_000
    max_ledger_events: int = 100_000

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


DEFAULT_OPENUSD_READ_LIMITS = OpenUsdReadLimits()


@dataclass(frozen=True)
class OpenUsdKeyPair:
    """Raw Ed25519 key material used by the optional provenance signer."""

    key_id: str
    private_key: bytes = field(repr=False)
    public_key: bytes

    def __post_init__(self) -> None:
        if not self.key_id or len(self.key_id) > 256:
            raise ValueError("key_id must contain 1..256 characters")
        if len(self.private_key) != 32 or len(self.public_key) != 32:
            raise ValueError("Ed25519 private and public keys must each be 32 bytes")


@dataclass(frozen=True)
class OpenUsdSignatureInfo:
    present: bool
    algorithm: str | None = None
    key_id: str | None = None
    verified: bool = False


@dataclass(frozen=True)
class OpenUsdLoadResult:
    world: World
    trace_events: tuple[TraceEvent, ...]
    snapshot_digest: str
    carrier_version: int
    signature: OpenUsdSignatureInfo
    ledger: ExecutionLedger | None


@dataclass(frozen=True)
class OpenUsdMigrationReport:
    source_version: int
    target_version: int
    snapshot_digest: str
    ledger_head: str
    source_signature: OpenUsdSignatureInfo
    destination_signed_by: str | None


@dataclass
class _ReadBudget:
    limits: OpenUsdReadLimits
    json_chars: int = 0
    quantities: int = 0

    def text(self, value: object, label: str) -> str:
        if not isinstance(value, str):
            raise OpenUsdError(f"{label} must be text")
        self.json_chars += len(value)
        if self.json_chars > self.limits.max_json_chars:
            raise OpenUsdError(
                f"OpenUSD JSON budget exceeds {self.limits.max_json_chars} characters"
            )
        return value

    def json(self, value: object, label: str) -> object:
        return _decode_json(self.text(value, label), label)


def openusd_available() -> bool:
    """Return whether the optional Pixar Python runtime can be imported."""
    try:
        from pxr import Usd  # noqa: F401
    except ImportError:
        return False
    return True


def generate_openusd_keypair(key_id: str | None = None) -> OpenUsdKeyPair:
    """Generate raw Ed25519 key material for carrier provenance signing.

    Production callers should normally use their existing key-management
    system and construct :class:`OpenUsdKeyPair` from controlled key material.
    """
    Ed25519PrivateKey, _, _ = _crypto()
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes_raw()
    public_bytes = private.public_key().public_bytes_raw()
    resolved_id = key_id or f"ed25519:{hashlib.sha256(public_bytes).hexdigest()[:16]}"
    return OpenUsdKeyPair(resolved_id, private_bytes, public_bytes)


def write_openusd(
    world: World,
    path: str | Path,
    trace_events: Iterable[TraceEvent] = (),
    *,
    include_geometry: bool = True,
    signing_key: OpenUsdKeyPair | None = None,
    ledger: ExecutionLedger | None = None,
) -> str:
    """Write state plus its causal ledger and return the snapshot digest."""
    Gf, Sdf, Usd, UsdGeom = _pxr()
    output = Path(path)
    if output.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise OpenUsdError("OpenUSD carrier path must end in .usd, .usda, or .usdc")
    document = capture_snapshot(world, trace_events)
    payload = _as_mapping(document["payload"], "snapshot payload")
    module = _as_mapping(payload["module"], "snapshot module")
    belief = _as_mapping(payload["belief"], "snapshot belief")
    provenance = _as_mapping(payload["provenance"], "snapshot provenance")
    integrity = _as_mapping(document["integrity"], "snapshot integrity")
    resolved_ledger = ledger or ExecutionLedger.genesis(world)
    try:
        resolved_ledger.validate()
    except LedgerError as exc:
        raise OpenUsdError(f"cannot embed invalid execution ledger: {exc}") from exc
    if resolved_ledger.events[-1].result_world_digest != world.digest():
        raise OpenUsdError("execution ledger head does not describe the exported world")
    signature = (
        None
        if signing_key is None
        else _sign_document(
            document, OPENUSD_CARRIER_VERSION, signing_key, resolved_ledger
        )
    )

    try:
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Scope.Define(stage, "/GAT").GetPrim()
        stage.SetDefaultPrim(root)
        _set(root, "gat:carrierFormat", Sdf.ValueTypeNames.String, OPENUSD_CARRIER_FORMAT)
        _set(root, "gat:carrierVersion", Sdf.ValueTypeNames.Int, OPENUSD_CARRIER_VERSION)
        _set(root, "gat:signaturePresent", Sdf.ValueTypeNames.Bool, signature is not None)
        if signature is not None:
            _set(
                root,
                "gat:signatureAlgorithm",
                Sdf.ValueTypeNames.Token,
                OPENUSD_SIGNATURE_ALGORITHM,
            )
            _set(root, "gat:signatureKeyId", Sdf.ValueTypeNames.String, signing_key.key_id)
            _set(root, "gat:signature", Sdf.ValueTypeNames.String, signature)

        state = UsdGeom.Scope.Define(stage, "/GAT/State").GetPrim()
        _set(state, "gat:authoritative", Sdf.ValueTypeNames.Bool, True)
        _set(state, "gat:snapshotFormat", Sdf.ValueTypeNames.String, document["format"])
        _set(state, "gat:snapshotVersion", Sdf.ValueTypeNames.Int, document["schema_version"])
        _set(state, "gat:runtimeContract", Sdf.ValueTypeNames.String, document["runtime_contract"])
        _set(state, "gat:integrityAlgorithm", Sdf.ValueTypeNames.String, integrity["algorithm"])
        _set(state, "gat:integrityDigest", Sdf.ValueTypeNames.String, integrity["digest"])
        _set(state, "gat:moduleMeta", Sdf.ValueTypeNames.String, _json(module["meta"]))
        _set(state, "gat:sourceModuleDigest", Sdf.ValueTypeNames.String, payload["source_module_digest"])
        _set(state, "gat:sourceWorldDigest", Sdf.ValueTypeNames.String, payload["source_world_digest"])
        _set(
            state,
            "gat:sourceConfigurationDigest",
            Sdf.ValueTypeNames.String,
            payload["source_configuration_digest"],
        )

        entities_scope = UsdGeom.Scope.Define(stage, "/GAT/State/Entities").GetPrim()
        entity_paths: dict[tuple[str, str], object] = {}
        quantity_paths: dict[tuple[str, str, str], object] = {}
        entities = _as_list(module["entities"], "module entities")
        for entity_ordinal, entity_value in enumerate(entities):
            entity = _as_mapping(entity_value, "entity")
            eid = _as_mapping(entity["id"], "entity id")
            entity_key = (str(eid["ifc_class"]), str(eid["global_id"]))
            entity_path = entities_scope.GetPath().AppendChild(
                f"E_{entity_ordinal:04d}_{_fragment(*entity_key)}"
            )
            prim = stage.DefinePrim(entity_path, "Scope")
            entity_paths[entity_key] = entity_path
            _set(prim, "gat:ordinal", Sdf.ValueTypeNames.Int, entity_ordinal)
            _set(prim, "gat:ifcClass", Sdf.ValueTypeNames.String, entity_key[0])
            _set(prim, "gat:globalId", Sdf.ValueTypeNames.String, entity_key[1])
            _set(prim, "gat:name", Sdf.ValueTypeNames.String, entity["name"])
            _set(prim, "gat:attributes", Sdf.ValueTypeNames.String, _json(entity["attrs"]))
            _write_optional_int(prim, "gat:sourceRef", entity["source_ref"], Sdf)
            placement = entity["placement"]
            _set(prim, "gat:hasPlacement", Sdf.ValueTypeNames.Bool, placement is not None)
            if placement is not None:
                placed = _as_mapping(placement, "entity placement")
                _set(
                    prim,
                    "gat:placementOrigin",
                    Sdf.ValueTypeNames.Double3,
                    Gf.Vec3d(float(placed["x"]), float(placed["y"]), float(placed["z"])),
                )
                _set(prim, "gat:placementAngle", Sdf.ValueTypeNames.Double, float(placed["angle"]))

            quantities_scope = stage.DefinePrim(entity_path.AppendChild("Quantities"), "Scope")
            for slot_ordinal, slot_value in enumerate(_as_list(entity["slots"], "entity slots")):
                slot = _as_mapping(slot_value, "quantity slot")
                var = _as_mapping(slot["var"], "slot variable")
                quantity = str(var["quantity"])
                quantity_path = quantities_scope.GetPath().AppendChild(
                    f"Q_{slot_ordinal:04d}_{_fragment(quantity)}"
                )
                quantity_prim = stage.DefinePrim(quantity_path, "Scope")
                quantity_paths[(entity_key[0], entity_key[1], quantity)] = quantity_path
                _set(quantity_prim, "gat:ordinal", Sdf.ValueTypeNames.Int, slot_ordinal)
                _set(quantity_prim, "gat:quantity", Sdf.ValueTypeNames.String, quantity)
                _set(quantity_prim, "gat:role", Sdf.ValueTypeNames.Token, slot["role"])
                _set(quantity_prim, "gat:unit", Sdf.ValueTypeNames.Token, slot["unit"])
                _set(quantity_prim, "gat:priorMu", Sdf.ValueTypeNames.Double, float(slot["prior_mu"]))
                _set(quantity_prim, "gat:priorSigma", Sdf.ValueTypeNames.Double, float(slot["prior_sigma"]))
                _set(quantity_prim, "gat:expression", Sdf.ValueTypeNames.String, _json(slot["expr"]))
                _write_optional_int(quantity_prim, "gat:sourceRef", slot["source_ref"], Sdf)
                quantity_prim.CreateRelationship("gat:owner", custom=True).SetTargets([entity_path])

        relationships_scope = UsdGeom.Scope.Define(stage, "/GAT/State/Relationships").GetPrim()
        for ordinal, rel_value in enumerate(_as_list(module["relationships"], "relationships")):
            rel = _as_mapping(rel_value, "relationship")
            prim = stage.DefinePrim(relationships_scope.GetPath().AppendChild(f"R_{ordinal:04d}"), "Scope")
            _set(prim, "gat:ordinal", Sdf.ValueTypeNames.Int, ordinal)
            _set(prim, "gat:kind", Sdf.ValueTypeNames.Token, rel["kind"])
            _write_optional_int(prim, "gat:sourceRef", rel["source_ref"], Sdf)
            source = _entity_key(_as_mapping(rel["source"], "relationship source"))
            target = _entity_key(_as_mapping(rel["target"], "relationship target"))
            prim.CreateRelationship("gat:source", custom=True).SetTargets([entity_paths[source]])
            prim.CreateRelationship("gat:target", custom=True).SetTargets([entity_paths[target]])

        constraints_scope = UsdGeom.Scope.Define(stage, "/GAT/State/Constraints").GetPrim()
        for ordinal, constraint in enumerate(_as_list(module["constraints"], "constraints")):
            prim = stage.DefinePrim(constraints_scope.GetPath().AppendChild(f"C_{ordinal:04d}"), "Scope")
            _set(prim, "gat:ordinal", Sdf.ValueTypeNames.Int, ordinal)
            _set(prim, "gat:constraint", Sdf.ValueTypeNames.String, _json(constraint))

        belief_prim = stage.DefinePrim("/GAT/State/Belief", "Scope")
        raw_variables = _as_list(belief["raw_variables"], "raw variables")
        raw_targets = []
        for var_value in raw_variables:
            var = _as_mapping(var_value, "raw variable")
            owner = _entity_key(_as_mapping(var["entity"], "raw variable owner"))
            raw_targets.append(quantity_paths[(owner[0], owner[1], str(var["quantity"]))])
        belief_prim.CreateRelationship("gat:rawVariables", custom=True).SetTargets(raw_targets)
        _set(
            belief_prim,
            "gat:rawMean",
            Sdf.ValueTypeNames.DoubleArray,
            [float(value) for value in _as_list(belief["mean"], "belief mean")],
        )
        covariance = _as_mapping(belief["covariance"], "belief covariance")
        _set(belief_prim, "gat:covarianceStorage", Sdf.ValueTypeNames.Token, covariance["storage"])
        _set(belief_prim, "gat:covarianceDimension", Sdf.ValueTypeNames.Int, covariance["dimension"])
        _set(
            belief_prim,
            "gat:rawCovariance",
            Sdf.ValueTypeNames.DoubleArray,
            [float(value) for value in _as_list(covariance["values"], "covariance values")],
        )

        provenance_prim = stage.DefinePrim("/GAT/State/Provenance", "Scope")
        _set(
            provenance_prim,
            "gat:traceEvents",
            Sdf.ValueTypeNames.String,
            _json(provenance["trace_events"]),
        )
        _write_execution_ledger(stage, state, resolved_ledger, Sdf)

        if include_geometry:
            _write_geometry_view(stage, world, entity_paths, Gf, Sdf, UsdGeom)

        output.parent.mkdir(parents=True, exist_ok=True)
        if not stage.GetRootLayer().Export(str(output)):
            raise OpenUsdError(f"OpenUSD could not export {output}")
    except (OpenUsdError, SnapshotError):
        raise
    except Exception as exc:
        raise OpenUsdError(f"could not write OpenUSD state carrier: {exc}") from exc
    return str(integrity["digest"])


def read_openusd(
    path: str | Path,
    *,
    limits: OpenUsdReadLimits = DEFAULT_OPENUSD_READ_LIMITS,
    trusted_public_keys: Mapping[str, bytes] | None = None,
    require_signature: bool = False,
) -> OpenUsdLoadResult:
    """Read, authenticate when requested, and reconstruct an OpenUSD world."""
    _, _, Usd, _ = _pxr()
    source = Path(path)
    if source.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise OpenUsdError("OpenUSD carrier path must end in .usd, .usda, or .usdc")
    try:
        file_size = source.stat().st_size
        if file_size > limits.max_file_bytes:
            raise OpenUsdError(
                f"OpenUSD root layer exceeds {limits.max_file_bytes} byte limit"
            )
        stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
        if stage is None:
            raise OpenUsdError(f"OpenUSD could not open {source}")
        _enforce_composed_prim_limit(stage, limits.max_composed_prims)
        root = stage.GetDefaultPrim()
        if not root:
            raise OpenUsdError("OpenUSD carrier has no default prim")
        if _attr(root, "gat:carrierFormat") != OPENUSD_CARRIER_FORMAT:
            raise OpenUsdError("unsupported GAT OpenUSD carrier format")
        carrier_version = int(_attr(root, "gat:carrierVersion"))
        if carrier_version not in OPENUSD_SUPPORTED_VERSIONS:
            raise OpenUsdError(
                f"unsupported GAT OpenUSD carrier version {carrier_version}"
            )
        budget = _ReadBudget(limits)
        state = stage.GetPrimAtPath(root.GetPath().AppendChild("State"))
        if not state or not bool(_attr(state, "gat:authoritative")):
            raise OpenUsdError("OpenUSD carrier has no authoritative State prim")

        entities_scope = _child(stage, state, "Entities")
        entities: list[dict[str, object]] = []
        quantity_by_path: dict[str, dict[str, object]] = {}
        for entity_prim in _ordered_children(
            entities_scope, limits.max_entities, "entities"
        ):
            eid = _entity_record(entity_prim)
            placement = None
            if bool(_attr(entity_prim, "gat:hasPlacement")):
                origin = list(_attr(entity_prim, "gat:placementOrigin"))
                if len(origin) != 3:
                    raise OpenUsdError("entity placement origin must contain three values")
                placement = {
                    "x": float(origin[0]),
                    "y": float(origin[1]),
                    "z": float(origin[2]),
                    "angle": float(_attr(entity_prim, "gat:placementAngle")),
                }
            quantities_scope = _child(stage, entity_prim, "Quantities")
            slots: list[dict[str, object]] = []
            quantity_prims = _ordered_children(
                quantities_scope, limits.max_quantities, "quantities per entity"
            )
            budget.quantities += len(quantity_prims)
            if budget.quantities > limits.max_quantities:
                raise OpenUsdError(
                    f"OpenUSD quantity count exceeds {limits.max_quantities}"
                )
            for quantity_prim in quantity_prims:
                quantity = str(_attr(quantity_prim, "gat:quantity"))
                owner_targets = _targets(quantity_prim, "gat:owner", count=1)
                owner_prim = stage.GetPrimAtPath(owner_targets[0])
                owner = _entity_record(owner_prim)
                var = {"entity": owner, "quantity": quantity}
                slot = {
                    "var": var,
                    "role": str(_attr(quantity_prim, "gat:role")),
                    "unit": str(_attr(quantity_prim, "gat:unit")),
                    "prior_mu": float(_attr(quantity_prim, "gat:priorMu")),
                    "prior_sigma": float(_attr(quantity_prim, "gat:priorSigma")),
                    "expr": budget.json(
                        _attr(quantity_prim, "gat:expression"), "quantity expression"
                    ),
                    "source_ref": _read_optional_int(quantity_prim, "gat:sourceRef"),
                }
                slots.append(slot)
                quantity_by_path[str(quantity_prim.GetPath())] = var
            entities.append(
                {
                    "id": eid,
                    "name": str(_attr(entity_prim, "gat:name")),
                    "attrs": budget.json(
                        _attr(entity_prim, "gat:attributes"), "entity attributes"
                    ),
                    "placement": placement,
                    "source_ref": _read_optional_int(entity_prim, "gat:sourceRef"),
                    "slots": slots,
                }
            )

        relationships_scope = _child(stage, state, "Relationships")
        relationships: list[dict[str, object]] = []
        for rel_prim in _ordered_children(
            relationships_scope, limits.max_relationships, "relationships"
        ):
            source_path = _targets(rel_prim, "gat:source", count=1)[0]
            target_path = _targets(rel_prim, "gat:target", count=1)[0]
            relationships.append(
                {
                    "kind": str(_attr(rel_prim, "gat:kind")),
                    "source": _entity_record(stage.GetPrimAtPath(source_path)),
                    "target": _entity_record(stage.GetPrimAtPath(target_path)),
                    "source_ref": _read_optional_int(rel_prim, "gat:sourceRef"),
                }
            )

        constraints_scope = _child(stage, state, "Constraints")
        constraints = [
            budget.json(_attr(prim, "gat:constraint"), "constraint")
            for prim in _ordered_children(
                constraints_scope, limits.max_constraints, "constraints"
            )
        ]

        belief_prim = _child(stage, state, "Belief")
        raw_variables = []
        raw_targets = _targets(belief_prim, "gat:rawVariables")
        if len(raw_targets) > limits.max_raw_variables:
            raise OpenUsdError(
                f"OpenUSD raw variable count exceeds {limits.max_raw_variables}"
            )
        dimension = int(_attr(belief_prim, "gat:covarianceDimension"))
        if dimension < 0 or dimension > limits.max_raw_variables:
            raise OpenUsdError("OpenUSD covariance dimension exceeds the read policy")
        expected_covariance_values = dimension * dimension
        if expected_covariance_values > limits.max_covariance_values:
            raise OpenUsdError(
                "OpenUSD covariance exceeds the configured dense-value budget"
            )
        if dimension != len(raw_targets):
            raise OpenUsdError("OpenUSD belief dimension differs from raw variable targets")
        for target in raw_targets:
            target_prim = stage.GetPrimAtPath(target)
            key = str(target_prim.GetPath())
            if key not in quantity_by_path:
                # The target may be path-translated by composition; read it
                # directly instead of trusting its original namespace.
                quantity = str(_attr(target_prim, "gat:quantity"))
                owner_path = _targets(target_prim, "gat:owner", count=1)[0]
                quantity_by_path[key] = {
                    "entity": _entity_record(stage.GetPrimAtPath(owner_path)),
                    "quantity": quantity,
                }
            raw_variables.append(quantity_by_path[key])

        provenance_prim = _child(stage, state, "Provenance")
        mean_values = _attr(belief_prim, "gat:rawMean")
        covariance_values = _attr(belief_prim, "gat:rawCovariance")
        if len(mean_values) != dimension:
            raise OpenUsdError("OpenUSD raw mean length differs from belief dimension")
        if len(covariance_values) != expected_covariance_values:
            raise OpenUsdError("OpenUSD covariance length differs from belief dimension")
        trace_events = budget.json(
            _attr(provenance_prim, "gat:traceEvents"), "trace events"
        )
        if not isinstance(trace_events, list):
            raise OpenUsdError("trace events must decode to an array")
        if len(trace_events) > limits.max_trace_events:
            raise OpenUsdError(
                f"OpenUSD trace event count exceeds {limits.max_trace_events}"
            )
        document: dict[str, object] = {
            "format": str(_attr(state, "gat:snapshotFormat")),
            "schema_version": int(_attr(state, "gat:snapshotVersion")),
            "runtime_contract": str(_attr(state, "gat:runtimeContract")),
            "payload": {
                "module": {
                    "meta": budget.json(
                        _attr(state, "gat:moduleMeta"), "module metadata"
                    ),
                    "entities": entities,
                    "relationships": relationships,
                    "constraints": constraints,
                },
                "belief": {
                    "raw_variables": raw_variables,
                    "mean": [float(value) for value in mean_values],
                    "covariance": {
                        "storage": str(_attr(belief_prim, "gat:covarianceStorage")),
                        "dimension": dimension,
                        "values": [float(value) for value in covariance_values],
                    },
                },
                "provenance": {"trace_events": trace_events},
                "source_module_digest": str(_attr(state, "gat:sourceModuleDigest")),
                "source_world_digest": str(_attr(state, "gat:sourceWorldDigest")),
                "source_configuration_digest": str(
                    _attr(state, "gat:sourceConfigurationDigest")
                ),
            },
            "integrity": {
                "algorithm": str(_attr(state, "gat:integrityAlgorithm")),
                "digest": str(_attr(state, "gat:integrityDigest")),
            },
        }
        ledger = (
            _read_execution_ledger(stage, state, budget)
            if carrier_version >= 3
            else None
        )
        signature = _read_and_verify_signature(
            root,
            document,
            carrier_version,
            ledger,
            trusted_public_keys,
            require_signature,
        )
        restored = reconstruct_snapshot(document)
        if (
            ledger is not None
            and ledger.events[-1].result_world_digest != restored.world.digest()
        ):
            raise OpenUsdError(
                "embedded execution ledger head does not describe the restored world"
            )
        return OpenUsdLoadResult(
            restored.world,
            restored.trace_events,
            restored.snapshot_digest,
            carrier_version,
            signature,
            ledger,
        )
    except (OpenUsdError, SnapshotError):
        raise
    except Exception as exc:
        raise OpenUsdError(f"could not read OpenUSD state carrier: {exc}") from exc


def migrate_openusd(
    source: str | Path,
    destination: str | Path,
    *,
    signing_key: OpenUsdKeyPair | None = None,
    trusted_public_keys: Mapping[str, bytes] | None = None,
    require_source_signature: bool = False,
    limits: OpenUsdReadLimits = DEFAULT_OPENUSD_READ_LIMITS,
    include_geometry: bool = True,
) -> OpenUsdMigrationReport:
    """Canonicalize a supported older carrier into the current version.

    A signed source is never silently stripped or re-blessed: it must verify
    under a supplied trusted key, and the destination must be signed anew.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise OpenUsdError("migration source and destination must differ")
    loaded = read_openusd(
        source_path,
        limits=limits,
        trusted_public_keys=trusted_public_keys,
        require_signature=require_source_signature,
    )
    if loaded.signature.present:
        if not loaded.signature.verified:
            raise OpenUsdError("signed migration source must verify under a trusted key")
        if signing_key is None:
            raise OpenUsdError("signed migration source requires a destination signing key")
    migrated_ledger = loaded.ledger or ExecutionLedger.genesis(
        loaded.world,
        {
            "checkpoint": "openusd-carrier-migration",
            "source_carrier_version": loaded.carrier_version,
            "source_snapshot_digest": loaded.snapshot_digest,
        },
    )
    digest = write_openusd(
        loaded.world,
        destination_path,
        loaded.trace_events,
        include_geometry=include_geometry,
        signing_key=signing_key,
        ledger=migrated_ledger,
    )
    return OpenUsdMigrationReport(
        loaded.carrier_version,
        OPENUSD_CARRIER_VERSION,
        digest,
        migrated_ledger.head,
        loaded.signature,
        None if signing_key is None else signing_key.key_id,
    )


def _write_execution_ledger(stage, state, ledger: ExecutionLedger, Sdf) -> None:
    """Author the ledger as inspectable event prims beneath authoritative State."""
    document = ledger.to_dict()
    integrity = _as_mapping(document["integrity"], "ledger integrity")
    ledger_prim = stage.DefinePrim(state.GetPath().AppendChild("Ledger"), "Scope")
    _set(ledger_prim, "gat:authoritative", Sdf.ValueTypeNames.Bool, True)
    _set(ledger_prim, "gat:ledgerFormat", Sdf.ValueTypeNames.String, document["format"])
    _set(
        ledger_prim,
        "gat:ledgerSchemaVersion",
        Sdf.ValueTypeNames.Int,
        document["schema_version"],
    )
    _set(
        ledger_prim,
        "gat:ledgerRuntimeContract",
        Sdf.ValueTypeNames.String,
        document["runtime_contract"],
    )
    _set(
        ledger_prim,
        "gat:ledgerIntegrityAlgorithm",
        Sdf.ValueTypeNames.String,
        integrity["algorithm"],
    )
    _set(ledger_prim, "gat:ledgerHead", Sdf.ValueTypeNames.String, integrity["head"])
    _set(
        ledger_prim,
        "gat:ledgerEventCount",
        Sdf.ValueTypeNames.Int64,
        len(ledger.events),
    )
    events_prim = stage.DefinePrim(ledger_prim.GetPath().AppendChild("Events"), "Scope")
    for event in ledger.events:
        record = event.to_dict()
        prim = stage.DefinePrim(
            events_prim.GetPath().AppendChild(f"E_{event.seq:08d}"), "Scope"
        )
        _set(prim, "gat:ordinal", Sdf.ValueTypeNames.Int64, event.seq)
        _set(prim, "gat:eventKind", Sdf.ValueTypeNames.Token, event.kind)
        _set(prim, "gat:operation", Sdf.ValueTypeNames.String, _json(record["operation"]))
        _set(prim, "gat:provenance", Sdf.ValueTypeNames.String, _json(record["provenance"]))
        _set(
            prim,
            "gat:priorWorldDigest",
            Sdf.ValueTypeNames.String,
            event.prior_world_digest,
        )
        _set(
            prim,
            "gat:resultWorldDigest",
            Sdf.ValueTypeNames.String,
            event.result_world_digest,
        )
        _set(
            prim,
            "gat:verificationPresent",
            Sdf.ValueTypeNames.Bool,
            event.verification is not None,
        )
        if event.verification is not None:
            _set(
                prim,
                "gat:verification",
                Sdf.ValueTypeNames.String,
                _json(event.verification),
            )
            _set(
                prim,
                "gat:verificationDigest",
                Sdf.ValueTypeNames.String,
                event.verification_digest,
            )
        _set(
            prim,
            "gat:errorPresent",
            Sdf.ValueTypeNames.Bool,
            event.error_type is not None,
        )
        if event.error_type is not None:
            _set(prim, "gat:errorType", Sdf.ValueTypeNames.String, event.error_type)
            _set(prim, "gat:errorMessage", Sdf.ValueTypeNames.String, event.error_message)
            _set(prim, "gat:errorDigest", Sdf.ValueTypeNames.String, event.error_digest)
        _set(prim, "gat:previousHash", Sdf.ValueTypeNames.String, event.previous_hash)
        _set(prim, "gat:eventHash", Sdf.ValueTypeNames.String, event.event_hash)


def _read_execution_ledger(stage, state, budget: _ReadBudget) -> ExecutionLedger:
    ledger_prim = _child(stage, state, "Ledger")
    if not bool(_attr(ledger_prim, "gat:authoritative")):
        raise OpenUsdError("OpenUSD execution ledger is not authoritative")
    count = int(_attr(ledger_prim, "gat:ledgerEventCount"))
    if count <= 0 or count > budget.limits.max_ledger_events:
        raise OpenUsdError("OpenUSD ledger event count exceeds the read policy")
    events_scope = _child(stage, ledger_prim, "Events")
    event_prims = _ordered_children(
        events_scope, budget.limits.max_ledger_events, "ledger events"
    )
    if len(event_prims) != count:
        raise OpenUsdError("OpenUSD ledger event count differs from authored events")
    events: list[dict[str, object]] = []
    for seq, prim in enumerate(event_prims):
        verification_present = bool(_attr(prim, "gat:verificationPresent"))
        error_present = bool(_attr(prim, "gat:errorPresent"))
        verification = (
            budget.json(_attr(prim, "gat:verification"), f"ledger event {seq} verification")
            if verification_present
            else None
        )
        verification_digest = (
            str(_attr(prim, "gat:verificationDigest"))
            if verification_present
            else None
        )
        error_type = (
            budget.text(_attr(prim, "gat:errorType"), f"ledger event {seq} error type")
            if error_present
            else None
        )
        error_message = (
            budget.text(_attr(prim, "gat:errorMessage"), f"ledger event {seq} error message")
            if error_present
            else None
        )
        error_digest = (
            str(_attr(prim, "gat:errorDigest")) if error_present else None
        )
        events.append(
            {
                "seq": seq,
                "kind": str(_attr(prim, "gat:eventKind")),
                "operation": budget.json(
                    _attr(prim, "gat:operation"), f"ledger event {seq} operation"
                ),
                "provenance": budget.json(
                    _attr(prim, "gat:provenance"), f"ledger event {seq} provenance"
                ),
                "prior_world_digest": str(_attr(prim, "gat:priorWorldDigest")),
                "result_world_digest": str(_attr(prim, "gat:resultWorldDigest")),
                "verification": verification,
                "verification_digest": verification_digest,
                "error_type": error_type,
                "error_message": error_message,
                "error_digest": error_digest,
                "previous_hash": str(_attr(prim, "gat:previousHash")),
                "event_hash": str(_attr(prim, "gat:eventHash")),
            }
        )
    document = {
        "format": str(_attr(ledger_prim, "gat:ledgerFormat")),
        "schema_version": int(_attr(ledger_prim, "gat:ledgerSchemaVersion")),
        "runtime_contract": str(_attr(ledger_prim, "gat:ledgerRuntimeContract")),
        "events": events,
        "integrity": {
            "algorithm": str(_attr(ledger_prim, "gat:ledgerIntegrityAlgorithm")),
            "head": str(_attr(ledger_prim, "gat:ledgerHead")),
        },
    }
    try:
        return ExecutionLedger.from_dict(document)
    except LedgerError as exc:
        raise OpenUsdError(f"invalid embedded execution ledger: {exc}") from exc


def _write_geometry_view(stage, world, entity_paths, Gf, Sdf, UsdGeom) -> None:
    view = UsdGeom.Scope.Define(stage, "/GAT/View").GetPrim()
    _set(view, "gat:authoritative", Sdf.ValueTypeNames.Bool, False)
    _set(view, "gat:worldDigest", Sdf.ValueTypeNames.String, world.digest())
    entities_scope = UsdGeom.Scope.Define(stage, "/GAT/View/Entities").GetPrim()
    geometric_classes = {"IfcWall", "IfcSpace", "IfcDoor"}
    if not any(
        entity.placement is not None and eid.ifc_class in geometric_classes
        for eid, entity in world.module.entities.items()
    ):
        return
    scene = derive_scene(world)
    colors = {
        "IfcWall": Gf.Vec3f(0.72, 0.72, 0.75),
        "IfcSpace": Gf.Vec3f(0.30, 0.60, 0.95),
        "IfcDoor": Gf.Vec3f(0.55, 0.32, 0.16),
    }
    for element in scene.elements:
        eid = element.entity_id
        key = (eid.ifc_class, eid.global_id)
        path = entities_scope.GetPath().AppendChild(f"E_{element.row:04d}_{_fragment(*key)}")
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*element.box.origin))
        xform.AddRotateZOp().Set(math.degrees(element.box.angle))
        prim = xform.GetPrim()
        _set(prim, "gat:ifcClass", Sdf.ValueTypeNames.String, eid.ifc_class)
        _set(prim, "gat:globalId", Sdf.ValueTypeNames.String, eid.global_id)
        _set(prim, "gat:name", Sdf.ValueTypeNames.String, element.name)
        _set(prim, "gat:worldDigest", Sdf.ValueTypeNames.String, scene.version)
        prim.CreateRelationship("gat:source", custom=True).SetTargets([entity_paths[key]])
        cube = UsdGeom.Cube.Define(stage, path.AppendChild("Bounds"))
        cube.CreateSizeAttr(2.0)
        half = tuple(0.5 * float(value) for value in element.box.extents)
        cube.AddTranslateOp().Set(Gf.Vec3d(*half))
        cube.AddScaleOp().Set(Gf.Vec3d(*half))
        cube.CreateDisplayColorAttr([colors.get(eid.ifc_class, Gf.Vec3f(0.5))])
        if eid.ifc_class == "IfcSpace":
            cube.CreatePurposeAttr(UsdGeom.Tokens.guide)


def _sign_document(
    document: Mapping[str, object],
    carrier_version: int,
    signing_key: OpenUsdKeyPair,
    ledger: ExecutionLedger | None,
) -> str:
    Ed25519PrivateKey, _, _ = _crypto()
    try:
        private = Ed25519PrivateKey.from_private_bytes(signing_key.private_key)
        derived_public = private.public_key().public_bytes_raw()
    except (TypeError, ValueError) as exc:
        raise OpenUsdError(f"invalid Ed25519 signing key: {exc}") from exc
    if not hmac.compare_digest(derived_public, signing_key.public_key):
        raise OpenUsdError("Ed25519 private and public key material do not match")
    signature = private.sign(_signature_material(document, carrier_version, ledger))
    return base64.b64encode(signature).decode("ascii")


def _read_and_verify_signature(
    root,
    document: Mapping[str, object],
    carrier_version: int,
    ledger: ExecutionLedger | None,
    trusted_public_keys: Mapping[str, bytes] | None,
    require_signature: bool,
) -> OpenUsdSignatureInfo:
    if carrier_version == 1:
        if require_signature:
            raise OpenUsdError("carrier version 1 has no provenance signature")
        return OpenUsdSignatureInfo(False)

    present = bool(_attr(root, "gat:signaturePresent"))
    if not present:
        if require_signature:
            raise OpenUsdError("trusted provenance requires a signed OpenUSD carrier")
        return OpenUsdSignatureInfo(False)

    algorithm = str(_attr(root, "gat:signatureAlgorithm"))
    key_id = str(_attr(root, "gat:signatureKeyId"))
    encoded = _attr(root, "gat:signature")
    if algorithm != OPENUSD_SIGNATURE_ALGORITHM:
        raise OpenUsdError(f"unsupported OpenUSD signature algorithm {algorithm!r}")
    if not isinstance(encoded, str):
        raise OpenUsdError("OpenUSD provenance signature must be base64 text")
    if not key_id or len(key_id) > 256:
        raise OpenUsdError("OpenUSD signature key id must contain 1..256 characters")
    if len(encoded) > 256:
        raise OpenUsdError("OpenUSD provenance signature text is unexpectedly large")
    try:
        signature = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise OpenUsdError("OpenUSD provenance signature is not valid base64") from exc
    if len(signature) != 64:
        raise OpenUsdError("Ed25519 provenance signature must contain 64 bytes")

    trusted_key = None if trusted_public_keys is None else trusted_public_keys.get(key_id)
    if trusted_key is None:
        if require_signature:
            raise OpenUsdError(f"no trusted public key is configured for {key_id!r}")
        return OpenUsdSignatureInfo(True, algorithm, key_id, False)
    if not isinstance(trusted_key, bytes) or len(trusted_key) != 32:
        raise OpenUsdError(f"trusted Ed25519 public key {key_id!r} must be 32 bytes")

    _, Ed25519PublicKey, InvalidSignature = _crypto()
    try:
        public = Ed25519PublicKey.from_public_bytes(trusted_key)
        public.verify(
            signature, _signature_material(document, carrier_version, ledger)
        )
    except InvalidSignature as exc:
        raise OpenUsdError(f"provenance signature verification failed for {key_id!r}") from exc
    except (TypeError, ValueError) as exc:
        raise OpenUsdError(f"invalid trusted Ed25519 public key {key_id!r}") from exc
    return OpenUsdSignatureInfo(True, algorithm, key_id, True)


def _signature_material(
    document: Mapping[str, object],
    carrier_version: int,
    ledger: ExecutionLedger | None,
) -> bytes:
    integrity = _as_mapping(document["integrity"], "snapshot integrity")
    material = {
        "domain": _SIGNATURE_DOMAINS.get(carrier_version),
        "carrier_format": OPENUSD_CARRIER_FORMAT,
        "carrier_version": carrier_version,
        "snapshot_format": document["format"],
        "snapshot_schema_version": document["schema_version"],
        "runtime_contract": document["runtime_contract"],
        "snapshot_digest": integrity["digest"],
    }
    if carrier_version >= 3:
        if ledger is None:
            raise OpenUsdError("carrier version 3 signature requires an execution ledger")
        material["ledger_format"] = ledger.to_dict()["format"]
        material["ledger_schema_version"] = ledger.to_dict()["schema_version"]
        material["ledger_head"] = ledger.head
    return _json(material).encode("utf-8")


def _crypto():
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except ImportError as exc:
        raise OpenUsdError(
            "signed OpenUSD provenance requires the optional cryptography runtime; "
            "install with `pip install 'gat-bim[openusd]'`"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature


def _pxr():
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom
    except ImportError as exc:
        raise OpenUsdError(
            "OpenUSD support requires Pixar's optional runtime; "
            "install with `pip install 'gat-bim[openusd]'`"
        ) from exc
    return Gf, Sdf, Usd, UsdGeom


def _set(prim, name: str, value_type, value: object) -> None:
    from pxr import Sdf

    attribute = prim.CreateAttribute(
        name, value_type, custom=True, variability=Sdf.VariabilityUniform
    )
    if not attribute.Set(value):
        raise OpenUsdError(f"could not author {prim.GetPath()}.{name}")


def _attr(prim, name: str) -> object:
    if not prim:
        raise OpenUsdError(f"invalid prim while reading {name}")
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        raise OpenUsdError(f"missing required attribute {prim.GetPath()}.{name}")
    value = attribute.Get()
    if value is None:
        raise OpenUsdError(f"attribute {prim.GetPath()}.{name} has no value")
    return value


def _targets(prim, name: str, count: int | None = None):
    relationship = prim.GetRelationship(name)
    if not relationship:
        raise OpenUsdError(f"missing required relationship {prim.GetPath()}.{name}")
    targets = relationship.GetTargets()
    if count is not None and len(targets) != count:
        raise OpenUsdError(
            f"relationship {prim.GetPath()}.{name} requires {count} target(s)"
        )
    return targets


def _child(stage, parent, name: str):
    prim = stage.GetPrimAtPath(parent.GetPath().AppendChild(name))
    if not prim:
        raise OpenUsdError(f"missing required prim {parent.GetPath()}/{name}")
    return prim


def _enforce_composed_prim_limit(stage, limit: int) -> None:
    for count, _ in enumerate(stage.Traverse(), start=1):
        if count > limit:
            raise OpenUsdError(f"OpenUSD composed prim count exceeds {limit}")


def _ordered_children(parent, limit: int, label: str) -> list:
    children = list(parent.GetChildren())
    if len(children) > limit:
        raise OpenUsdError(f"OpenUSD {label} count exceeds {limit}")
    decorated = []
    for child in children:
        ordinal = int(_attr(child, "gat:ordinal"))
        decorated.append((ordinal, child))
    decorated.sort(key=lambda item: item[0])
    if [ordinal for ordinal, _ in decorated] != list(range(len(decorated))):
        raise OpenUsdError(f"non-contiguous or duplicate ordinals beneath {parent.GetPath()}")
    return [child for _, child in decorated]


def _entity_record(prim) -> dict[str, str]:
    if not prim:
        raise OpenUsdError("relationship targets an invalid entity prim")
    return {
        "ifc_class": str(_attr(prim, "gat:ifcClass")),
        "global_id": str(_attr(prim, "gat:globalId")),
    }


def _write_optional_int(prim, stem: str, value: object, Sdf) -> None:
    present = value is not None
    _set(prim, f"{stem}Present", Sdf.ValueTypeNames.Bool, present)
    if present:
        _set(prim, stem, Sdf.ValueTypeNames.Int64, int(value))


def _read_optional_int(prim, stem: str) -> int | None:
    if not bool(_attr(prim, f"{stem}Present")):
        return None
    return int(_attr(prim, stem))


def _fragment(*parts: str) -> str:
    material = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:12]


def _entity_key(record: Mapping[str, object]) -> tuple[str, str]:
    return str(record["ifc_class"]), str(record["global_id"])


def _json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise OpenUsdError(f"value cannot be represented in OpenUSD JSON field: {exc}") from exc


def _decode_json(value: object, label: str) -> object:
    if not isinstance(value, str):
        raise OpenUsdError(f"{label} must be a JSON string")

    def reject_constant(token: str) -> object:
        raise OpenUsdError(f"{label} contains non-finite number {token!r}")

    try:
        return json.loads(value, parse_constant=reject_constant)
    except OpenUsdError:
        raise
    except json.JSONDecodeError as exc:
        raise OpenUsdError(f"{label} is invalid JSON: {exc}") from exc


def _as_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OpenUsdError(f"{label} must be an object")
    return value


def _as_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise OpenUsdError(f"{label} must be an array")
    return value


__all__ = [
    "DEFAULT_OPENUSD_READ_LIMITS",
    "OPENUSD_CARRIER_FORMAT",
    "OPENUSD_CARRIER_VERSION",
    "OPENUSD_SIGNATURE_ALGORITHM",
    "OPENUSD_SUPPORTED_VERSIONS",
    "OpenUsdKeyPair",
    "OpenUsdLoadResult",
    "OpenUsdMigrationReport",
    "OpenUsdReadLimits",
    "OpenUsdSignatureInfo",
    "generate_openusd_keypair",
    "migrate_openusd",
    "openusd_available",
    "read_openusd",
    "write_openusd",
]
