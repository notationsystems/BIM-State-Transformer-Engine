"""Versioned IFC source identity, separate from source location and state.

The gat-world-v1 hash algorithm is unchanged. New imports explicitly bind
their source bytes through tagged module metadata instead of a path string.
Historical modules retain their metadata and digest exactly as serialized.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import re

from gat.ir.core import Module
from gat.ir.printer import print_module

IFC_SOURCE_IDENTITY = "gat-ifc-content-v2"


def validate_identity_version(version: int) -> None:
    if type(version) is not int or version not in (1, 2):
        raise ValueError("IFC identity_version must be 1 (legacy) or 2 (content-bound)")


def bind_source_content(module: Module, source_sha256: str) -> Module:
    """Bind an import to the hash of the exact bytes parsed by its caller."""
    if not isinstance(source_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    return replace(module, meta={
        **module.meta,
        "source": f"sha256:{source_sha256}",
        "source_content_sha256": source_sha256,
        "source_identity_contract": IFC_SOURCE_IDENTITY,
    })


def semantic_model_digest(module: Module) -> str:
    """Fingerprint the printed computational IR, independent of source bytes.

This is not an authority to rebind evidence. IFC STEP source references and
file metadata are not part of this projection; EntityIds, names, quantities,
placements, units, relationships, constraints and adapter metadata are.
"""
    projected = replace(module, meta={
        key: value for key, value in module.meta.items()
        if key not in {"source", "source_content_sha256", "source_identity_contract"}
    })
    return hashlib.sha256(
        b"gat-semantic-ir-v1\n" + print_module(projected).encode("utf-8")
    ).hexdigest()
