"""GAT Evidence and Assurance Blender extension.

This UI consumes read-only responses from ``gat-headless``.  It never edits
IFC data or executes a proposed transformation.  Bonsai objects can expose a
``gat_entity_name`` custom property when their Blender object name differs
from the IFC/GAT entity name.
"""

from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator, Panel

from .bridge import load_response


bl_info = {
    "name": "GAT Evidence and Assurance",
    "author": "Notation Systems",
    "version": (0, 1, 0),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > GAT",
    "description": "Review evidence-bound BIM and beam assurance decisions",
    "category": "3D View",
}


class GAT_OT_load_workflow_response(Operator):
    bl_idname = "gat.load_workflow_response"
    bl_label = "Load GAT Decision"
    bl_description = "Load a read-only workflow or beam response from gat-headless"

    def execute(self, context):
        scene = context.scene
        try:
            view = load_response(bpy.path.abspath(scene.gat_response_path))
        except (OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        scene.gat_case_id = view.case_id
        scene.gat_subject = view.subject
        scene.gat_disposition = view.disposition
        scene.gat_reason = "; ".join(view.reasons)
        scene.gat_next_evidence = (
            ""
            if not view.requests
            else "; ".join(
                f"{request.action}: {request.target}" for request in view.requests
            )
        )
        scene.gat_world_digest = view.world_digest
        scene.gat_method = str(getattr(view, "method", ""))
        scene.gat_oracle = str(getattr(view, "oracle_id", ""))
        prior_capacity = getattr(view, "prior_capacity_n_m", None)
        revised_capacity = getattr(view, "revised_capacity_n_m", None)
        scene.gat_capacity_change = (
            ""
            if prior_capacity is None or revised_capacity is None
            else (
                f"{prior_capacity / 1000.0:.1f} -> "
                f"{revised_capacity / 1000.0:.1f} kN*m"
            )
        )

        targets = set(view.overlay_subjects)
        colored = 0
        for obj in bpy.data.objects:
            entity_name = str(obj.get("gat_entity_name", obj.name))
            if entity_name in targets:
                obj.color = view.color
                colored += 1
        self.report(
            {"INFO"},
            f"Loaded {view.disposition}; highlighted {colored} matching objects",
        )
        return {"FINISHED"}


class GAT_PT_evidence_assurance(Panel):
    bl_label = "Evidence & Assurance"
    bl_idname = "GAT_PT_evidence_assurance"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "GAT"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        layout.prop(scene, "gat_response_path", text="Decision file")
        layout.operator(GAT_OT_load_workflow_response.bl_idname)
        if not scene.gat_disposition:
            layout.label(text="No GAT decision loaded")
            return
        box = layout.box()
        box.label(text=f"{scene.gat_disposition}: {scene.gat_subject}")
        box.label(text=scene.gat_reason)
        if scene.gat_next_evidence:
            evidence = layout.box()
            evidence.label(text="Next evidence")
            evidence.label(text=scene.gat_next_evidence)
        if scene.gat_method:
            calculation = layout.box()
            calculation.label(text="Validated beam calculation")
            calculation.label(text=scene.gat_method)
            calculation.label(text=f"Capacity: {scene.gat_capacity_change}")
            calculation.label(text=f"Oracle: {scene.gat_oracle}")
        layout.label(text=f"Case: {scene.gat_case_id}")
        layout.label(text=f"State: {scene.gat_world_digest[:12]}…")
        layout.label(text="Read-only: no IFC state was changed")


_CLASSES = (GAT_OT_load_workflow_response, GAT_PT_evidence_assurance)
_SCENE_PROPERTIES = {
    "gat_response_path": StringProperty(
        name="GAT response",
        description="Local gat-headless acceptance response",
        subtype="FILE_PATH",
    ),
    "gat_case_id": StringProperty(name="Case ID"),
    "gat_subject": StringProperty(name="Subject"),
    "gat_disposition": StringProperty(name="Disposition"),
    "gat_reason": StringProperty(name="Reason"),
    "gat_next_evidence": StringProperty(name="Next evidence"),
    "gat_world_digest": StringProperty(name="World digest"),
    "gat_method": StringProperty(name="Engineering method"),
    "gat_oracle": StringProperty(name="Validation oracle"),
    "gat_capacity_change": StringProperty(name="Capacity change"),
}


def register():
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    for name, prop in _SCENE_PROPERTIES.items():
        setattr(bpy.types.Scene, name, prop)


def unregister():
    for name in reversed(tuple(_SCENE_PROPERTIES)):
        delattr(bpy.types.Scene, name)
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
