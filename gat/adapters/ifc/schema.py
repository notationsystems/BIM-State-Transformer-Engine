"""Attribute-position maps for the supported IFC entity subset.

The parser is schema-agnostic; this table is the only place that knows
which argument position means what, for the ~25 entity types the v0
adapter lowers.  Anything not listed here survives parsing as an opaque
instance and round-trips verbatim.
"""

from __future__ import annotations

#: type name -> {attribute name: argument position}
SUPPORTED_ENTITIES: dict[str, dict[str, int]] = {
    "IFCPROJECT": {"GlobalId": 0, "Name": 2},
    "IFCBUILDING": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCBUILDINGSTOREY": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCWALL": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCWALLSTANDARDCASE": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCSPACE": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCOPENINGELEMENT": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCDOOR": {"GlobalId": 0, "Name": 2, "ObjectPlacement": 5},
    "IFCRELAGGREGATES": {"GlobalId": 0, "RelatingObject": 4, "RelatedObjects": 5},
    "IFCRELCONTAINEDINSPATIALSTRUCTURE": {
        "GlobalId": 0,
        "RelatedElements": 4,
        "RelatingStructure": 5,
    },
    "IFCRELVOIDSELEMENT": {
        "GlobalId": 0,
        "RelatingBuildingElement": 4,
        "RelatedOpeningElement": 5,
    },
    "IFCRELFILLSELEMENT": {
        "GlobalId": 0,
        "RelatingOpeningElement": 4,
        "RelatedBuildingElement": 5,
    },
    "IFCRELSPACEBOUNDARY": {
        "GlobalId": 0,
        "RelatingSpace": 4,
        "RelatedBuildingElement": 5,
        "InternalOrExternalBoundary": 8,
    },
    "IFCRELDEFINESBYPROPERTIES": {
        "GlobalId": 0,
        "RelatedObjects": 4,
        "RelatingPropertyDefinition": 5,
    },
    "IFCPROPERTYSET": {"GlobalId": 0, "Name": 2, "HasProperties": 4},
    "IFCPROPERTYSINGLEVALUE": {"Name": 0, "NominalValue": 2},
    "IFCELEMENTQUANTITY": {"GlobalId": 0, "Name": 2, "Quantities": 5},
    "IFCQUANTITYLENGTH": {"Name": 0, "Value": 3},
    "IFCQUANTITYAREA": {"Name": 0, "Value": 3},
    "IFCQUANTITYVOLUME": {"Name": 0, "Value": 3},
    "IFCLOCALPLACEMENT": {"PlacementRelTo": 0, "RelativePlacement": 1},
    "IFCAXIS2PLACEMENT3D": {"Location": 0, "Axis": 1, "RefDirection": 2},
    "IFCCARTESIANPOINT": {"Coordinates": 0},
    "IFCDIRECTION": {"DirectionRatios": 0},
    "IFCSIUNIT": {"UnitType": 1, "Prefix": 2, "Name": 3},
}

#: The building-element classes the adapter lowers to IR entities,
#: normalized to their canonical IFC class spelling.
PRODUCT_CLASSES: dict[str, str] = {
    "IFCBUILDINGSTOREY": "IfcBuildingStorey",
    "IFCWALL": "IfcWall",
    "IFCWALLSTANDARDCASE": "IfcWall",  # normalized: a wall is a wall
    "IFCSPACE": "IfcSpace",
    "IFCOPENINGELEMENT": "IfcOpeningElement",
    "IFCDOOR": "IfcDoor",
}
