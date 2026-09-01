# GAT Evidence and Assurance for Blender/Bonsai

This extension is the read-only presentation edge of the GAT workflow.
It loads an `acceptance` response produced by `gat-headless`, shows the
case disposition and next evidence request, and colors matching Blender or
Bonsai objects.

It does not edit IFC data, execute transformations, approve field work, or
recompute probabilities inside Blender. Objects are matched by name or by a
`gat_entity_name` custom property.

Build with Blender 4.2 or newer:

```text
blender --command extension build --source-dir integrations/blender/gat_assurance
```

Install the resulting ZIP through **Preferences > Get Extensions > Install
from Disk**. The extension requests only local file access.
