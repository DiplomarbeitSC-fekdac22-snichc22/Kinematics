"""Export top-level STEP shapes as STL files from the FreeCAD Python console.

Run with FreeCADCmd, not ordinary Python:

    FreeCADCmd simulation/tools/export_step_parts.py INPUT.step OUTPUT_DIRECTORY

The functional Webots model does not depend on these visual meshes. Exported
meshes can replace the primitive Shape nodes after solids are assigned to the
correct articulated links; collision boxes should remain primitive.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import FreeCAD
import Import
import Mesh
import Part


def _safe_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return cleaned or fallback


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "Usage: FreeCADCmd export_step_parts.py INPUT.step OUTPUT_DIRECTORY",
            file=sys.stderr,
        )
        return 2

    source = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    destination.mkdir(parents=True, exist_ok=True)

    document = FreeCAD.newDocument("robot_step_export")
    Import.insert(str(source), document.Name)
    document.recompute()

    exported = 0
    for object_index, obj in enumerate(document.Objects, start=1):
        shape = getattr(obj, "Shape", None)
        if shape is None or shape.isNull():
            continue

        solids = list(shape.Solids)
        if not solids:
            solids = [shape]

        base_name = _safe_name(
            getattr(obj, "Label", ""),
            f"object_{object_index:03d}",
        )

        for solid_index, solid in enumerate(solids, start=1):
            feature = document.addObject("Part::Feature", "export_feature")
            feature.Shape = Part.makeCompound([solid])
            suffix = f"_{solid_index:03d}" if len(solids) > 1 else ""
            output = destination / f"{base_name}{suffix}.stl"
            Mesh.export([feature], str(output))
            document.removeObject(feature.Name)
            exported += 1
            print(output)

    FreeCAD.closeDocument(document.Name)
    print(f"Exported {exported} STEP shape(s).")
    return 0 if exported else 1


if __name__ == "__main__":
    raise SystemExit(main())
