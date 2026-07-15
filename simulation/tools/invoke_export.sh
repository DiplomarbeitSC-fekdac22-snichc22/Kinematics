#!/usr/bin/env bash
FreeCADCmd <<'PY'
import os
import runpy
import sys

root = os.getcwd()

sys.argv = [
    "export_step_parts.py",
    r"/home/christophschnitzer/Windows-SSD/Users/chris/OneDrive - HTBLA Kaindorf/Diplomarbeit/Robot/V5Baugruppe_20260701.step",
    os.path.join(root, "simulation", "meshes", "cad_export"),
]

runpy.run_path(
    os.path.join(
        root,
        "simulation",
        "tools",
        "export_step_parts.py",
    ),
    run_name="__main__",
)
PY