# Figures

These images are rendered from **real Navigation v2 world models** saved during
real-ROM acceptance runs. They are produced by code, not drawn by hand, and no
game RAM / walkthrough data is involved in rendering.

| File | Source | What it shows |
|---|---|---|
| `navigation_v2_oaks_lab_map.png` | `artifacts/nav_v2/g2/navigation` (G2 acceptance: blind exploration of Oak's Lab, real ROM + PyBoy) | 353 logical cells, 79 confirmed walkable, 56 blocked, 1 exit, 97.53% coverage. Legend: green = confirmed walkable, red = blocked/wall, orange = temporary/uncertain obstacle, yellow = exit/interactive, grey = seen but unconfirmed, white = player, purple = planned path, cyan = target. |
| `navigation_v2_world_graph.png` | `artifacts/nav_v2/g4/navigation` (G4 unattended multi-area run, no LLM) | The persistent cross-area world graph: 4 areas / 2 connections, each node records walkable-cell counts and exploration state. |

Regenerate them from a saved world model:

```powershell
.\.venv\Scripts\python.exe scripts\nav_v2_map_gallery.py `
    --world-dir artifacts\nav_v2\g2\navigation --out artifacts\nav_v2\map_gallery
```

The local `artifacts/` evidence directory is gitignored (it contains screenshots
and run logs); only the final rendered figures are committed here.
