# Interactive LEO Visualization

This app runs a local Dash server that animates Iridium satellites in 3D and overlays routing links with costs. It highlights Dijkstra paths for selected source/target satellites and exposes basic playback and routing stats.

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r interactive\requirements.txt
```

## Run

```bash
python interactive\app.py
```

Open http://127.0.0.1:8050 in your browser.

## Notes

- The Earth surface is a procedural texture built at runtime.
- Use the sidebar to select satellites, control playback, and view routing stats.
- Link labels show per-edge cost (ms). Hover links or satellites for details.
