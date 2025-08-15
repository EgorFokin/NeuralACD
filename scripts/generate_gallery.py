#!/usr/bin/env python3
"""
generate_gallery.py

Usage:
    python generate_gallery.py /path/to/out

Outputs: out/gallery.html
"""
import sys
from pathlib import Path
import json
import html
import os

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/visual")
STATS_JSON = ROOT / "results.json"
OUT_HTML = ROOT / "gallery.html"

def load_stats(stats_path: Path):
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def model_viewer_html(src):
    esc = html.escape(src)
    return f"""
<div style="display:flex; align-items:center; justify-content:center;">
  <model-viewer src="{esc}" camera-controls autoplay
                style="width:300px; height:300px; background:#222; border-radius:6px;">
  </model-viewer>
</div>
"""

def img_thumb_html(src):
    esc = html.escape(src)
    return f'<a href="{esc}" target="_blank"><img src="{esc}" style="width:300px; height:300px; object-fit:cover; border-radius:6px;"></a>'

def stats_table_html(stats, method):
    if method not in stats:
        return ''
    s = stats[method]
    conc = s.get("concavity", "")
    num  = s.get("num_parts", "")
    return f"""
    <table style="border-collapse:collapse; font-size:12px; margin:auto; margin-top:4px;">
      <tr><th style="border:1px solid #2a2a2a; padding:2px 6px;">Concav</th>
          <th style="border:1px solid #2a2a2a; padding:2px 6px;">NParts</th></tr>
      <tr><td style="border:1px solid #2a2a2a; padding:2px 6px;">{conc}</td>
          <td style="border:1px solid #2a2a2a; padding:2px 6px;">{num}</td></tr>
    </table>
    """

def generate_html(meshes, stats):
    html_head = """<!doctype html>
<html>
<head>
<title>ACD</title>
<meta charset="utf-8">
<script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
<style>
body { background:#111; color:#eee; font-family: sans-serif; padding:16px; }
table.gallery { width:100%; border-collapse:collapse; }
table.gallery th, table.gallery td { border:1px solid #2a2a2a; padding:8px; text-align:center; vertical-align:top; }
table.gallery th { background:#171717; font-weight:600; }
.model-col { font-weight:600; width:160px; }
</style>
</head>
<body>
<a href="../../" style="color:#4af; text-decoration:none; font-weight:bold;">&larr; Back to Home</a>
<br><br>
<table class="gallery">
<thead>
<tr>
  <th>Name</th>
  <th>Original</th>
  <th>CutProb</th>
  <th>NeuralACD</th>
  <th>CoACD</th>
</tr>
</thead>
<tbody>
"""
    html_tail = """
</tbody>
</table>
</body></html>
"""

    rows = []
    for mesh in sorted(meshes):
        mesh_stats = stats.get(mesh, {})

        # Original GLB + stats
        orig_path = os.path.join(mesh, "original.glb")
        orig_cell = model_viewer_html(orig_path) + stats_table_html(mesh_stats, "original")

        # Prediction PNG
        pred_path = os.path.join(mesh, "neural_acd", "prediction.png")
        if (ROOT / pred_path).exists():
            pred_cell = img_thumb_html(pred_path)
        else:
            pred_cell = '<div style="color:#777; width:200px; height:200px; background:#222; border-radius:6px; display:flex; align-items:center; justify-content:center;">missing</div>'

        # NeuralACD GLB + stats
        neural_glb = os.path.join(mesh, "neural_acd", "decomposed.glb")
        neural_cell = model_viewer_html(neural_glb) + stats_table_html(mesh_stats, "neural_acd")

        # CoACD GLB + stats
        coacd_glb = os.path.join(mesh, "coacd", "decomposition.glb")
        coacd_cell = model_viewer_html(coacd_glb) + stats_table_html(mesh_stats, "coacd")

        rows.append(
            "<tr>"
            f"<td class='model-col'>{html.escape(mesh)}</td>"
            f"<td>{orig_cell}</td>"
            f"<td>{pred_cell}</td>"
            f"<td>{neural_cell}</td>"
            f"<td>{coacd_cell}</td>"
            "</tr>"
        )

    return html_head + "\n".join(rows) + html_tail

def main():
    meshes = [m for m in os.listdir(ROOT) if (ROOT / m).is_dir()]
    stats = load_stats(STATS_JSON)
    html_text = generate_html(meshes, stats)
    OUT_HTML.write_text(html_text, encoding="utf-8")
    print("Wrote", OUT_HTML)

if __name__ == "__main__":
    main()
