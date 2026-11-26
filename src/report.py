"""Turn results/runs.jsonl into a CPU-vs-GPU table and a stage-time chart.

    python src/report.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

runs = [json.loads(l) for l in Path("results/runs.jsonl").read_text().splitlines() if l.strip()]
# Keep the latest run per (backend, rows)
latest = {}
for r in runs:
    latest[(r["backend"], r["rows_model"])] = r

lines = ["| rows | backend | load | clean | features | join | train | predict | total (s) | ROC-AUC | speedup |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
for rows in sorted({k[1] for k in latest}):
    cpu, gpu = latest.get(("cpu", rows)), latest.get(("gpu", rows))
    for r in (cpu, gpu):
        if not r:
            continue
        s = r["stages_s"]
        speed = f"{cpu['total_s'] / r['total_s']:.1f}x" if cpu and r is gpu else "1.0x"
        lines.append(f"| {rows:,} | {r['backend']} | " + " | ".join(
            str(s.get(k, "")) for k in ["load", "clean", "features", "join", "train", "predict"])
            + f" | {r['total_s']} | {r['roc_auc']} | {speed} |")
Path("results/results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

# Stacked bar of stage times for the largest dataset
rows = max(k[1] for k in latest)
stages = ["load", "clean", "features", "join", "split", "train", "predict"]
fig, ax = plt.subplots(figsize=(7, 3))
for i, b in enumerate(["cpu", "gpu"]):
    r = latest.get((b, rows))
    if not r:
        continue
    left = 0
    for s in stages:
        v = r["stages_s"].get(s, 0)
        ax.barh(i, v, left=left, label=s if i == 0 else None)
        left += v
ax.set_yticks([0, 1], ["CPU", "GPU"])
ax.set_xlabel("seconds")
ax.set_title(f"End-to-end pipeline, {rows:,} rows")
ax.legend(ncol=4, fontsize=8)
fig.tight_layout()
fig.savefig("results/stage_times.png", dpi=150)
print("saved results/stage_times.png")
