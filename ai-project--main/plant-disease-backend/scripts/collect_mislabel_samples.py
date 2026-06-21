"""
Collect sample images and metadata for top mislabel pairs from a misprediction JSONL.
Usage:
  python scripts/collect_mislabel_samples.py --report eval_output/misprediction_report_20260601T071202Z.jsonl --outdir eval_output/mislabel_samples --pairs 20 --per_pair 5
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

p = argparse.ArgumentParser()
p.add_argument("--report", default="eval_output/misprediction_report_20260601T071202Z.jsonl")
p.add_argument("--outdir", default="eval_output/mislabel_samples")
p.add_argument("--pairs", type=int, default=20)
p.add_argument("--per_pair", type=int, default=5)
args = p.parse_args()

report = Path(args.report)
outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

if not report.exists():
    print(f"Report not found: {report}")
    raise SystemExit(2)

rows = []
with report.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

pair_counts = Counter()
by_pair = defaultdict(list)
for r in rows:
    expected = r.get("expected") or ""
    predicted = r.get("predicted") or r.get("disease") or ""
    pair = f"{expected}__to__{predicted}"
    pair_counts[pair] += 1
    by_pair[pair].append(r)

top_pairs = [p for p, _ in pair_counts.most_common(args.pairs)]
print(f"Top {len(top_pairs)} pairs selected")

for pair in top_pairs:
    pair_dir = outdir / pair
    pair_dir.mkdir(parents=True, exist_ok=True)
    samples = by_pair.get(pair, [])[: args.per_pair]
    for i, s in enumerate(samples, 1):
        inp = Path(s.get("input"))
        if not inp.exists():
            # try resolving relative to backend
            backend_root = Path(__file__).resolve().parent.parent
            alt = backend_root / s.get("input")
            if alt.exists():
                inp = alt
        if not inp.exists():
            # skip if missing
            meta_path = pair_dir / f"sample_{i:02d}.json"
            meta_path.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
            continue
        # copy image
        try:
            dest_img = pair_dir / f"sample_{i:02d}{inp.suffix}"
            dest_img.write_bytes(inp.read_bytes())
        except Exception:
            pass
        # write metadata
        meta = s.copy()
        meta_path = pair_dir / f"sample_{i:02d}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"Samples collected under: {outdir}")
