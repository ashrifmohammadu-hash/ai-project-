"""
Audit per-class validation F1 from results.txt and scan dataset for label issues.

Run:
  python audit_training_labels.py
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "output" / "results.txt"
DATASET = ROOT / "dataset_sri_lanka"
REPORT = ROOT / "output" / "label_audit_report.txt"
F1_MIN = 0.80

# Classes built from wrong crop (documented proxy — hurts F1)
PROXY_LABEL_ISSUES = {
    "Coconut_Gray_leaf_spot": "Was papaya Ringspot proxy — relabel to palm-like source",
    "Coconut_Leaf_rot": "Was papaya Bacterial_spot proxy",
    "Coconut_healthy": "Was papaya healthy proxy",
    "Chili_Bacterial_spot": "Copy of Pepper,_bell_Bacterial_spot (duplicate labels)",
    "Chili_healthy": "Copy of Pepper,_bell_healthy (duplicate labels)",
}


def parse_f1_from_results(text: str) -> list[tuple[str, float, int]]:
    rows: list[tuple[str, float, int]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("PLANT", "=", "Algorithm", "Feature", "Image", "Training", "Validation", "Number", "accuracy", "macro", "weighted")):
            continue
        m = re.match(
            r"^(\S+(?:\([^)]*\))?(?:_\S+)*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$",
            line,
        )
        if not m:
            # class names with spaces e.g. Cherry_(including_sour)_healthy
            parts = line.split()
            if len(parts) >= 5 and parts[-1].isdigit():
                try:
                    f1 = float(parts[-2])
                    support = int(parts[-1])
                    name = " ".join(parts[:-4])
                    if parts[-4].replace(".", "").isdigit():
                        name = " ".join(parts[:-4])
                        prec, rec, f1 = float(parts[-4]), float(parts[-3]), float(parts[-2])
                        rows.append((name.replace(" ", "_") if " " in name else parts[0], f1, support))
                except ValueError:
                    pass
            continue
        name, _prec, _rec, f1_s, sup = m.groups()
        rows.append((name, float(f1_s), int(sup)))
    return rows


def parse_f1_simple(text: str) -> list[tuple[str, float, int]]:
    """Parse sklearn classification_report lines (fixed-width)."""
    rows: list[tuple[str, float, int]] = []
    for line in text.splitlines():
        if not line.strip() or "avg" in line or line.strip().startswith("accuracy"):
            continue
        # Last three numbers before support: precision recall f1 support
        tokens = line.split()
        if len(tokens) < 5:
            continue
        try:
            support = int(tokens[-1])
            f1 = float(tokens[-2])
            recall = float(tokens[-3])
            precision = float(tokens[-4])
            name = " ".join(tokens[:-4]).strip()
            if not name or name[0].isdigit():
                continue
            rows.append((name, f1, support))
        except ValueError:
            continue
    return rows


def count_images() -> dict[str, int]:
    counts: dict[str, int] = {}
    if not DATASET.is_dir():
        return counts
    for folder in DATASET.iterdir():
        if folder.is_dir():
            n = sum(
                1
                for f in folder.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            counts[folder.name] = n
    return counts


def main() -> None:
    lines: list[str] = []
    lines.append("LABEL & VALIDATION AUDIT")
    lines.append("=" * 60)

    if RESULTS.exists():
        text = RESULTS.read_text(encoding="utf-8", errors="replace")
        f1_rows = parse_f1_simple(text)
        weak = [(n, f, s) for n, f, s in f1_rows if f < F1_MIN]
        ok = [(n, f, s) for n, f, s in f1_rows if f >= F1_MIN]
        lines.append(f"\nClasses with F1 >= {F1_MIN:.0%}: {len(ok)}")
        lines.append(f"Classes with F1 <  {F1_MIN:.0%}: {len(weak)}")
        lines.append("\n--- BELOW 80% F1 (need fix or more real photos) ---")
        for name, f1, sup in sorted(weak, key=lambda x: x[1]):
            note = PROXY_LABEL_ISSUES.get(name, "")
            lines.append(f"  {name:45s} F1={f1:.2f}  n={sup:4d}  {note}")
        lines.append("\n--- OK (>= 80% F1) ---")
        for name, f1, sup in sorted(ok, key=lambda x: -x[1])[:15]:
            lines.append(f"  {name:45s} F1={f1:.2f}  n={sup}")
        if len(ok) > 15:
            lines.append(f"  ... and {len(ok) - 15} more")
    else:
        lines.append("No output/results.txt — run train.py first.")

    counts = count_images()
    lines.append("\n--- DATASET IMAGE COUNTS (SL crops) ---")
    sl_prefixes = ("Banana_", "Rice_", "Coconut_", "Tea_", "Chili_", "Mango_", "Papaya_")
    for name in sorted(counts):
        if name.startswith(sl_prefixes):
            flag = " [PROXY?]" if name in PROXY_LABEL_ISSUES else ""
            lines.append(f"  {name:45s} {counts[name]:4d} images{flag}")

    lines.append("\n--- RECOMMENDED FIXES ---")
    lines.append("  0. python recheck_dataset_mislabels.py   # CSV of folder vs model; optional --quarantine 0.72")
    lines.append("  1. python fix_dataset_mislabels.py   # rebuild coconut/chili, drop duplicates")
    lines.append("  2. python build_dataset_sri_lanka.py")
    lines.append("  3. python train.py")
    lines.append("  4. Add YOUR real coconut/banana/chili photos (150+ per class)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report_text = "\n".join(lines) + "\n"
    REPORT.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"Saved -> {REPORT}")


if __name__ == "__main__":
    main()
