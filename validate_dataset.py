"""
validate_dataset.py
====================
Audits the RDD2022 dataset for annotation quality issues.

Checks performed per split (train / val / test):
  1.  Missing label files
  2.  Empty label files (no-damage images) — counted separately, not errors
  3.  Invalid annotation lines (wrong field count)
  4.  Out-of-range class IDs (must be 0-4)
  5.  Out-of-range YOLO coordinates (must be in (0,1))
  6.  Degenerate boxes (width or height <= 0)
  7.  Extremely large boxes (w or h > 0.95 of image dim) — likely wrong
  8.  Extremely small boxes (w*h < 0.0001) — too tiny to learn from
  9.  Corrupted image files (cannot be opened)
  10. Class distribution and imbalance ratio
  11. Bounding-box aspect-ratio statistics per class
  12. Duplicate filenames within a split

Outputs:
  validation_report.json     — full structured report
  validation_summary.txt     — human-readable summary

Run:
    python validate_dataset.py
    python validate_dataset.py --splits train val   (only those splits)
    python validate_dataset.py --fix-empty          (write empty placeholder label for missing)
"""

import os
import sys
import json
import argparse
from collections import defaultdict

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("WARNING: Pillow not installed — image-corruption check disabled.")

CLASS_NAMES = {
    0: "D00 Longitudinal Crack",
    1: "D10 Transverse Crack",
    2: "D20 Alligator Crack",
    3: "D40 Pothole",
    4: "D43/D44 Other Damage",
}
LARGE_BOX_THRESHOLD  = 0.95    # w or h fraction of image side
SMALL_BOX_THRESHOLD  = 0.0001  # w*h area fraction
VALID_CLASS_IDS      = set(range(5))


def validate_split(root: str, split: str, fix_missing: bool = False) -> dict:
    img_dir = os.path.join(root, split, 'images')
    lbl_dir = os.path.join(root, split, 'labels')

    if not os.path.isdir(img_dir):
        return {"error": f"Image directory not found: {img_dir}"}

    img_files = sorted(f for f in os.listdir(img_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png')))

    report = {
        "split":              split,
        "total_images":       len(img_files),
        "missing_labels":     [],
        "empty_labels":       [],
        "corrupted_images":   [],
        "invalid_lines":      [],
        "class_id_errors":    [],
        "coord_range_errors": [],
        "degenerate_boxes":   [],
        "large_boxes":        [],
        "small_boxes":        [],
        "duplicate_names":    [],
        "class_counts":       defaultdict(int),
        "box_stats":          defaultdict(list),   # class_id → list of (w, h, area)
        "total_boxes":        0,
        "total_errors":       0,
    }

    seen_names = {}
    for img_name in img_files:
        # Duplicate check
        stem = os.path.splitext(img_name)[0]
        if stem in seen_names:
            report["duplicate_names"].append(img_name)
        seen_names[stem] = True

        # Image corruption check
        if PIL_AVAILABLE:
            img_path = os.path.join(img_dir, img_name)
            try:
                with PILImage.open(img_path) as im:
                    im.verify()
            except Exception as e:
                report["corrupted_images"].append(
                    {"file": img_name, "error": str(e)})
                continue

        # Label file
        lbl_name = stem + '.txt'
        lbl_path = os.path.join(lbl_dir, lbl_name)

        if not os.path.exists(lbl_path):
            report["missing_labels"].append(img_name)
            if fix_missing:
                os.makedirs(lbl_dir, exist_ok=True)
                open(lbl_path, 'w').close()
            continue

        with open(lbl_path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            report["empty_labels"].append(img_name)
            continue

        for line_no, line in enumerate(lines, 1):
            parts = line.split()
            if len(parts) < 5:
                report["invalid_lines"].append(
                    {"file": lbl_name, "line": line_no, "content": line})
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
            except ValueError:
                report["invalid_lines"].append(
                    {"file": lbl_name, "line": line_no, "content": line})
                continue

            # Class ID
            if cls_id not in VALID_CLASS_IDS:
                report["class_id_errors"].append(
                    {"file": lbl_name, "line": line_no,
                     "class_id": cls_id})
                continue

            # Coordinate range
            coords_ok = all(0.0 <= v <= 1.0 for v in [xc, yc, bw, bh])
            if not coords_ok:
                report["coord_range_errors"].append(
                    {"file": lbl_name, "line": line_no,
                     "coords": [xc, yc, bw, bh]})
                continue

            # Degenerate box
            if bw <= 0 or bh <= 0:
                report["degenerate_boxes"].append(
                    {"file": lbl_name, "line": line_no,
                     "w": bw, "h": bh})
                continue

            # Very large box
            if bw > LARGE_BOX_THRESHOLD or bh > LARGE_BOX_THRESHOLD:
                report["large_boxes"].append(
                    {"file": lbl_name, "line": line_no,
                     "class": CLASS_NAMES[cls_id],
                     "w": round(bw, 4), "h": round(bh, 4)})

            # Very small box
            if bw * bh < SMALL_BOX_THRESHOLD:
                report["small_boxes"].append(
                    {"file": lbl_name, "line": line_no,
                     "class": CLASS_NAMES[cls_id],
                     "area": round(bw * bh, 6)})

            report["class_counts"][cls_id] += 1
            report["box_stats"][cls_id].append((bw, bh, bw * bh))
            report["total_boxes"] += 1

    # Compute imbalance
    counts = report["class_counts"]
    if counts:
        max_c = max(counts.values())
        min_c = min(counts.values())
        report["imbalance_ratio"] = round(max_c / max(min_c, 1), 2)
        report["dominant_class"]  = CLASS_NAMES[max(counts, key=counts.get)]
        report["rarest_class"]    = CLASS_NAMES[min(counts, key=counts.get)]
    else:
        report["imbalance_ratio"] = 0
        report["dominant_class"]  = "N/A"
        report["rarest_class"]    = "N/A"

    # Per-class box-size summary
    box_summary = {}
    for cid, sizes in report["box_stats"].items():
        areas = [s[2] for s in sizes]
        ws    = [s[0] for s in sizes]
        hs    = [s[1] for s in sizes]
        import statistics
        box_summary[CLASS_NAMES[cid]] = {
            "count":      len(areas),
            "median_area_pct": round(statistics.median(areas) * 100, 3),
            "mean_w":     round(statistics.mean(ws), 4),
            "mean_h":     round(statistics.mean(hs), 4),
            "median_w":   round(statistics.median(ws), 4),
            "median_h":   round(statistics.median(hs), 4),
        }
    report["box_summary"] = box_summary

    # Convert defaultdicts to plain dicts for JSON
    report["class_counts"] = dict(report["class_counts"])
    report["class_counts_named"] = {CLASS_NAMES[int(k)]: v
                                     for k, v in report["class_counts"].items()}
    del report["box_stats"]

    # Total errors (excluding warnings like large/small boxes)
    report["total_errors"] = (
        len(report["missing_labels"]) +
        len(report["corrupted_images"]) +
        len(report["invalid_lines"]) +
        len(report["class_id_errors"]) +
        len(report["coord_range_errors"]) +
        len(report["degenerate_boxes"])
    )
    report["total_warnings"] = (
        len(report["large_boxes"]) +
        len(report["small_boxes"]) +
        len(report["duplicate_names"])
    )
    return report


def print_summary(reports: dict, out_file: str):
    lines = []
    lines.append("=" * 65)
    lines.append("  RDD2022 Dataset Validation Report")
    lines.append("=" * 65)

    for split, rep in reports.items():
        if "error" in rep:
            lines.append(f"\n[{split.upper()}] ERROR: {rep['error']}")
            continue
        lines.append(f"\n[{split.upper()}]  {rep['total_images']} images  "
                     f"{rep['total_boxes']} boxes")
        lines.append(f"  Errors   : {rep['total_errors']}")
        lines.append(f"  Warnings : {rep['total_warnings']}")
        lines.append(f"  Empty labels (no-damage images): {len(rep['empty_labels'])}")
        lines.append(f"  Class distribution:")
        for name, cnt in sorted(rep.get("class_counts_named", {}).items(),
                                key=lambda x: -x[1]):
            lines.append(f"    {name:<30} {cnt:>6}")
        lines.append(f"  Imbalance ratio: {rep.get('imbalance_ratio', 'N/A')}x  "
                     f"(most: {rep.get('dominant_class','?')}  "
                     f"least: {rep.get('rarest_class','?')})")
        lines.append(f"  Median bbox area by class:")
        for cname, bs in rep.get("box_summary", {}).items():
            lines.append(f"    {cname:<30} {bs['median_area_pct']:.2f}% of image")

        if rep["total_errors"] > 0:
            lines.append(f"\n  ERROR DETAILS:")
            if rep["missing_labels"]:
                lines.append(f"    Missing labels   : {len(rep['missing_labels'])} "
                              f"(first: {rep['missing_labels'][0]})")
            if rep["corrupted_images"]:
                lines.append(f"    Corrupted images : {len(rep['corrupted_images'])}")
            if rep["invalid_lines"]:
                lines.append(f"    Invalid lines    : {len(rep['invalid_lines'])}")
            if rep["class_id_errors"]:
                lines.append(f"    Bad class IDs    : {len(rep['class_id_errors'])}")
            if rep["coord_range_errors"]:
                lines.append(f"    Coord range errs : {len(rep['coord_range_errors'])}")
            if rep["degenerate_boxes"]:
                lines.append(f"    Degenerate boxes : {len(rep['degenerate_boxes'])}")

        if rep["total_warnings"] > 0:
            lines.append(f"\n  WARNINGS:")
            if rep["large_boxes"]:
                lines.append(f"    Oversized boxes  : {len(rep['large_boxes'])} "
                              f"(w or h > {LARGE_BOX_THRESHOLD})")
            if rep["small_boxes"]:
                lines.append(f"    Tiny boxes       : {len(rep['small_boxes'])} "
                              f"(area < {SMALL_BOX_THRESHOLD})")
            if rep["duplicate_names"]:
                lines.append(f"    Duplicate names  : {len(rep['duplicate_names'])}")

    lines.append("\n" + "=" * 65)
    total_err = sum(r.get("total_errors", 0) for r in reports.values())
    total_wrn = sum(r.get("total_warnings", 0) for r in reports.values())
    lines.append(f"  OVERALL: {total_err} errors, {total_wrn} warnings")
    lines.append("=" * 65)

    text = "\n".join(lines)
    print(text)
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"\nSummary saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate RDD2022 dataset annotations")
    parser.add_argument('--root',   default='.',
                        help="Project root directory")
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'],
                        help="Splits to validate")
    parser.add_argument('--fix-empty', action='store_true',
                        help="Create empty label file for images with no label")
    args = parser.parse_args()

    print(f"Validating splits: {args.splits}")
    print(f"Root: {os.path.abspath(args.root)}\n")

    all_reports = {}
    for split in args.splits:
        print(f"  Checking {split}...", end=' ', flush=True)
        rep = validate_split(args.root, split, fix_missing=args.fix_empty)
        all_reports[split] = rep
        err = rep.get("total_errors", 0)
        wrn = rep.get("total_warnings", 0)
        print(f"{rep.get('total_images', 0)} images  "
              f"{rep.get('total_boxes', 0)} boxes  "
              f"{err} errors  {wrn} warnings")

    json_out = 'validation_report.json'
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nFull report saved to: {json_out}")

    print_summary(all_reports, 'validation_summary.txt')


if __name__ == '__main__':
    main()
