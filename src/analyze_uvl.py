"""Automated analysis of the YOLOv11 UVL feature model via FlamaPy (BDD backend).

Reports: validity (satisfiable), #features, #leaf features, tree depth,
top-level variability dimensions, core/dead features, and the exact number of
valid configurations (counted with a BDD, so it scales past enumeration).

Run:  python src/analyze_uvl.py            # full analysis + 5 sample configs
      python src/analyze_uvl.py --sample 10
"""
import argparse
from pathlib import Path

from flamapy.metamodels.fm_metamodel.transformations import UVLReader
from flamapy.metamodels.bdd_metamodel.transformations import FmToBDD
from flamapy.metamodels.bdd_metamodel.operations import (
    BDDConfigurationsNumber, BDDSatisfiable, BDDCoreFeatures, BDDDeadFeatures,
    BDDSampling,
)

ROOT = Path(__file__).resolve().parent.parent
UVL = ROOT / "models" / "yolo_custom_model.uvl"


def children(feature):
    kids = []
    for rel in feature.get_relations():
        kids.extend(rel.children)
    return kids


def depth_nodes(feature):
    kids = children(feature)
    return 1 if not kids else 1 + max(depth_nodes(k) for k in kids)


def run(op, bdd):
    op.execute(bdd)
    return op.get_result()


def sample_configs(bdd, leaf_names, n):
    op = BDDSampling()
    op.set_sample_size(n)
    op.set_with_replacement(False)
    op.execute(bdd)
    out = []
    for cfg in op.get_result():
        names = []
        for f, on in cfg.elements.items():
            name = f.name if hasattr(f, "name") else str(f)
            if on and name in leaf_names:
                names.append(name)
        out.append(sorted(names))
    return out


def analyze(uvl_path=UVL):
    """Run the full BDD analysis and return results as a dict (importable API)."""
    fm = UVLReader(str(uvl_path)).transform()
    features = fm.get_features()
    leaves = [f for f in features if not children(f)]
    root = fm.root
    dims = [c.name for c in children(root)]

    bdd = FmToBDD(fm).transform()
    return {
        "model_file": Path(uvl_path).name,
        "root": root.name,
        "satisfiable": run(BDDSatisfiable(), bdd),
        "n_features": len(features),
        "n_leaves": len(leaves),
        "max_depth": depth_nodes(root),
        "dimensions": dims,
        "n_core": len(run(BDDCoreFeatures(), bdd)),
        "n_dead": len(run(BDDDeadFeatures(), bdd)),
        "n_configs": run(BDDConfigurationsNumber(), bdd),
        "_bdd": bdd,
        "_leaf_names": {f.name for f in leaves},
    }


def sample(n=5, uvl_path=UVL):
    """Return n random valid configurations (list of selected-leaf lists)."""
    res = analyze(uvl_path)
    return sample_configs(res["_bdd"], res["_leaf_names"], n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5, help="number of random valid configs to list")
    args = ap.parse_args()

    r = analyze()
    satisfiable = r["satisfiable"]
    n_configs = r["n_configs"]

    print(f"Root feature          : {r['root']}")
    print(f"Satisfiable (valid)   : {satisfiable}")
    print(f"Total features        : {r['n_features']}")
    print(f"Leaf features         : {r['n_leaves']}")
    print(f"Max tree depth (nodes): {r['max_depth']}")
    print(f"Top-level dimensions  : {len(r['dimensions'])} -> {r['dimensions']}")
    print(f"Core features         : {r['n_core']}")
    print(f"Dead features         : {r['n_dead']}")
    print(f"VALID CONFIGURATIONS  : {n_configs:,}")

    print(f"\nRandom sample of {args.sample} valid configurations (selected leaf features):")
    for i, sel in enumerate(sample_configs(r["_bdd"], r["_leaf_names"], args.sample), 1):
        print(f"  [{i}] {', '.join(sel)}")


if __name__ == "__main__":
    main()
