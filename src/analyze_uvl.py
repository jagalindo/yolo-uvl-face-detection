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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=5, help="number of random valid configs to list")
    args = ap.parse_args()

    fm = UVLReader(str(UVL)).transform()
    features = fm.get_features()
    leaves = [f for f in features if not children(f)]
    leaf_names = {f.name for f in leaves}
    root = fm.root
    dims = [c.name for c in children(root)]

    bdd = FmToBDD(fm).transform()
    satisfiable = run(BDDSatisfiable(), bdd)
    n_configs = run(BDDConfigurationsNumber(), bdd)
    core = run(BDDCoreFeatures(), bdd)
    dead = run(BDDDeadFeatures(), bdd)

    print(f"Model file            : {UVL.name}")
    print(f"Root feature          : {root.name}")
    print(f"Satisfiable (valid)   : {satisfiable}")
    print(f"Total features        : {len(features)}")
    print(f"Leaf features         : {len(leaves)}")
    print(f"Max tree depth (nodes): {depth_nodes(root)}")
    print(f"Top-level dimensions  : {len(dims)} -> {dims}")
    print(f"Core features         : {len(core)}")
    print(f"Dead features         : {len(dead)}")
    print(f"VALID CONFIGURATIONS  : {n_configs:,}")

    print(f"\nRandom sample of {args.sample} valid configurations (selected leaf features):")
    for i, sel in enumerate(sample_configs(bdd, leaf_names, args.sample), 1):
        print(f"  [{i}] {', '.join(sel)}")


if __name__ == "__main__":
    main()
