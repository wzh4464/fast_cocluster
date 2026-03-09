#!/usr/bin/env python3
"""Re-run Classic4 standalone PNMTF with 10 seeds. Saves to dedicated file."""

import json, sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_classic4_baselines import (
    load_classic4, run_pnmtf, evaluate, DATA_DIR, RESULTS_DIR
)

def main():
    X_raw, labels = load_classic4()
    k = 4
    print(f"Classic4: {X_raw.shape}, k={k}")

    results = []
    for seed in range(10):
        print(f"  PNMTF seed={seed}...", end=" ", flush=True)
        try:
            pred, elapsed = run_pnmtf(X_raw, labels, k, seed)
            nmi, ari = evaluate(labels, pred)
            print(f"NMI={nmi:.4f}, ARI={ari:.4f}, Time={elapsed:.1f}s")
            results.append(dict(method="PNMTF", seed=seed, nmi=float(nmi),
                               ari=float(ari), time_s=round(elapsed, 2)))
        except Exception as e:
            print(f"FAILED: {e}")
            results.append(dict(method="PNMTF", seed=seed, nmi=None,
                               ari=None, time_s=None, error=str(e)))

    ok = [r for r in results if r.get('nmi') is not None]
    if ok:
        nmis = np.array([r['nmi'] for r in ok])
        aris = np.array([r['ari'] for r in ok])
        print(f"\nMean: NMI={nmis.mean():.3f}±{nmis.std():.3f}, "
              f"ARI={aris.mean():.3f}±{aris.std():.3f}")

    out_path = RESULTS_DIR / "classic4_pnmtf_10seeds.json"
    with open(out_path, "w") as f:
        json.dump({
            "experiment": "Classic4 standalone PNMTF (10 seeds)",
            "dataset": "classic4_paper",
            "shape": list(X_raw.shape),
            "k": k,
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "results": results,
        }, f, indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
