from pathlib import Path

import numpy as np
import pandas as pd

from families import FAMILIES
from forecast import election_votes

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ELECTIONS = [21, 22, 23, 24, 25]
TEST_ELECTIONS = [23, 24, 25]
MIN_EST_PER_STRATUM = 200


def strata(census: pd.DataFrame) -> pd.DataFrame:
    c = census.copy()
    c["religiosity"] = c["hh_MidatDatiyut_Name"].round().astype("Int64").astype(str)
    c["size"] = pd.qcut(c["pop_total"].rank(method="first"), 3,
                         labels=["small", "medium", "large"])
    c["rel_x_likud"] = c["religiosity"] + "|" + c["size"].astype(str)
    return c[["religiosity", "size", "rel_x_likud"]]


def deltas(shares: dict, a: int, b: int) -> pd.DataFrame:
    prev, cur = shares[a], shares[b]
    idx = prev.index.intersection(cur.index)
    return cur.loc[idx] - prev.loc[idx]


def main() -> None:
    long = pd.read_parquet(PROC / "results_long.parquet")
    census = pd.read_csv(PROC / "locality_demographics.csv").set_index("locality_code")
    shares = {e: election_votes(long, e) for e in ELECTIONS}
    st = strata(census)

    rows = []
    for T in TEST_ELECTIONS:
        prev = T - 1
        train_pairs = [(a, a + 1) for a in range(21, prev) if a + 1 <= prev]
        if not train_pairs:
            continue

        dd = pd.concat([deltas(shares, a, b).assign(_pair=f"{a}->{b}")
                        for a, b in train_pairs])
        dd["_stratum"] = st["rel_x_likud"].reindex(dd.index.get_level_values(0)).values

        w = (long[(long["knesset"] == T) & (~long["is_special"])]
             .groupby(["locality_code", "kalpi"]).valid.first())
        d_test = deltas(shares, prev, T)
        d_test = d_test[w.reindex(d_test.index) > 0]
        d_test["_stratum"] = st["rel_x_likud"].reindex(d_test.index.get_level_values(0)).values
        wt = w.reindex(d_test.index).astype(float)

        uniform = dd[FAMILIES].mean()
        by_stratum = dd.groupby("_stratum")[FAMILIES].mean()
        support = dd.groupby("_stratum").size()
        usable = support[support >= MIN_EST_PER_STRATUM].index

        est = by_stratum.reindex(d_test["_stratum"].values)
        est = est.where(est.notna(), pd.DataFrame([uniform] * len(est),
                                                  index=est.index, columns=est.columns))
        ev = est.to_numpy()
        wsum = float(wt.sum())
        wv = wt.to_numpy()
        obs = d_test[FAMILIES].to_numpy()
        mae_u = np.abs(obs - uniform[FAMILIES].to_numpy()).T @ wv / wsum
        mae_s = np.abs(obs - ev).T @ wv / wsum
        rows.append({
            "transition": f"{prev}->{T}",
            "n_train_pairs": len(train_pairs),
            "n_strata_total": len(by_stratum),
            "n_strata_usable": len(usable),
            "share_votes_covered": float(wt[d_test["_stratum"].isin(usable)].sum() / wsum),
            **{f"MAE uniform {f}": float(mae_u[j]) for j, f in enumerate(FAMILIES)},
            **{f"MAE stratum {f}": float(mae_s[j]) for j, f in enumerate(FAMILIES)},
        })

    df = pd.DataFrame(rows).set_index("transition")
    fam_u = [c for c in df.columns if c.startswith("MAE uniform")]
    fam_s = [c for c in df.columns if c.startswith("MAE stratum")]

    print("Does a *differential* swing beat a *uniform* national swing?")
    print("Both are estimated only from elections < the held-out transition.")
    print("Target: actual change in box share, weighted by valid votes.\n")
    print(df[["n_train_pairs", "n_strata_total", "n_strata_usable",
              "share_votes_covered"]].round(3).to_string())

    u = df[fam_u].mean(axis=1).rename("uniform")
    s = df[fam_s].mean(axis=1).rename("differential (stratum)")
    print("\nweighted mean family MAE on the share *change*:")
    print(pd.DataFrame({"uniform swing": u.round(4),
                        "differential swing": s.round(4),
                        "delta": (s - u).round(4)}).to_string())
    print(f"\naverage: uniform {u.mean():.4f}  differential {s.mean():.4f}  "
          f"delta {s.mean() - u.mean():+.4f}")

    print("\nper-family, averaged over held-out transitions:")
    mu = df[fam_u].mean()
    ms = df[fam_s].mean()
    mu.index = [c.replace("MAE uniform ", "") for c in mu.index]
    ms.index = [c.replace("MAE stratum ", "") for c in ms.index]
    pf = pd.DataFrame({"uniform": mu.round(4), "differential": ms.round(4)})
    pf["delta"] = (pf["differential"] - pf["uniform"]).round(4)
    print(pf.to_string())

    print("\nhow much of the pooled variance in box share change is between-strata")
    print("(exact ANOVA decomposition, stratum means estimated from training pairs only):")
    for T in TEST_ELECTIONS:
        prev = T - 1
        dd = pd.concat([deltas(shares, a, a + 1) for a in range(21, prev)])
        dd["_stratum"] = st["rel_x_likud"].reindex(dd.index.get_level_values(0)).values
        dd = dd[dd["_stratum"].notna()]
        grp = dd.groupby("_stratum")[FAMILIES]
        within = float((grp.var(ddof=0) * grp.count()).sum().sum())
        total = float(dd[FAMILIES].var(ddof=0).sum() * len(dd))
        print(f"  train pairs into e{prev}: strata explain "
              f"{100 * (1 - within / total):.1f}% "
              f"({len(FAMILIES)} families x {len(dd)} box-transitions)")


if __name__ == "__main__":
    main()
