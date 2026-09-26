from pathlib import Path

import numpy as np
import pandas as pd

from match import box_table, build_panel, join_election

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ELECTIONS = [21, 22, 23, 24, 25]
THRESHOLD = 3


def locality_vectors(long: pd.DataFrame, e: int) -> pd.DataFrame:
    sub = long[(long["knesset"] == e) & (~long["is_special"])]
    vec = sub.groupby(["locality_code", "letter"])["votes"].sum().unstack(fill_value=0)
    vec = vec.div(vec.sum(axis=1), axis=0)
    return vec


def box_align(panel: pd.DataFrame, loc_vecs: dict, e: int) -> pd.Series:
    cols = [f"votes_{e}_{c}" for c in loc_vecs[e].columns]
    sub = panel[["locality_code", "kalpi"] + cols].set_index(["locality_code", "kalpi"])
    sub = sub[sub.notna().all(axis=1) & (sub.sum(axis=1) > 0)]
    box_shares = sub.div(sub.sum(axis=1), axis=0)
    loc = loc_vecs[e].reindex(columns=loc_vecs[e].columns)
    align = box_shares.mul(loc.loc[sub.index.get_level_values(0)].values).sum(axis=1)
    return align


def pick(g, key, how):
    return g.loc[g[key].agg(how)]


def loo_validation(long: pd.DataFrame, b: pd.DataFrame, loc_vecs: dict,
                   test: int, threshold: int) -> dict:
    train = [e for e in ELECTIONS if e != test]
    panel = build_panel(b, long, train)
    panel = join_election(panel, long, test)

    align = pd.DataFrame({e: box_align(panel, loc_vecs, e) for e in ELECTIONS}).reset_index()
    align["train_mean"] = align[[e for e in train]].mean(axis=1)
    align["test"] = align[test]
    align["has_test"] = align["test"].notna()

    n_pool = align.groupby("locality_code").size()
    n_test = align[align["has_test"]].groupby("locality_code").size()
    usable = n_pool[n_pool >= threshold].index
    usable = usable[n_test.reindex(usable).fillna(0) >= 2]
    a = align[align["locality_code"].isin(usable)].copy()

    best = a.groupby("locality_code", group_keys=False).apply(lambda g: pick(g, "train_mean", "idxmax"))
    worst = a.groupby("locality_code", group_keys=False).apply(lambda g: pick(g, "train_mean", "idxmin"))
    rng = np.random.default_rng(0)
    rand = a.groupby("locality_code", group_keys=False).apply(
        lambda g: g.iloc[rng.integers(0, len(g))])

    rows = []
    for name, s in [("best", best), ("random", rand), ("worst", worst)]:
        rows.append({
            "selection": name,
            "train_align": s["train_mean"].mean(),
            "survives": s["has_test"].mean(),
            "test_align": s["test"].mean(),
        })
    m = a.groupby("locality_code")[["train_mean", "test"]].transform(lambda s: s - s.mean())
    return {"test": test, "n_localities": len(best),
            "n_boxes_pool": len(panel), "persistence": m["train_mean"].corr(m["test"]),
            "table": pd.DataFrame(rows)}


def main() -> None:
    long = pd.read_parquet(PROC / "results_long.parquet")
    b = box_table(long)
    loc_vecs = {e: locality_vectors(long, e) for e in ELECTIONS}

    res = [loo_validation(long, b, loc_vecs, test, THRESHOLD) for test in ELECTIONS]

    print(f"leave-one-out validation (locality threshold >= {THRESHOLD} panel boxes):")
    print(f"{'held out':>9} {'localities':>10} {'persistence':>12} | "
          f"{'best':>7} {'random':>7} {'worst':>7} | {'survive best':>12} {'survive rand':>12}")
    for r in res:
        t = r["table"].set_index("selection")
        print(f"{r['test']:>9} {r['n_localities']:>10} {r['persistence']:>12.3f} | "
              f"{t.loc['best','test_align']:>7.3f} {t.loc['random','test_align']:>7.3f} "
              f"{t.loc['worst','test_align']:>7.3f} | "
              f"{t.loc['best','survives']:>12.2f} {t.loc['random','survives']:>12.2f}")
    allt = pd.concat([r["table"].set_index("selection") for r in res]).groupby(level=0).mean()
    print(f"{'avg':>9} {'':>10} {'':>12} | " + " ".join(f"{v:>7.3f}" for v in allt["test_align"])
          + " | " + " ".join(f"{v:>12.2f}" for v in allt["survives"]))
    print(f"\naverage persistence corr: {np.mean([r['persistence'] for r in res]):.3f}")

    census = pd.read_csv(PROC / "locality_demographics.csv")
    r25 = res[ELECTIONS.index(25)]
    best25 = r25["table"].set_index("selection").loc["best"]
    panel25 = pd.read_parquet(PROC / "box_panel_train.parquet")
    panel25 = panel25.merge(census[["locality_code", "pop_total", "pct_jews", "hh_MidatDatiyut_Name"]],
                            on="locality_code", how="left")
    print(f"\ne25 best-in-train test align: {best25['test_align']:.3f} (n={r25['n_localities']})")


if __name__ == "__main__":
    main()
