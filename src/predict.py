from pathlib import Path

import numpy as np
import pandas as pd

from families import ALL_FAMILIES, FAMILIES, UNRESOLVED, family_letters
from match import box_table
from representative import locality_vectors

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ELECTIONS = [21, 22, 23, 24, 25]
TRAIN = [21, 22, 23, 24]
TEST = 25

LETTERS = family_letters()


def norm(df: pd.DataFrame) -> pd.DataFrame:
    return df.div(df.sum(axis=1), axis=0)


def box_family_shares(panel: pd.DataFrame, e: int) -> pd.DataFrame:
    named = pd.DataFrame(index=panel.index)
    for f in FAMILIES:
        cs = [f"votes_{e}_{l}" for l in LETTERS[e][f] if f"votes_{e}_{l}" in panel.columns]
        named[f] = panel[cs].sum(axis=1, min_count=1) if cs else 0.0
    total = panel[[c for c in panel.columns if c.startswith(f"votes_{e}_")]].sum(axis=1)
    named[UNRESOLVED] = total - named.sum(axis=1)
    return norm(named)


def loc_family_shares(long: pd.DataFrame, e: int) -> pd.DataFrame:
    vec = locality_vectors(long, e)
    out = pd.DataFrame(index=vec.index)
    for f in FAMILIES:
        ls = [l for l in LETTERS[e][f] if l in vec.columns]
        out[f] = vec[ls].sum(axis=1) if ls else 0.0
    out[UNRESOLVED] = 1 - out[FAMILIES].sum(axis=1)
    return out


def main() -> None:
    panel = pd.read_parquet(PROC / "box_panel_train.parquet").set_index(["locality_code", "kalpi"])
    long = pd.read_parquet(PROC / "results_long.parquet")

    shares = {e: box_family_shares(panel, e) for e in ELECTIONS}
    loc_shares = {e: loc_family_shares(long, e) for e in ELECTIONS}

    actual = shares[TEST]
    has25 = actual.notna().all(axis=1) & (actual.sum(axis=1) > 0)
    actual = actual[has25]
    weights = panel.loc[has25, [c for c in panel.columns if c.startswith(f"votes_{TEST}_")]].sum(axis=1)

    box_med = pd.concat([shares[e] for e in TRAIN], keys=TRAIN).groupby(level=[1, 2]).median()
    box_med = norm(box_med.loc[has25])
    box_last = shares[24].loc[has25]
    loc_med = pd.concat([loc_shares[e] for e in TRAIN], keys=TRAIN).groupby(level=1).median()
    loc_med = norm(loc_med.loc[panel.loc[has25].index.get_level_values(0)])
    loc_med.index = actual.index
    blend = norm(0.5 * box_med + 0.5 * loc_med)

    national = actual.mean(axis=0)
    equal = pd.Series(1 / len(FAMILIES), index=FAMILIES)

    methods = {
        "box median 21-24": box_med,
        "box last (24)": box_last,
        "locality median 21-24": loc_med,
        "blend 0.5/0.5": blend,
        "national average": pd.DataFrame([national[FAMILIES]] * len(actual), index=actual.index),
        "equal shares": pd.DataFrame([equal] * len(actual), index=actual.index),
    }
    methods = {k: v[FAMILIES] for k, v in methods.items()}
    actual_f = actual[FAMILIES]

    print(f"predicting election {TEST} for {len(actual)} panel boxes "
          f"(weighted by election-{TEST} valid votes)")
    print(f"\nactual national family shares (e{TEST}): " +
          ", ".join(f"{f}={national[f]:.1%}" for f in ALL_FAMILIES))
    print(f"\nweighted mean family MAE (|pred - actual|, per-family, "
          f"over {len(FAMILIES)} mapped families; residual '{UNRESOLVED}' excluded):")
    print(f"{'method':26s} {'MAE':>8s} {'vs locality median':>18s}")
    loc_l1 = methods["locality median 21-24"].sub(actual_f).abs().sum(axis=1)
    for name, pred in methods.items():
        mae = pred.sub(actual_f).abs().mean(axis=1)
        wmae = (mae * weights).sum() / weights.sum()
        if name != "locality median 21-24":
            delta = wmae - (loc_l1 * weights).sum() / weights.sum() / len(FAMILIES)
            print(f"{name:26s} {wmae:8.4f} {delta:18.4f}")
        else:
            print(f"{name:26s} {wmae:8.4f} {'(reference)':>18}")

    oth = pd.DataFrame({
        "box median": (box_med[UNRESOLVED].sub(actual[UNRESOLVED]).abs().mul(weights).sum() / weights.sum()),
        "locality median": (loc_med[UNRESOLVED].sub(actual[UNRESOLVED]).abs().mul(weights).sum() / weights.sum()),
    }, index=["residual Other"])
    print(f"\nresidual '{UNRESOLVED}' weighted MAE (reported separately, not in headline):")
    print(oth.round(4).to_string())

    box_l1 = methods["box median 21-24"].sub(actual_f).abs().sum(axis=1)
    wins = (box_l1 < loc_l1).mean()
    print(f"\nbox identity beats locality median in {wins:.1%} of boxes "
          f"(mean L1 {box_l1.mean():.3f} vs {loc_l1.mean():.3f})")

    print(f"\nper-family weighted MAE (box median vs locality median):")
    fam = pd.DataFrame({"box median": (box_med[FAMILIES].sub(actual_f).abs().mul(weights, axis=0)).sum() / weights.sum(),
                         "locality median": (loc_med[FAMILIES].sub(actual_f).abs().mul(weights, axis=0)).sum() / weights.sum()})
    print(fam.round(4).to_string())


    b = box_table(long)
    b25 = b[b["knesset"] == TEST][["locality_code", "kalpi", "eligible", "voters"]]
    tun = panel.join(b25.set_index(["locality_code", "kalpi"]), how="inner", lsuffix="_x")
    tun25 = (tun["voters"] / tun["eligible"]).rename("t25")
    tun25 = tun25[has25]
    box_turn = pd.concat([
        panel.loc[has25, f"voters_{e}"] / panel.loc[has25, f"eligible_{e}"] for e in TRAIN
    ], axis=1, keys=TRAIN).median(axis=1)
    loc_turn = pd.concat([
        (lambda v: v["voters"] / v["eligible"].replace(0, np.nan))(
            long[long["knesset"] == e].groupby("locality_code")[["voters", "eligible"]].sum())
        for e in TRAIN], axis=1, keys=TRAIN).median(axis=1)
    loc_turn = loc_turn.loc[tun25.index.get_level_values(0)].set_axis(tun25.index)

    print("\nturnout (voters/eligible) prediction, weighted MAE on election 25:")
    t_w = tun.loc[has25, "voters"]
    for name, pred in {"box median 21-24": box_turn, "locality median 21-24": loc_turn,
                       "blend 0.5/0.5": 0.5 * box_turn + 0.5 * loc_turn}.items():
        wmae = (pred.sub(tun25).abs().mul(t_w)).sum() / t_w.sum()
        print(f"  {name:26s} {wmae:.4f}")


if __name__ == "__main__":
    main()
