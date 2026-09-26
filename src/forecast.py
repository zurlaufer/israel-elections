from pathlib import Path

import numpy as np
import pandas as pd

from families import ALL_FAMILIES, FAMILIES, UNRESOLVED, family_letters
from predict import loc_family_shares

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output"

ELECTIONS = [21, 22, 23, 24, 25]
TRAIN_ELECTIONS = [22, 23, 24]
TEST = 25
N_LAGS = 3

LETTERS = family_letters()

CENSUS_FEATS = ["pop_total", "pct_jews", "hh_MidatDatiyut_Name", "age_median",
                "employeesAnnual_medWage", "AcadmCert_pcnt", "own_pcnt",
                "age0_19_pcnt", "age65_pcnt"]


def election_votes(long: pd.DataFrame, e: int) -> pd.DataFrame:
    sub = long[(long["knesset"] == e) & (~long["is_special"])]
    wide = sub.pivot_table(index=["locality_code", "kalpi"], columns="letter",
                           values="votes", aggfunc="sum", fill_value=0)
    total = wide.sum(axis=1)
    fam = pd.DataFrame(index=wide.index)
    for f in FAMILIES:
        ls = [c for c in LETTERS[e][f] if c in wide.columns]
        fam[f] = wide[ls].sum(axis=1) if ls else 0.0
    fam[UNRESOLVED] = total - fam[FAMILIES].sum(axis=1)
    fam = fam.div(total, axis=0)
    fam = fam[total > 0]
    return fam


def election_turnout(long: pd.DataFrame, e: int) -> pd.DataFrame:
    sub = long[(long["knesset"] == e) & (~long["is_special"])]
    t = sub.groupby(["locality_code", "kalpi"]).agg(
        eligible=("eligible", "first"), voters=("voters", "first"))
    t["turnout"] = t["voters"] / t["eligible"].replace(0, np.nan)
    return t


def assemble(long: pd.DataFrame, census: pd.DataFrame, e: int,
             loc_shares: dict, turns: dict, shares: dict) -> pd.DataFrame:
    fam = shares[e]
    t = turns[e]
    idx = fam.index
    rows = pd.DataFrame(index=idx)

    for lag in range(1, N_LAGS + 1):
        s = e - lag
        has = f"has_lag{lag}"
        for f in FAMILIES:
            rows[f"own_{f}_l{lag}"] = 0.0
            rows[f"loc_{f}_l{lag}"] = 0.0
        rows[has] = 0.0

        if s in shares:
            prev = shares[s].reindex(idx)
            for f in FAMILIES:
                rows[f"own_{f}_l{lag}"] = prev[f].fillna(0.0)
            rows[has] = prev["Likud"].notna().astype(float)

        if s in loc_shares:
            ls = loc_shares[s]
            for f in FAMILIES:
                rows[f"loc_{f}_l{lag}"] = ls[f].reindex(idx.get_level_values(0)).values

    t1 = turns.get(e - 1)
    if t1 is not None:
        r = t1.reindex(idx)
        rows["own_turnout_l1"] = r["turnout"].fillna(0.0)
        rows["log_eligible_l1"] = np.log(r["eligible"]).fillna(0.0)
        rows["has_eligible_l1"] = r["eligible"].notna().astype(float)
    else:
        rows["own_turnout_l1"] = 0.0
        rows["log_eligible_l1"] = 0.0
        rows["has_eligible_l1"] = 0.0

    c = census.reindex(idx.get_level_values(0))
    for f in CENSUS_FEATS:
        rows[f] = c[f].values
    rows[CENSUS_FEATS] = rows[CENSUS_FEATS].fillna(census.median())

    rows["n_boxes_loc"] = fam.groupby(level=0).size().reindex(idx.get_level_values(0)).values
    feat = [c for c in rows.columns if c.startswith(("own_", "loc_", "has_", "n_boxes_loc"))]
    rows[feat] = rows[feat].fillna(0.0)
    rows["valid"] = t.reindex(idx)["voters"].fillna(0.0)
    for f in FAMILIES:
        rows[f"target_{f}"] = fam[f]
    rows["target_turnout"] = t.reindex(idx)["turnout"].fillna(0.0)
    return rows


def standardize(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def fit_ridge(X: np.ndarray, Y: np.ndarray, w: np.ndarray, lam: float = 1e-2) -> np.ndarray:
    wcol = w[:, None]
    S = X.T @ (X * wcol)
    B = X.T @ (Y * wcol)
    d = X.shape[1]
    return np.linalg.solve(S + lam * np.eye(d), B)


def predict_linear(Xf: np.ndarray, W: np.ndarray) -> np.ndarray:
    P = Xf @ W
    P = np.clip(P, 0.0, None)
    return P / P.sum(axis=1, keepdims=True)


def wmae(pred: pd.DataFrame, actual: pd.DataFrame, w: pd.Series) -> pd.Series:
    mae = pred.sub(actual).abs()
    den = mae.notna().mul(w, axis=0).sum()
    return mae.mul(w, axis=0).sum() / den


def main() -> None:
    long = pd.read_parquet(PROC / "results_long.parquet")
    census = pd.read_csv(PROC / "locality_demographics.csv").set_index("locality_code")
    for f in ["pop_total", "employeesAnnual_medWage"]:
        census[f] = np.log1p(census[f])
    census = census[CENSUS_FEATS].apply(lambda s: s.fillna(s.median()))

    shares = {e: election_votes(long, e) for e in ELECTIONS}
    turns = {e: election_turnout(long, e) for e in ELECTIONS}
    loc_shares = {e: loc_family_shares(long, e) for e in ELECTIONS}

    train_frames = [assemble(long, census, e, loc_shares, turns, shares) for e in TRAIN_ELECTIONS]
    train = pd.concat(train_frames)
    test = assemble(long, census, TEST, loc_shares, turns, shares)

    feat_cols = [c for c in train.columns
                 if c not in ("valid", "target_turnout")
                 and not c.startswith("target_")]
    X = train[feat_cols].values.astype(float)
    Y = train[[f"target_{f}" for f in FAMILIES]].values
    w = train["valid"].values

    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = standardize(X, mu, sd)
    Xs = np.column_stack([np.ones(len(Xs)), Xs])

    print(f"training: {len(train)} box-transitions, {len(feat_cols)} features")
    W = fit_ridge(Xs, Y, w)
    print("weighted ridge (linear) fitted (reported for comparison)")

    Ztest = np.column_stack([np.ones(len(test)), standardize(test[feat_cols].values.astype(float), mu, sd)])
    model_pred = pd.DataFrame(predict_linear(Ztest, W), index=test.index, columns=FAMILIES)
    test = test.join(model_pred.rename(columns={f: f"model_{f}" for f in FAMILIES}))
    actual = test[[f"target_{f}" for f in FAMILIES]].rename(columns={f"target_{f}": f for f in FAMILIES})
    test_w = test["valid"]

    own24 = shares[24].reindex(test.index).fillna(0.0)
    loc24 = loc_shares[24].reindex(test.index.get_level_values(0))
    loc24.index = test.index
    national = actual.mean()
    equal = pd.Series(1 / len(FAMILIES), index=FAMILIES)

    nat24 = pd.Series(
        [shares[24][f].mean() for f in ALL_FAMILIES], index=ALL_FAMILIES)
    mro = own24.where(test["has_lag1"] == 1).fillna(loc24)
    n_fallback = int(mro[FAMILIES].isna().any(axis=1).sum())
    mro = mro.fillna(nat24)
    actual_f = actual[FAMILIES]
    mro_f, own24_f, loc24_f = mro[FAMILIES], own24[FAMILIES], loc24[FAMILIES]

    methods = {
        "own 24, else locality 24": mro_f,
        "box own 24 (matched only)": own24_f.where(test["has_lag1"] == 1),
        "locality 24": loc24_f,
        "ridge model": test[[f"model_{f}" for f in FAMILIES]].rename(
            columns={f"model_{f}": f for f in FAMILIES}),
        "national average": pd.DataFrame([national] * len(test), index=test.index),
        "equal shares": pd.DataFrame([equal] * len(test), index=test.index),
    }

    print(f"\nweighted mean family MAE on election 25 "
          f"({len(test)} boxes, weight = valid votes, "
          f"{len(FAMILIES)} mapped families; residual '{UNRESOLVED}' excluded):")
    print(f"{'method':32s} {'all':>8s} {'matched':>8s} {'unmatched':>10s}")
    msk = test["has_lag1"] == 1
    for name, pred in methods.items():
        if pred is None or pred.isna().all().all():
            continue
        w_all = wmae(pred, actual_f, test_w).mean()
        w_m = wmae(pred[msk], actual_f[msk], test_w[msk]).mean()
        w_u = wmae(pred[~msk], actual_f[~msk], test_w[~msk]).mean()
        print(f"{name:32s} {w_all:8.4f} {w_m:8.4f} {w_u:10.4f}")

    print(f"\nresidual '{UNRESOLVED}' weighted MAE (reported separately, not in headline):")
    oth_actual = shares[TEST].reindex(test.index)[UNRESOLVED]
    oth_pred = mro[UNRESOLVED]
    print(f"  own 24, else locality 24: "
          f"{(oth_pred.sub(oth_actual).abs().mul(test_w).sum() / test_w.sum()):.4f} "
          f"(actual national share {oth_actual.mean():.2%})")

    print(f"\nmatched (has own 24 history): {msk.sum()}, unmatched: {(~msk).sum()}")
    print(f"unmatched boxes whose locality is also absent from election 24, "
          f"filled with the national e24 share: {n_fallback}")

    print("\nper-family weighted MAE (forecast = most-recent own, else locality-24):")
    fam = pd.DataFrame({
        "forecast": wmae(mro_f, actual_f, test_w),
        "locality 24": wmae(loc24_f, actual_f, test_w),
        "box own 24": wmae(methods["box own 24 (matched only)"], actual_f, test_w),
    })
    print(fam.round(4).to_string())

    print("\nstructural break: mean family-share shift per transition (matched boxes):")
    shift_rows = []
    for e in [22, 23, 24, 25]:
        if e == 21:
            continue
        s_prev, s_cur = shares[e - 1], shares[e]
        idx = s_cur.index.intersection(s_prev.index)
        shift_rows.append([e - 1, e] + (s_cur.loc[idx] - s_prev.loc[idx]).mean().round(4).tolist())
    shifts = pd.DataFrame(shift_rows, columns=["from", "to"] + ALL_FAMILIES).set_index(["from", "to"])
    print(shifts.to_string())
    print("\nconclusion: each transition has a distinct national swing signature; "
          "a model learned on 21-24 cannot anticipate the 24->25 swing "
          "(hence the ridge model trails the simple baselines).")

    print("\nvalue of knowing the national swing (matched boxes, per-family MAE):")
    n_m = actual_f[msk].mean()
    n24_m = own24_f[msk].mean()
    swing = pd.DataFrame({
        "own 24": wmae(own24_f[msk], actual_f[msk], test_w[msk]),
        "own 24 + oracle swing": wmae(own24_f[msk].add(n_m - n24_m, axis=1),
                                      actual_f[msk], test_w[msk]),
    }, index=FAMILIES)
    print(swing.round(4).to_string())
    print(f"mean family MAE: own-24 = {swing['own 24'].mean():.4f}, "
          f"+ oracle swing = {swing['own 24 + oracle swing'].mean():.4f}")

    def weighted_group_mean(err, groups, w):
        return pd.Series(err).groupby(pd.Series(groups).values).apply(
            lambda g: (g * w[g.index]).sum() / w[g.index].sum())

    print("\nerror by locality type (forecast, weighted mean family MAE):")
    c = census.copy()
    c["rel_q"] = pd.cut(c["hh_MidatDatiyut_Name"], [0, 1, 2, 3, 4, np.inf],
                        labels=["1-secular", "2-traditional", "3-religious",
                                "4-haredi", "5-unclassified"])
    c["pop_q"] = pd.qcut(c["pop_total"], 4, labels=None, duplicates="drop")
    meta = c[["rel_q", "pop_q"]].reindex(test.index.get_level_values(0))
    err = mro_f.sub(actual_f).abs().sum(axis=1) / len(FAMILIES)
    for col, lab in [("rel_q", "religiosity"), ("pop_q", "locality size")]:
        g = weighted_group_mean(err, meta[col].values, test_w)
        print(f"  {lab}: " + ", ".join(f"{i}={v:.4f}" for i, v in g.items()))

    nb = test.groupby(level=0).size()
    gbins = pd.cut(nb.reindex(test.index.get_level_values(0)).values,
                   [0, 1, 4, 9, np.inf], labels=["1", "2-4", "5-9", "10+"])
    g = weighted_group_mean(err, np.asarray(gbins), test_w)
    cnt = err.groupby(np.asarray(gbins)).size()
    print("  boxes per locality: " + ", ".join(
        f"{i}={v:.4f} (n={cnt[i]})" for i, v in g.items()))

    out = test.reset_index()[["locality_code", "kalpi", "valid"]]
    out["locality_name"] = (long.drop_duplicates("locality_code")
                            .set_index("locality_code")["locality_name"].reindex(out["locality_code"]).values)
    for f in ALL_FAMILIES:
        out[f"actual_{f}"] = shares[TEST].reindex(test.index)[f].values
        out[f"pred_{f}"] = mro[f].values
        out[f"loc24_{f}"] = loc24[f].values
    OUT.mkdir(exist_ok=True)
    out.to_parquet(OUT / "e25_predictions.parquet", index=False)
    print(f"\nwrote: {OUT / 'e25_predictions.parquet'}")


if __name__ == "__main__":
    main()
