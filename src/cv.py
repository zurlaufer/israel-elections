from pathlib import Path

import numpy as np
import pandas as pd

from families import ALL_FAMILIES, FAMILIES
from forecast import (
    CENSUS_FEATS,
    ELECTIONS,
    assemble,
    election_turnout,
    election_votes,
    fit_ridge,
    predict_linear,
    standardize,
    wmae,
)
from predict import loc_family_shares

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

TEST_ELECTIONS = [22, 23, 24, 25]


def load_census() -> pd.DataFrame:
    c = pd.read_csv(PROC / "locality_demographics.csv").set_index("locality_code")
    for f in ["pop_total", "employeesAnnual_medWage"]:
        c[f] = np.log1p(c[f])
    return c[CENSUS_FEATS].apply(lambda s: s.fillna(s.median()))


def fit_fold(train: pd.DataFrame):
    feat_cols = [c for c in train.columns
                 if c not in ("valid", "target_turnout")
                 and not c.startswith("target_")]
    X = train[feat_cols].values.astype(float)
    Y = train[[f"target_{f}" for f in FAMILIES]].values
    w = train["valid"].values
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = np.column_stack([np.ones(len(X)), standardize(X, mu, sd)])
    W = fit_ridge(Xs, Y, w)
    return feat_cols, mu, sd, W


def main() -> None:
    long = pd.read_parquet(PROC / "results_long.parquet")
    census = load_census()
    shares = {e: election_votes(long, e) for e in ELECTIONS}
    turns = {e: election_turnout(long, e) for e in ELECTIONS}
    loc_shares = {e: loc_family_shares(long, e) for e in ELECTIONS}
    frames = {e: assemble(long, census, e, loc_shares, turns, shares) for e in ELECTIONS}

    rows = []
    for T in TEST_ELECTIONS:
        prev = T - 1
        test = frames[T]
        actual = test[[f"target_{f}" for f in FAMILIES]].rename(
            columns={f"target_{f}": f for f in FAMILIES})
        w = test["valid"]
        msk = test["has_lag1"] == 1

        own = shares[prev].reindex(test.index)
        loc = loc_shares[prev].reindex(test.index.get_level_values(0))
        loc.index = test.index
        nat_prev = pd.Series([shares[prev][f].mean() for f in ALL_FAMILIES],
                             index=ALL_FAMILIES)
        mro = own.where(msk).fillna(loc).fillna(nat_prev)
        national = actual.mean()
        equal = pd.Series(1 / len(FAMILIES), index=FAMILIES)

        preds = {
            "own prev, else locality prev": mro[FAMILIES],
            "box own prev (matched only)": own[FAMILIES].where(msk),
            "locality prev": loc[FAMILIES],
            "national average": pd.DataFrame([national] * len(test), index=test.index),
            "equal shares": pd.DataFrame([equal] * len(test), index=test.index),
        }

        train_e = [e for e in ELECTIONS if 22 <= e < T]
        if train_e:
            tr = pd.concat([frames[e] for e in train_e])
            feat_cols, mu, sd, W = fit_fold(tr)
            Z = np.column_stack([np.ones(len(test)),
                                 standardize(test[feat_cols].values.astype(float), mu, sd)])
            preds["ridge model"] = pd.DataFrame(predict_linear(Z, W), index=test.index,
                                                columns=FAMILIES)
        else:
            tr = None

        for name, pred in preds.items():
            per = wmae(pred, actual, w)
            rows.append({
                "transition": f"{prev}->{T}",
                "method": name,
                "all": per.mean(),
                "matched": wmae(pred[msk], actual[msk], w[msk]).mean(),
                "unmatched": wmae(pred[~msk], actual[~msk], w[~msk]).mean(),
                "n_train_transitions": 0 if tr is None else len(train_e),
            })

    df = pd.DataFrame(rows)
    piv = df.pivot(index="method", columns="transition", values="all")
    order = ["own prev, else locality prev", "box own prev (matched only)",
             "locality prev", "ridge model", "national average", "equal shares"]
    piv = piv.reindex([o for o in order if o in piv.index])
    piv["avg"] = piv.mean(axis=1)
    piv = piv.sort_values("avg")

    print("leave-one-election-out CV: weighted mean family MAE on all boxes with votes")
    print(f"({len(FAMILIES)} mapped families, residual '{ALL_FAMILIES[-1]}' excluded)")
    print("forecast for election T uses only elections < T\n")
    print(piv.round(4).to_string())

    print("\nmatched / unmatched split per transition (production method):")
    sub = df[df["method"] == "own prev, else locality prev"]
    print(sub[["transition", "all", "matched", "unmatched"]].round(4).to_string(index=False))

    print("\nper-family MAE of the production method, by transition:")
    fam_rows = []
    for T in TEST_ELECTIONS:
        prev = T - 1
        test = frames[T]
        actual = test[[f"target_{f}" for f in FAMILIES]].rename(
            columns={f"target_{f}": f for f in FAMILIES})
        w = test["valid"]
        msk = test["has_lag1"] == 1
        own = shares[prev].reindex(test.index)
        loc = loc_shares[prev].reindex(test.index.get_level_values(0))
        loc.index = test.index
        nat_prev = pd.Series([shares[prev][f].mean() for f in ALL_FAMILIES],
                             index=ALL_FAMILIES)
        mro = own.where(msk).fillna(loc).fillna(nat_prev)[FAMILIES]
        fam_rows.append(wmae(mro, actual, w).rename(f"{prev}->{T}"))
    print(pd.concat(fam_rows, axis=1).round(4).to_string())

    best = piv.index[0]
    print(f"\nbest method on average: {best}")
    prod = "own prev, else locality prev"
    cols = [c for c in piv.columns if c != "avg"]
    print(f"'{prod}' beats 'locality prev' in "
          f"{(piv.loc[prod, cols] < piv.loc['locality prev', cols]).sum()}/{len(cols)} "
          f"held-out elections")
    print(f"'box own prev' beats 'locality prev' on matched boxes in "
          f"{(piv.loc['box own prev (matched only)', cols] < piv.loc['locality prev', cols]).sum()}"
          f"/{len(cols)} (matched-only row, not comparable to the all-box 'avg')")
    print(f"\nwrote: {PROC / 'cv_summary.csv'}")
    df.to_csv(PROC / "cv_summary.csv", index=False)


if __name__ == "__main__":
    main()
