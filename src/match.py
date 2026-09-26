from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

ELECTIONS = [21, 22, 23, 24, 25]
PAIRS = [(21, 22), (22, 23), (23, 24), (24, 25), (22, 25), (21, 25)]


def load_long() -> pd.DataFrame:
    return pd.read_parquet(PROC / "results_long.parquet")


def box_table(long: pd.DataFrame) -> pd.DataFrame:
    b = (long.groupby(["knesset", "locality_code", "kalpi", "box_serial", "is_special"],
                      dropna=False)
         .agg(eligible=("eligible", "first"), voters=("voters", "first"),
              valid=("valid", "first")).reset_index())
    return b


def match_stats(b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, c in PAIRS:
        A = b[b["knesset"] == a][["locality_code", "kalpi"]]
        C = b[b["knesset"] == c][["locality_code", "kalpi"]]
        n_a, n_c = len(A), len(C)
        n_int = len(A.merge(C, on=["locality_code", "kalpi"]))
        rows.append({"pair": f"{a}-{c}", "n_a": n_a, "n_c": n_c,
                     "matched": n_int, "match_a": n_int / n_a, "match_c": n_int / n_c})
    return pd.DataFrame(rows)


def serial_stability(b: pd.DataFrame) -> None:
    print("box_serial stability across elections (matched by (locality, kalpi)):")
    for a, c in PAIRS:
        A = b[(b["knesset"] == a) & b["box_serial"].notna()][["locality_code", "kalpi", "box_serial"]]
        C = b[(b["knesset"] == c) & b["box_serial"].notna()][["locality_code", "kalpi", "box_serial"]]
        m = A.merge(C, on=["locality_code", "kalpi"])
        if len(m):
            agree = (m["box_serial_x"] == m["box_serial_y"]).mean()
            print(f"  {a}-{c}: {len(m):6d} boxes, serial agreement {agree:6.1%}")
        else:
            print(f"  {a}-{c}: 0 boxes")


def build_panel(b: pd.DataFrame, long: pd.DataFrame, elections: list[int]) -> pd.DataFrame:
    b = b[~b["is_special"] & (b["voters"] > 0)]
    present = b.groupby(["locality_code", "kalpi"])["knesset"].apply(lambda s: set(s))
    stable = set(present[present.apply(lambda s: set(elections).issubset(s))].index)

    b_s = b[b[["locality_code", "kalpi"]].apply(tuple, axis=1).isin(stable)]
    long_s = long[long[["locality_code", "kalpi"]].apply(tuple, axis=1).isin(stable)]

    votes = (long_s.pivot_table(index=["locality_code", "kalpi", "knesset"],
                                columns="letter", values="votes",
                                aggfunc="sum", fill_value=0))

    wide_parts = []
    for e in elections:
        sub = b_s[b_s["knesset"] == e].set_index(["locality_code", "kalpi"])
        sub = sub.drop(columns="knesset")
        sub.columns = [f"{c}_{e}" for c in sub.columns]
        v = votes.xs(e, level="knesset")
        v.columns = [f"votes_{e}_{c}" for c in v.columns]
        wide_parts.append(sub.join(v))
    wide = wide_parts[0]
    for w in wide_parts[1:]:
        wide = wide.join(w, how="inner")
    return wide.reset_index()


def join_election(wide: pd.DataFrame, long: pd.DataFrame, e: int) -> pd.DataFrame:
    sub = long[long["knesset"] == e]
    votes = (sub.pivot_table(index=["locality_code", "kalpi"], columns="letter",
                             values="votes", aggfunc="sum", fill_value=0))
    votes.columns = [f"votes_{e}_{c}" for c in votes.columns]
    out = wide.set_index(["locality_code", "kalpi"]).join(votes, how="left")
    return out.reset_index()


def main() -> None:
    long = load_long()
    b = box_table(long)

    print(f"boxes per election: {b.groupby('knesset').size().to_dict()}")
    print()
    print(match_stats(b).round(3).to_string(index=False))
    print()
    serial_stability(b)
    print()

    wide5 = build_panel(b, long, [21, 22, 23, 24, 25])
    wide5.to_parquet(PROC / "box_panel.parquet", index=False)
    print(f"5-election panel: {len(wide5)} boxes, {wide5['locality_code'].nunique()} localities "
          f"-> {PROC / 'box_panel.parquet'}")

    train = [21, 22, 23, 24]
    wide4 = build_panel(b, long, train)
    wide4 = join_election(wide4, long, 25)
    v25 = [c for c in wide4.columns if c.startswith("votes_25_")]
    n_with25 = wide4[v25].notna().any(axis=1).sum()
    wide4.to_parquet(PROC / "box_panel_train.parquet", index=False)
    print(f"train panel (21-24 required, election-25 joined where box survives): "
          f"{len(wide4)} boxes, {wide4['locality_code'].nunique()} localities, "
          f"{n_with25} boxes also found in election 25 "
          f"-> {PROC / 'box_panel_train.parquet'}")


if __name__ == "__main__":
    main()
