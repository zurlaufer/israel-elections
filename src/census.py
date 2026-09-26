import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

RELIGION = {1: "jews", 2: "muslims", 3: "druze", 4: "christians", 5: "christians", 6: "other"}

WEIGHTED_VARS = [
    "age0_19_pcnt", "age65_pcnt", "age_median", "hh_MidatDatiyut_Name",
    "employeesAnnual_medWage", "AcadmCert_pcnt", "own_pcnt",
]


def num(s):
    if s is None:
        return None
    if isinstance(s, str):
        s = s.replace(",", "").replace("%", "").strip()
        if s in ("", "-", ".."):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return float(s)


def build_profiles() -> pd.DataFrame:
    df = pd.DataFrame(json.loads((RAW / "census_2022_localities.json").read_text(encoding="utf-8")))
    for col in ["pop_approx"] + WEIGHTED_VARS:
        df[col] = df[col].apply(num)

    sa = df[df["StatArea"].notna()].copy()
    sa = sa[sa["pop_approx"].notna() & (sa["pop_approx"] > 0)]
    if len(sa):
        sa["pct_share"] = sa.groupby("LocalityCode")["pop_approx"].transform(lambda s: s / s.sum())
        rows = []
        for code, g in sa.groupby("LocalityCode"):
            d = {
                "locality_code": int(code),
                "locality_name": g["LocNameHeb"].dropna().iloc[0],
                "pop_total": g["pop_approx"].sum(),
            }
            for r in [1, 2, 3, 4, 5, 6]:
                key = "pct_" + RELIGION[r]
                d[key] = g.loc[g["religion"] == r, "pct_share"].sum()
            for var in WEIGHTED_VARS:
                d[var] = (g[var] * g["pop_approx"]).sum() / g["pop_approx"].sum()
            rows.append(d)
        profiles = pd.DataFrame(rows)
    else:
        profiles = pd.DataFrame(columns=["locality_code", "locality_name", "pop_total"])

    loc = df[df["StatArea"].isna()].copy()
    loc = loc[loc["LocalityCode"].notna()]
    covered = set(profiles["locality_code"])
    missing = loc[~loc["LocalityCode"].astype(int).isin(covered)]
    if len(missing):
        rows = []
        for _, r in missing.iterrows():
            rows.append({
                "locality_code": int(r["LocalityCode"]),
                "locality_name": r["LocalityNameHeb"] if "LocalityNameHeb" in r else r.get("LocNameHeb"),
                "pop_total": r["pop_approx"],
                "pct_jews": 1.0 if r["religion"] == 1 else 0.0,
                "pct_muslims": 1.0 if r["religion"] == 2 else 0.0,
                "pct_druze": 1.0 if r["religion"] == 3 else 0.0,
                "pct_christians": 1.0 if r["religion"] in (4, 5) else 0.0,
                "pct_other": 1.0 if r["religion"] == 6 else 0.0,
                **{var: r[var] for var in WEIGHTED_VARS},
            })
        small = pd.DataFrame(rows)
        profiles = pd.concat([profiles, small], ignore_index=True)

    profiles["locality_code"] = profiles["locality_code"].astype(int)
    return profiles


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    profiles = build_profiles()
    profiles = profiles.sort_values("locality_code").reset_index(drop=True)
    profiles.to_csv(PROC / "locality_demographics.csv", index=False)

    long = pd.read_parquet(PROC / "results_long.parquet")
    loc_codes = set(long["locality_code"].dropna().astype(int))
    covered = set(profiles["locality_code"])
    missing = loc_codes - covered
    print(f"census profiles: {len(profiles)} localities")
    print(f"election localities: {len(loc_codes)} | covered: {len(loc_codes & covered)} | missing: {len(missing)}")
    if missing:
        names = (long.drop_duplicates("locality_code").set_index("locality_code")["locality_name"].
                 loc[sorted(missing)].head(10).to_dict())
        print("sample missing localities:", names)
    print(f"wrote: {PROC / 'locality_demographics.csv'}")


if __name__ == "__main__":
    main()
