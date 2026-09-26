import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

ELECTIONS = {
    21: "2019-04-09",
    22: "2019-09-17",
    23: "2020-03-02",
    24: "2021-03-23",
    25: "2022-11-01",
}

CONTROL_COLS = {
    "_id", "סמל ועדה", "ברזל", "שם ישוב", "סמל ישוב", "קלפי", "מספר קלפי",
    "ריכוז", "שופט", "בזב", "מצביעים", "פסולים", "כשרים",
}

SPECIAL_NAMES = {"מעטפות חיצוניות", "מעטפות כפולות", "מעטפות חוץ"}


def load_raw(kn: int) -> pd.DataFrame:
    recs = json.loads((RAW / f"knesset{kn}.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(recs).drop(columns=["_id"], errors="ignore")
    return df


def normalize_election(kn: int, df: pd.DataFrame, party_map: dict) -> pd.DataFrame:
    new_schema = "ברזל" in df.columns
    kalpi_col = "קלפי" if new_schema else "מספר קלפי"
    box_serial = df["ברזל"].astype("Int64") if new_schema else pd.Series([pd.NA] * len(df))
    committee = df["סמל ועדה"].astype("Int64") if new_schema else pd.Series([pd.NA] * len(df))

    party_cols = [c for c in df.columns if c not in CONTROL_COLS]
    parts = []
    for letter in party_cols:
        votes = pd.to_numeric(df[letter], errors="coerce").fillna(0).astype("int64")
        party = party_map.get(letter, "Other")
        parts.append(pd.DataFrame({
            "knesset": kn,
            "date": ELECTIONS[kn],
            "locality_code": pd.to_numeric(df["סמל ישוב"], errors="coerce").astype("Int64"),
            "locality_name": df["שם ישוב"].astype(str),
            "kalpi": df[kalpi_col].astype(str).str.strip(),
            "box_serial": box_serial.to_numpy(),
            "committee": committee.to_numpy(),
            "eligible": pd.to_numeric(df["בזב"], errors="coerce").fillna(0).astype("int64"),
            "voters": pd.to_numeric(df["מצביעים"], errors="coerce").fillna(0).astype("int64"),
            "invalid": pd.to_numeric(df["פסולים"], errors="coerce").fillna(0).astype("int64"),
            "valid": pd.to_numeric(df["כשרים"], errors="coerce").fillna(0).astype("int64"),
            "is_special": df["שם ישוב"].astype(str).isin(SPECIAL_NAMES),
            "letter": letter,
            "party": party,
            "votes": votes,
        }))
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    parties = pd.read_csv(PROC / "parties.csv")
    frames = []
    for kn in sorted(ELECTIONS):
        party_map = dict(zip(
            parties.loc[parties["knesset"] == kn, "letter"],
            parties.loc[parties["knesset"] == kn, "party"],
        ))
        df = load_raw(kn)
        out = normalize_election(kn, df, party_map)
        frames.append(out)
        n_boxes = df.groupby(df["שם ישוב"].astype(str) + df["קלפי" if "קלפי" in df.columns else "מספר קלפי"].astype(str)).ngroups
        print(f"knesset {kn}: {len(df)} boxes -> {len(out)} rows, {n_boxes} distinct box ids")

    long = pd.concat(frames, ignore_index=True)
    long.to_parquet(PROC / "results_long.parquet")

    per_box = long.drop_duplicates(subset=["knesset", "locality_code", "kalpi"])
    summary = per_box.groupby(["knesset", "date", "is_special"]).agg(
        boxes=("kalpi", "count"),
        eligible=("eligible", "sum"),
        voters=("voters", "sum"),
        valid=("valid", "sum"),
    ).reset_index()
    print("\nsummary:")
    print(summary.to_string(index=False))
    print(f"\nwrote: {PROC / 'results_long.parquet'}")


if __name__ == "__main__":
    main()
