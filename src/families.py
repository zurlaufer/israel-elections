from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"

UNRESOLVED = "Other"

FAMILIES = ["Likud", "Center", "Labor", "Shas", "UTJ", "YB", "Meretz", "Arab",
            "Religious Right"]
ALL_FAMILIES = FAMILIES + [UNRESOLVED]

PARTY_FAMILY = {
    "Likud": "Likud",
    "Yesh Atid": "Center",
    "Blue and White": "Center",
    "National Unity": "Center",
    "Kulanu": "Center",
    "New Hope": "Center",
    "Labor": "Labor",
    "Shas": "Shas",
    "UTJ": "UTJ",
    "Yisrael Beiteinu": "YB",
    "Meretz": "Meretz",
    "Joint List": "Arab",
    "Hadash-Taal": "Arab",
    "Ra'am": "Arab",
    "Balad": "Arab",
    "Da'am": "Arab",
    "United Arab List": "Arab",
    "Yamina": "Religious Right",
    "Religious Zionism": "Religious Right",
    "New Right": "Religious Right",
    "Otzma Yehudit": "Religious Right",
    "URP": "Religious Right",
    "Jewish Home": "Religious Right",
    "Noam": "Religious Right",
}


def load_party_letters() -> pd.DataFrame:
    p = pd.read_csv(PROC / "parties.csv")
    p = p[p["party"] != UNRESOLVED]
    p["family"] = p["party"].map(PARTY_FAMILY)
    return p


def family_letters(parties: pd.DataFrame | None = None) -> dict[int, dict[str, list[str]]]:
    p = load_party_letters() if parties is None else parties
    out: dict[int, dict[str, list[str]]] = {}
    for kn, g in p.groupby("knesset"):
        out[int(kn)] = {f: sorted(g.loc[g["family"] == f, "letter"]) for f in FAMILIES}
    return out


def audit(long: pd.DataFrame | None = None) -> pd.DataFrame:
    if long is None:
        long = pd.read_parquet(PROC / "results_long.parquet")
    d = long[~long["is_special"]]
    fl = family_letters()
    rows = []
    for kn, g in d.groupby("knesset"):
        tot_letter = g.groupby("letter").votes.sum()
        total = tot_letter.sum()
        covered = set()
        for ls in fl[int(kn)].values():
            covered |= set(ls)
        r = {"knesset": int(kn), "n_letters": len(tot_letter), "total": int(total)}
        for f in FAMILIES:
            ls = [l for l in fl[int(kn)][f] if l in tot_letter.index]
            v = sum(tot_letter.get(l, 0) for l in ls)
            r[f] = round(100 * v / total, 2)
        named = sum(r[f] for f in FAMILIES)
        unmapped_letters = sorted(set(tot_letter.index) - covered)
        r[UNRESOLVED] = round(100 - named, 2)
        r["unmapped_letters"] = len(unmapped_letters)
        r["unmapped_pct"] = round(
            100 * sum(tot_letter.get(l, 0) for l in unmapped_letters) / total, 2)
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    print("letter -> party -> family crosswalk:")
    p = load_party_letters()
    for kn, g in p.groupby("knesset"):
        named = g.dropna(subset=["family"])
        print(f"  e{kn}: " + ", ".join(
            f"{r.letter}={r.party}({r.family})" for r in named.sort_values("party").itertuples()))
        miss = sorted(set(p[p.knesset == kn].letter) - set(named.letter))
        if miss:
            print(f"       unmapped letters: {miss}")

    print("\nnational share % by family (unmapped letters -> Other):")
    print(audit().to_string(index=False))
