import json
from pathlib import Path

import pandas as pd
import requests

API = "https://data.gov.il/api/3/action/datastore_search"

CANDIDATES_RESOURCES = {
    21: "597e0059-099e-4200-9131-b3ba645bc685",
    22: "ae484c34-86a5-4dd7-8462-a22a366287cc",
    23: "83d16550-75af-4c47-9f2c-dfeba4394fb1",
    24: "0f07d5bf-5b8d-4c28-bf37-e69459ddb4ed",
}

NAME_PATTERNS = [
    ("הליכוד", "Likud"),
    ("יש עתיד", "Yesh Atid"),
    ("העבודה", "Labor"),
    ("מרצ", "Meretz"),
    ("המחנה הדמוקרטי", "Meretz"),
    ("התאחדות הספרדים", "Shas"),
    ("הספרדים", "Shas"),
    ("ש\"ס", "Shas"),
    ("אגודת ישראל", "UTJ"),
    ("דגל התורה", "UTJ"),
    ("יהדות התורה", "UTJ"),
    ("הרשימה המשותפת", "Joint List"),
    ("חד\"ש", "Hadash-Taal"),
    ("החזית הדמוקרטית", "Hadash-Taal"),
    ("תע\"ל", "Hadash-Taal"),
    ("הערבית המאוחדת", "Ra'am"),
    ("בל\"ד", "Balad"),
    ("ישראל ביתנו", "Yisrael Beiteinu"),
    ("כחול לבן", "Blue and White"),
    ("הציונות הדתית", "Religious Zionism"),
    ("ימינה", "Yamina"),
    ("הימין החדש", "New Right"),
    ("איחוד מפלגות הימין", "URP"),
    ("עוצמה יהודית", "Otzma Yehudit"),
    ("כולנו", "Kulanu"),
    ("האיחוד הלאומי", "Jewish Home"),
    ("הבית היהודי", "Jewish Home"),
    ("תקווה חדשה", "New Hope"),
    ("המחנה הממלכתי", "National Unity"),
    ("כלכלית", "New Economic Party"),
    ("חופש כלכלי", "Economic Freedom"),
    ("הפיראטים", "Pirates"),
    ("צומת", "Tzomet"),
    ("הישראלים", "HaYisraelim"),
    ("דמוקרטית", "Democratic Party"),
    ("מען", "Ma'an"),
    ("קמ\"ה", "Kamah"),
    ("משפט צדק", "Justice Party"),
    ("הבלתי אפשרי", "The Impossible"),
    ("הגוש התנכ\"י", "Biblical Bloc"),
    ("אנחנו", "Anachnu"),
    ("הלב היהודי", "Jewish Heart"),
    ("עולם חדש", "New World"),
    ("שמע", "Shma Party"),
    ("סדר חדש", "New Order"),
    ("רפא", "Rapha"),
    ("התקווה לשינוי", "Hope for Change"),
    ("עם שלם", "Am Shalem"),
    ("כבוד האדם", "Human Dignity"),
    ("המפץ החברתי", "Social Breakthrough"),
    ("ברית השותפות", "Partnership Alliance"),
    ("מנהיגות חברתית", "Social Leadership"),
    ("אני ואתה", "I and You"),
    ("הכלכלית", "New Economic Party"),
    ("גימלאים", "Pensioners"),
    ("להשפיע", "Koah Lhashpia"),
    ("קדום מעמד הפרט", "Kamah"),
    ("עצמנו", "Independent"),
    ("עצמאים וליברלים", "Independent"),
    ("חץ", "Chetz"),
    ("דעם", "Da'am"),
    ("כלכלה ירוקה", "Da'am"),
    ("מנהיגות", "Leadership"),
    ("השיח החדש", "New Discourse"),
]

MANUAL_25 = {
    "מחל": "Likud",
    "פה": "Yesh Atid",
    "ט": "Religious Zionism",
    "כן": "National Unity",
    "שס": "Shas",
    "ג": "UTJ",
    "ל": "Yisrael Beiteinu",
    "עם": "Ra'am",
    "ום": "Hadash-Taal",
    "אמת": "Labor",
    "מרצ": "Meretz",
    "ד": "Balad",
    "ב": "Jewish Home",
    "אצ": "Economic Freedom",
}

CONTROL_COLS = {
    "_id", "סמל ועדה", "ברזל", "שם ישוב", "סמל ישוב", "קלפי", "מספר קלפי",
    "ריכוז", "שופט", "בזב", "מצביעים", "פסולים", "כשרים", "ת. עדכון",
}


def fetch_candidates(kn: int) -> list[dict]:
    rid = CANDIDATES_RESOURCES[kn]
    r = requests.get(API, params={"resource_id": rid, "limit": 100000}, timeout=120)
    r.raise_for_status()
    return r.json()["result"]["records"]


def resolve_name(name: str) -> str:
    for needle, english in NAME_PATTERNS:
        if needle in name:
            return english
    return name


def build_for(kn: int) -> dict[str, tuple[str, str]]:
    recs = fetch_candidates(kn)
    mapping: dict[str, tuple[str, str]] = {}
    for rec in recs:
        letter = str(rec.get("אותיות") or "").strip()
        name = str(rec.get("שם הרשימה") or "").strip('"')
        if not letter or letter in mapping:
            continue
        mapping[letter] = (name, resolve_name(name))
    return mapping


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw = root / "data" / "raw"
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    all_maps: dict[int, dict[str, tuple[str, str]]] = {}
    unresolved: set[str] = set()
    for kn in [21, 22, 23, 24]:
        m = build_for(kn)
        all_maps[kn] = m
        print(f"knesset {kn}: {len(m)} lists mapped")
        for letter, (he, en) in m.items():
            if he == en:
                unresolved.add(f"{kn}:{letter}={he}")

    rows = []
    for kn, m in all_maps.items():
        for letter, (he, en) in m.items():
            rows.append({"knesset": kn, "letter": letter, "party": en, "party_he": he})
    for letter, en in MANUAL_25.items():
        rows.append({"knesset": 25, "letter": letter, "party": en, "party_he": ""})
    parties = pd.DataFrame(rows)
    parties.to_csv(proc / "parties.csv", index=False)

    validation = []
    for kn in [21, 22, 23, 24, 25]:
        path = raw / f"knesset{kn}.json"
        if not path.exists():
            continue
        df = pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
        party_cols = [c for c in df.columns if c not in CONTROL_COLS]
        totals = df[party_cols].sum().sort_values(ascending=False)
        for letter, votes in totals.head(12).items():
            name = all_maps.get(kn, {}).get(letter, (None, None))[1] or MANUAL_25.get(letter, "<UNMAPPED>")
            validation.append({"knesset": kn, "letter": letter, "votes": int(votes), "party": name})

    val = pd.DataFrame(validation)
    val.to_csv(proc / "letter_totals_validation.csv", index=False)
    print("\nTop lists per election (verification):")
    print(val.to_string(index=False))

    if unresolved:
        print("\nUNRESOLVED list names (kept as Hebrew):")
        for u in sorted(unresolved):
            print("  ", u)
    print(f"\nwrote: {proc / 'parties.csv'}")


if __name__ == "__main__":
    main()
