import argparse
import json
from pathlib import Path
import requests

API = "https://data.gov.il/api/3/action/datastore_search"

ELECTIONS = {
    21: {"rid": "f79f9ba5-fe12-4b90-96cc-916f1b7c1c34", "date": "2019-04-09"},
    22: {"rid": "22f3a195-3a79-436c-be23-cb606bc7b398", "date": "2019-09-17"},
    23: {"rid": "3b9e911a-2e90-4587-b209-84171664056b", "date": "2020-03-02"},
    24: {"rid": "419be3b0-fd30-455a-afc0-034ec36be990", "date": "2021-03-23"},
    25: {"rid": "cc223336-07bc-485d-b160-62df92967c0a", "date": "2022-11-01"},
}

CENSUS = {
    "rid": "9a9e085f-3bc8-41df-b15f-be0daaf99e30",
    "file": "census_2022_localities.json",
}


def fetch_all(rid: str) -> list[dict]:
    out = []
    offset = 0
    while True:
        r = requests.get(API, params={"resource_id": rid, "limit": 100000, "offset": offset}, timeout=180)
        r.raise_for_status()
        res = r.json()
        if not res.get("success"):
            raise RuntimeError(f"API error: {res}")
        records = res["result"]["records"]
        out.extend(records)
        total = res["result"]["total"]
        offset += len(records)
        if offset >= total or not records:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--elections", nargs="*", type=int, default=None)
    parser.add_argument("--no-census", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    elections = args.elections or sorted(ELECTIONS)
    for kn in elections:
        info = ELECTIONS[kn]
        dest = out / f"knesset{kn}.json"
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip (exists): {dest.name}")
            continue
        records = fetch_all(info["rid"])
        dest.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
        print(f"fetched: {dest.name} ({len(records):,} rows)")

    if not args.no_census:
        dest = out / CENSUS["file"]
        if not dest.exists() or dest.stat().st_size == 0:
            records = fetch_all(CENSUS["rid"])
            dest.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
            print(f"fetched: {dest.name} ({len(records):,} rows)")
        else:
            print(f"skip (exists): {dest.name}")

    meta = {
        "elections": {str(kn): {"file": f"knesset{kn}.json", "date": v["date"]} for kn, v in ELECTIONS.items()},
        "census": CENSUS,
    }
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
