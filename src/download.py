import argparse
import hashlib
import json
from pathlib import Path

import requests

API = "https://data.gov.il/api/3/action/datastore_search"

ELECTIONS = {
    21: {
        "rid": "f79f9ba5-fe12-4b90-96cc-916f1b7c1c34",
        "date": "2019-04-09",
        "sha256": "a7d70c0e4e1f3e8cfb03d5d2c87f6f4791f0abeea3017253f3e23eb3fe87b0df",
    },
    22: {
        "rid": "22f3a195-3a79-436c-be23-cb606bc7b398",
        "date": "2019-09-17",
        "sha256": "bfc71ead3c682c58ece0ada2fde61fa9d2a74ce2e1d72dfb2c56e553a0b96cfd",
    },
    23: {
        "rid": "3b9e911a-2e90-4587-b209-84171664056b",
        "date": "2020-03-02",
        "sha256": "6d60a5b50d92573a897ac80ec0b937ec5afa6ee2892a52afd68cbad23cbf1c82",
    },
    24: {
        "rid": "419be3b0-fd30-455a-afc0-034ec36be990",
        "date": "2021-03-23",
        "sha256": "bdcb0201d1d4dae18482cff41276639295a46a844173a1fc51f8cc99af15fe46",
    },
    25: {
        "rid": "cc223336-07bc-485d-b160-62df92967c0a",
        "date": "2022-11-01",
        "sha256": "3401bfe9016106c47ff54137265df60d9dfef22dac186f86981f08232ba3523c",
    },
}

CENSUS = {
    "rid": "9a9e085f-3bc8-41df-b15f-be0daaf99e30",
    "file": "census_2022_localities.json",
    "sha256": "286548cede3cd7cd0c8a36a6afecc99c671b50335506c199bbabe326e5f52f10",
}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


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
            actual_sha = hash_file(dest)
            expected_sha = info.get("sha256")
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(
                    f"SHA256 mismatch for existing file {dest.name}: expected {expected_sha}, got {actual_sha}"
                )
            print(f"skip (exists, sha256 verified): {dest.name}")
            continue
        records = fetch_all(info["rid"])
        content = json.dumps(records, ensure_ascii=False)
        actual_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
        expected_sha = info.get("sha256")
        if expected_sha and actual_sha != expected_sha:
            raise RuntimeError(
                f"Downloaded SHA256 mismatch for {dest.name}: expected {expected_sha}, got {actual_sha}"
            )
        dest.write_text(content, encoding="utf-8")
        print(f"fetched and verified: {dest.name} ({len(records):,} rows)")

    if not args.no_census:
        dest = out / CENSUS["file"]
        if not dest.exists() or dest.stat().st_size == 0:
            records = fetch_all(CENSUS["rid"])
            content = json.dumps(records, ensure_ascii=False)
            actual_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            expected_sha = CENSUS.get("sha256")
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(
                    f"Downloaded SHA256 mismatch for {dest.name}: expected {expected_sha}, got {actual_sha}"
                )
            dest.write_text(content, encoding="utf-8")
            print(f"fetched and verified: {dest.name} ({len(records):,} rows)")
        else:
            actual_sha = hash_file(dest)
            expected_sha = CENSUS.get("sha256")
            if expected_sha and actual_sha != expected_sha:
                raise RuntimeError(
                    f"SHA256 mismatch for existing file {dest.name}: expected {expected_sha}, got {actual_sha}"
                )
            print(f"skip (exists, sha256 verified): {dest.name}")

    meta = {
        "elections": {
            str(kn): {
                "file": f"knesset{kn}.json",
                "date": v["date"],
                "sha256": v.get("sha256") or (hash_file(out / f"knesset{kn}.json") if (out / f"knesset{kn}.json").exists() else None),
            }
            for kn, v in ELECTIONS.items()
        },
        "census": {
            "rid": CENSUS["rid"],
            "file": CENSUS["file"],
            "sha256": CENSUS.get("sha256") or (hash_file(out / CENSUS["file"]) if (out / CENSUS["file"]).exists() else None),
        },
    }
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
