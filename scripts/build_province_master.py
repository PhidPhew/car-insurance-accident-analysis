"""Build config/province_master.csv, the 77-province conformed dimension.

Run: ``uv run python scripts/build_province_master.py``

Province codes and names are extracted from thailand_rainfall_2021_2026_combined.csv
(Thai names) and thailand_heavy_rain_R35mm.csv (English names), per CLAUDE.md's
instruction to inspect the rainfall/R35mm files first and follow whatever coding
system they use rather than inventing one. Both files independently confirm the
same 77 two-digit codes (10-96, the standard Thai Ministry of Interior geocode
scheme), so this script cross-validates one against the other rather than
trusting either blindly.

Region assignment (6 regions) is NOT in any raw source file and needed research:
Thailand has multiple co-existing "official" regional classifications (the
National Geographic Committee's 6-region system from 1977, and the Meteorological
Department's 5-region system, among others) that disagree on where several
provinces belong (Tak, Phetchaburi, Prachuap Khiri Khan in particular). This
script uses the 6-region NGC system since CLAUDE.md asks for "6 regions"
specifically. The exact province lists per region were cross-checked against two
independent sources and then verified to reconcile exactly against all 77 codes
extracted from this project's own data (see REGION_BY_PROVINCE_CODE below) --
every code is accounted for exactly once, which the initial single-source lookup
did not achieve on its own.

Sources consulted (2026-08-12):
- https://en.wikipedia.org/wiki/Regions_of_Thailand (Northern, Northeastern,
  Central, Eastern, Western, Southern province lists; incomplete on lower-north
  provinces and ambiguous on Phetchaburi/Prachuap Khiri Khan)
- Thai-language web search results for "คณะกรรมการภูมิศาสตร์แห่งชาติ 6 ภาค"
  giving per-region province counts (North 9, Central 21, Northeast 19-20,
  East 7, West 5, South 14) that pinned down the West region as exactly 5
  provinces (Tak, Kanchanaburi, Ratchaburi, Phetchaburi, Prachuap Khiri Khan),
  resolving the ambiguity the English-language source left open.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from rri.utils.logging import configure_logging, get_logger  # noqa: E402

log = get_logger(__name__)

# Province code -> region, covering all 77 codes exactly once. See module
# docstring for sourcing. Bangkok (10) is grouped with Central per every
# source consulted.
REGION_BY_PROVINCE_CODE: dict[str, str] = {
    # Central (21 provinces + Bangkok = 22 codes)
    "10": "Central",  # Bangkok
    "11": "Central",  # Samut Prakan
    "12": "Central",  # Nonthaburi
    "13": "Central",  # Pathum Thani
    "14": "Central",  # Phra Nakhon Si Ayutthaya
    "15": "Central",  # Ang Thong
    "16": "Central",  # Lopburi
    "17": "Central",  # Sing Buri
    "18": "Central",  # Chai Nat
    "19": "Central",  # Saraburi
    "26": "Central",  # Nakhon Nayok
    "60": "Central",  # Nakhon Sawan
    "61": "Central",  # Uthai Thani
    "62": "Central",  # Kamphaeng Phet
    "64": "Central",  # Sukhothai
    "65": "Central",  # Phitsanulok
    "66": "Central",  # Phichit
    "67": "Central",  # Phetchabun
    "72": "Central",  # Suphan Buri
    "73": "Central",  # Nakhon Pathom
    "74": "Central",  # Samut Sakhon
    "75": "Central",  # Samut Songkhram
    # East (7 provinces)
    "20": "East",  # Chon Buri
    "21": "East",  # Rayong
    "22": "East",  # Chanthaburi
    "23": "East",  # Trat
    "24": "East",  # Chachoengsao
    "25": "East",  # Prachin Buri
    "27": "East",  # Sa Kaeo
    # Northeast / Isan (20 provinces)
    "30": "Northeast",  # Nakhon Ratchasima
    "31": "Northeast",  # Buriram
    "32": "Northeast",  # Surin
    "33": "Northeast",  # Sisaket
    "34": "Northeast",  # Ubon Ratchathani
    "35": "Northeast",  # Yasothon
    "36": "Northeast",  # Chaiyaphum
    "37": "Northeast",  # Amnat Charoen
    "38": "Northeast",  # Bueng Kan
    "39": "Northeast",  # Nong Bua Lamphu
    "40": "Northeast",  # Khon Kaen
    "41": "Northeast",  # Udon Thani
    "42": "Northeast",  # Loei
    "43": "Northeast",  # Nong Khai
    "44": "Northeast",  # Maha Sarakham
    "45": "Northeast",  # Roi Et
    "46": "Northeast",  # Kalasin
    "47": "Northeast",  # Sakon Nakhon
    "48": "Northeast",  # Nakhon Phanom
    "49": "Northeast",  # Mukdahan
    # North (9 provinces)
    "50": "North",  # Chiang Mai
    "51": "North",  # Lamphun
    "52": "North",  # Lampang
    "53": "North",  # Uttaradit
    "54": "North",  # Phrae
    "55": "North",  # Nan
    "56": "North",  # Phayao
    "57": "North",  # Chiang Rai
    "58": "North",  # Mae Hong Son
    # West (5 provinces)
    "63": "West",  # Tak
    "70": "West",  # Ratchaburi
    "71": "West",  # Kanchanaburi
    "76": "West",  # Phetchaburi
    "77": "West",  # Prachuap Khiri Khan
    # South (14 provinces)
    "80": "South",  # Nakhon Si Thammarat
    "81": "South",  # Krabi
    "82": "South",  # Phang Nga
    "83": "South",  # Phuket
    "84": "South",  # Surat Thani
    "85": "South",  # Ranong
    "86": "South",  # Chumphon
    "90": "South",  # Songkhla
    "91": "South",  # Satun
    "92": "South",  # Trang
    "93": "South",  # Phatthalung
    "94": "South",  # Pattani
    "95": "South",  # Yala
    "96": "South",  # Narathiwat
}


def load_thai_names() -> dict[str, str]:
    path = settings.raw_dir / "thailand_rainfall_2021_2026_combined.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["Province_ID"]: row["Province_NameTH"].strip() for row in reader}


def load_english_names() -> dict[str, str]:
    path = settings.raw_dir / "thailand_heavy_rain_R35mm.csv"
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["ProvinceID"]: row["Province_Name"].strip() for row in reader}


def build_master_rows() -> list[dict[str, str]]:
    thai_names = load_thai_names()
    english_names = load_english_names()

    codes_thai = set(thai_names)
    codes_english = set(english_names)
    codes_region = set(REGION_BY_PROVINCE_CODE)

    if codes_thai != codes_english:
        raise ValueError(
            f"rainfall and R35mm disagree on province codes: "
            f"only in rainfall={codes_thai - codes_english}, "
            f"only in R35mm={codes_english - codes_thai}"
        )
    if codes_thai != codes_region:
        raise ValueError(
            f"REGION_BY_PROVINCE_CODE doesn't cover exactly the data's codes: "
            f"missing from region map={codes_thai - codes_region}, "
            f"extra in region map={codes_region - codes_thai}"
        )

    rows = [
        {
            "province_id": code,
            "province_th": thai_names[code],
            "province_en": english_names[code],
            "region": REGION_BY_PROVINCE_CODE[code],
        }
        for code in sorted(codes_thai, key=int)
    ]
    if len(rows) != 77:
        raise ValueError(f"expected exactly 77 provinces, got {len(rows)}")
    return rows


def main() -> int:
    configure_logging()
    rows = build_master_rows()

    out_path = settings.config_dir / "province_master.csv"
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = ["province_id", "province_th", "province_en", "region"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    region_counts: dict[str, int] = {}
    for row in rows:
        region_counts[row["region"]] = region_counts.get(row["region"], 0) + 1

    log.info("province_master_written", path=str(out_path), rows=len(rows), by_region=region_counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
