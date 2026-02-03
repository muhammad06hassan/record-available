# ===================== IMPORTS =====================

import pandas as pd
import re
import logging
from tqdm import tqdm

# ===================== FILES =====================
DB_FILE = "CVN101_Data_07JAN2026.csv"
CDISC_FILE = "CVN-101_AK.csv"

OUT_DB_TO_CDISC = "db_to_cdisc_recon.csv"
OUT_CDISC_TO_DB = "cdisc_to_db_recon.csv"
LOG_FILE = "two_way_recon.log"

# ===================== MAPPING (CDISC short -> DB full) =====================
CDISC_QSCAT_TO_FULL = {
    "AIMS01": "Alberta Infant Motor Scale (AIMS)",
    "DEVM01": "CDC Developmental Milestones Checklist",
    "PRE1": "Pre-Examination Questionnaire",
    "HEAD01": "Measurement of Head Circumference",
    "GMFM88": "Gross Motor Function Measure 88-items (GMFM-88)",
    "HINE02": "Hammersmith Infant Neurological Examination, Section 2 (HINE-2)",
    "RSS01": "Response to Sensory Stimuli",
    "IMP01": "Infant Motor Profile (IMP)",
    "POST1": "Post Examination Questionnaire",
    "Bayley-4": "Bayley-4 Cognitive, Language and Motor",
    "TIMP1": "Test of Infant Motor Performance Screening Items (TIMPSI)",
    "Vineland-3 Comprehensive": "Vineland-3 Comprehensive Interview Form"
}

# ===================== LOGGING =====================
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ===================== HELPERS =====================
def norm(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_id(x) -> str:
    s = norm(x)
    if s.endswith(".0"):
        s = s[:-2]
    return s

def main():
    db = pd.read_csv(DB_FILE, dtype=str).fillna("")
    cdisc = pd.read_csv(CDISC_FILE, dtype=str).fillna("")

    # Validate required columns
    for col in ["USUBJID", "VISIT", "QSCAT"]:
        if col not in db.columns:
            raise SystemExit(f"Missing column '{col}' in DB file: {DB_FILE}")
        if col not in cdisc.columns:
            raise SystemExit(f"Missing column '{col}' in CDISC file: {CDISC_FILE}")

    # ------------------ Normalize DB ------------------
    db["USUBJID_N"] = db["USUBJID"].map(norm_id)
    db["VISIT_N"] = db["VISIT"].map(norm)
    db["QSCAT_FULL_N"] = db["QSCAT"].map(norm)

    # DB counts per key (sometimes DB export can have duplicates)
    db_counts = (
        db.groupby(["USUBJID_N", "VISIT_N", "QSCAT_FULL_N"])
          .size()
          .to_dict()
    )
    db_key_set = set(db_counts.keys())

    # ------------------ Normalize CDISC ------------------
    cdisc["USUBJID_N"] = cdisc["USUBJID"].map(norm_id)
    cdisc["VISIT_N"] = cdisc["VISIT"].map(norm)
    cdisc["QSCAT_SHORT_N"] = cdisc["QSCAT"].map(norm)

    # Map CDISC short -> full
    missing_map = set()
    cdisc["QSCAT_FULL_N"] = ""
    for i, short in enumerate(cdisc["QSCAT_SHORT_N"].tolist()):
        full = CDISC_QSCAT_TO_FULL.get(short, "")
        if not full:
            missing_map.add(short)
        cdisc.at[i, "QSCAT_FULL_N"] = full

    if missing_map:
        logging.warning(f"Missing CDISC mappings for: {sorted(missing_map)}")

    # Keep only mappable CDISC rows
    cdisc_ok = cdisc[cdisc["QSCAT_FULL_N"] != ""].copy()

    # CDISC counts per key (this is your “question rows count per record”)
    cdisc_counts = (
        cdisc_ok.groupby(["USUBJID_N", "VISIT_N", "QSCAT_FULL_N"])
                .size()
                .to_dict()
    )
    cdisc_key_set = set(cdisc_counts.keys())

    # ============================================================
    # 1) DB -> CDISC (one DB row = one record, but compare by key)
    # ============================================================
    db_to_cdisc_rows = []
    for _, r in tqdm(db.iterrows(), total=len(db), desc="DB -> CDISC", unit="row"):
        key = (r["USUBJID_N"], r["VISIT_N"], r["QSCAT_FULL_N"])
        if key in cdisc_key_set:
            status = "AVAILABLE_IN_CDISC"
            cdisc_row_count = cdisc_counts.get(key, 0)
            comment = "FOUND_IN_CDISC"
        else:
            status = "MISSING_IN_CDISC"
            cdisc_row_count = 0
            comment = "NO_MATCHING_CDISC_RECORD_KEY"

        out = r.to_dict()
        out["RECON_STATUS"] = status
        out["CDISC_ROW_COUNT_FOR_KEY"] = cdisc_row_count
        out["COMMENT"] = comment
        db_to_cdisc_rows.append(out)

    db_to_cdisc_df = pd.DataFrame(db_to_cdisc_rows)
    db_to_cdisc_df.drop(columns=["USUBJID_N", "VISIT_N", "QSCAT_FULL_N"], inplace=True, errors="ignore")
    db_to_cdisc_df.to_csv(OUT_DB_TO_CDISC, index=False)

    # ============================================================
    # 2) CDISC -> DB (group CDISC rows into record keys, then compare)
    # ============================================================
    cdisc_to_db_rows = []
    for key, cdisc_row_count in tqdm(cdisc_counts.items(), total=len(cdisc_counts), desc="CDISC -> DB", unit="key"):
        usubjid, visit, qscat_full = key

        if key in db_key_set:
            status = "AVAILABLE_IN_DB"
            db_row_count = db_counts.get(key, 0)
            comment = "FOUND_IN_DB"
        else:
            status = "MISSING_IN_DB"
            db_row_count = 0
            comment = "NO_MATCHING_DB_RECORD_KEY"

        cdisc_to_db_rows.append({
            "USUBJID": usubjid,
            "VISIT": visit,
            "QSCAT_FULL": qscat_full,
            "RECON_STATUS": status,
            "CDISC_ROW_COUNT_FOR_KEY": cdisc_row_count,  # ✅ your requested count
            "DB_ROW_COUNT_FOR_KEY": db_row_count,
            "COMMENT": comment,
        })

    cdisc_to_db_df = pd.DataFrame(cdisc_to_db_rows)
    cdisc_to_db_df.to_csv(OUT_CDISC_TO_DB, index=False)

    # ------------------ Summary ------------------
    total_db = len(db_to_cdisc_df)
    missing_in_cdisc = (db_to_cdisc_df["RECON_STATUS"] == "MISSING_IN_CDISC").sum()

    total_cdisc_keys = len(cdisc_to_db_df)
    missing_in_db = (cdisc_to_db_df["RECON_STATUS"] == "MISSING_IN_DB").sum()

    print("\n===== SUMMARY =====")
    print(f"DB rows processed:               {total_db}")
    print(f"DB records missing in CDISC:     {missing_in_cdisc}")
    print(f"CDISC record-keys processed:     {total_cdisc_keys}")
    print(f"CDISC records missing in DB:     {missing_in_db}")

    if missing_map:
        print(f"\n⚠️ Missing CDISC QSCAT mappings (logged): {sorted(missing_map)}")

    print(f"\n✅ Output 1 (DB → CDISC): {OUT_DB_TO_CDISC}")
    print(f"✅ Output 2 (CDISC → DB): {OUT_CDISC_TO_DB}")
    print(f"📝 Log:                  {LOG_FILE}")

    logging.info(
        f"Completed | db_rows={total_db} missing_in_cdisc={missing_in_cdisc} "
        f"| cdisc_keys={total_cdisc_keys} missing_in_db={missing_in_db}"
    )

if __name__ == "__main__":
    main()
