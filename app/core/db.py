import os
from functools import lru_cache
from typing import Optional
import pandas as pd

SKIP_DB = os.getenv("SKIP_DB", "0") == "1"

if not SKIP_DB:
    import oracledb

# env
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_DSN = os.getenv("DB_DSN", "")
DB_TNS_ADMIN = os.getenv("DB_TNS_ADMIN", "")
DB_WALLET_PASSWORD = os.getenv("DB_WALLET_PASSWORD", None)
DB_SCHEMA = os.getenv("DB_SCHEMA", None)

def get_conn():
    if SKIP_DB:
        raise RuntimeError("SKIP_DB=1 인 경우 DB 연결이 비활성화되어 있습니다.")
    return oracledb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=DB_DSN,
        config_dir=DB_TNS_ADMIN,
        wallet_location=DB_TNS_ADMIN,
        wallet_password=DB_WALLET_PASSWORD
    )

def read_sql_df(query: str, params: dict | None = None) -> pd.DataFrame:
    if SKIP_DB:
        return pd.DataFrame()
    with get_conn() as conn:
        df = pd.read_sql(query, conn, params=params)
    df.columns = df.columns.str.upper()
    return df

@lru_cache(maxsize=256)
def qualify(table_name: str) -> str:
    t = table_name.upper()
    # 내 스키마
    df_user = read_sql_df("""
        SELECT 'TABLE' AS OBJ_TYPE, TABLE_NAME AS OBJ_NAME FROM USER_TABLES WHERE TABLE_NAME = :t
        UNION ALL
        SELECT 'VIEW'  AS OBJ_TYPE, VIEW_NAME  AS OBJ_NAME FROM USER_VIEWS  WHERE VIEW_NAME  = :t
    """, {"t": t})
    if not df_user.empty:
        return t

    df_all = read_sql_df("""
        SELECT OWNER, TABLE_NAME AS OBJ_NAME, 'TABLE' AS OBJ_TYPE FROM ALL_TABLES WHERE TABLE_NAME = :t
        UNION ALL
        SELECT OWNER, VIEW_NAME  AS OBJ_NAME, 'VIEW'  AS OBJ_TYPE FROM ALL_VIEWS  WHERE VIEW_NAME  = :t
        UNION ALL
        SELECT OWNER, SYNONYM_NAME AS OBJ_NAME, 'SYNONYM' AS OBJ_TYPE FROM ALL_SYNONYMS WHERE SYNONYM_NAME = :t
    """, {"t": t})
    if df_all.empty:
        raise RuntimeError(f"객체 '{t}'를 찾을 수 없습니다. (권한/이름 확인)")

    owner = None
    if DB_SCHEMA:
        pick = df_all[df_all["OWNER"].str.upper() == DB_SCHEMA.upper()]
        if not pick.empty:
            owner = pick.iloc[0]["OWNER"]
    if owner is None:
        pick = df_all[df_all["OWNER"].str.upper().str.contains("REFIT")]
        if not pick.empty:
            owner = pick.iloc[0]["OWNER"]
    if owner is None:
        owner = df_all.iloc[0]["OWNER"]
    return f"{owner}.{t}"

def safe_select(table: str, preferred_cols: list[str]) -> pd.DataFrame:
    if SKIP_DB:
        # dry-run 모드일 때는 컬럼만 있는 빈 DF 반환
        return pd.DataFrame(columns=[c.upper() for c in preferred_cols])
    qname = qualify(table)
    if "." in qname:
        owner = qname.split(".")[0]
    else:
        owner = read_sql_df("SELECT USER AS U FROM DUAL").iloc[0]["U"]
    df_cols = read_sql_df("""
        SELECT COLUMN_NAME FROM ALL_TAB_COLUMNS
        WHERE TABLE_NAME = :t AND OWNER = :o
    """, {"t": table.upper(), "o": owner.upper()})
    cols = set(df_cols["COLUMN_NAME"].astype(str).str.upper())
    pick = [c for c in preferred_cols if c in cols] or preferred_cols  # 최소한 요청 컬럼 유지
    df = read_sql_df(f"SELECT {', '.join(pick)} FROM {qname}")
    if "DELETED_AT" in df.columns:
        df = df[df["DELETED_AT"].isna()].drop(columns=["DELETED_AT"])
    return df
