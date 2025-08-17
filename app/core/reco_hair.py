import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import faiss

from app.core.db import safe_select, read_sql_df, qualify
from app.core.embedding import encode_queries, encode_passages

# --------------------------------
# 파라미터
# --------------------------------
ALPHA  = 0.5   # 임베딩 유사도
BETA   = 0.7   # 고민→효능 매칭 (헤어 전용)
DELTA  = 0.2   # 인기도(log1p(UNITS_SOLD))
LAMBDA = 0.5   # 날씨 보정
TOPK   = 200
FINAL  = 10

# --------------------------------
# 데이터 적재(헤어 범위: CATEGORY_ID 6~7)
# --------------------------------
def load_frames_hair() -> dict:
    df_category = safe_select("CATEGORY", ["CATEGORY_ID","CATEGORY_NAME","DELETED_AT"])
    df_effect   = safe_select("EFFECT",   ["EFFECT_ID","EFFECT_NAME","BH_TYPE","DELETED_AT"])
    df_product  = safe_select("PRODUCT",  ["PRODUCT_ID","PRODUCT_NAME","CATEGORY_ID","PRICE","BRAND_NAME","STOCK","DISCOUNT_RATE","THUMBNAIL_URL","DELETED_AT"])
    df_product_effect = safe_select("PRODUCT_EFFECT", ["PRODUCT_ID","EFFECT_ID"])

    if {"CATEGORY_ID","CATEGORY_NAME"}.issubset(df_category.columns):
        df_product = df_product.merge(
            df_category[["CATEGORY_ID","CATEGORY_NAME"]],
            on="CATEGORY_ID", how="left"
        )

    df_product = df_product[df_product["CATEGORY_ID"].between(6, 7)].reset_index(drop=True)

    # 판매량 집계
    t_oi = qualify("ORDER_ITEM")
    t_o  = qualify("ORDERS")
    q_pop = f"""
    SELECT
        oi.PRODUCT_ID,
        SUM(NVL(oi.ITEM_COUNT, 1)) AS UNITS_SOLD,
        COUNT(DISTINCT oi.ORDER_ID) AS ORDERS_SOLD
    FROM {t_oi} oi
    JOIN {t_o}  o  ON o.ORDER_ID = oi.ORDER_ID
    WHERE (oi.DELETED_AT IS NULL) AND (o.DELETED_AT IS NULL)
    GROUP BY oi.PRODUCT_ID
    """
    df_pop = read_sql_df(q_pop)
    for c in ["UNITS_SOLD","ORDERS_SOLD"]:
        if c in df_pop.columns:
            df_pop[c] = pd.to_numeric(df_pop[c], errors="coerce").fillna(0)

    df_product = df_product.merge(df_pop, on="PRODUCT_ID", how="left")
    for c in ["UNITS_SOLD","ORDERS_SOLD"]:
        if c in df_product.columns:
            df_product[c] = pd.to_numeric(df_product[c], errors="coerce").fillna(0.0)

    # 회원 고민(헤어 중심)
    df_hi   = safe_select("HEALTH_INFO", ["MEMBER_ID","SKIN_TYPE"])
    df_hc   = read_sql_df(f"SELECT * FROM {qualify('HEALTH_CONCERN')}")
    df_hair = read_sql_df(f"SELECT * FROM {qualify('HAIR_CONCERN')}")
    df_skin = read_sql_df(f"SELECT * FROM {qualify('SKIN_CONCERN')}")

    return {
        "df_category": df_category,
        "df_effect": df_effect,
        "df_product": df_product,
        "df_product_effect": df_product_effect,
        "df_hi": df_hi, "df_hc": df_hc, "df_hair": df_hair, "df_skin": df_skin
    }

# --------------------------------
# 유틸/룰
# --------------------------------
def _nz(x, default):
    try:
        if x is None: return default
        v = float(x)
        if np.isfinite(v): return v
    except Exception:
        pass
    return default

HAIR_FLAG_2_LABEL = {
    "HAIR_LOSS":"탈모","DAMAGED_HAIR":"손상모","SCALP_TROUBLE":"두피트러블","DANDRUFF":"비듬"
}

def get_effect_vocab(df_effect: pd.DataFrame) -> set:
    if df_effect.empty or "EFFECT_NAME" not in df_effect.columns:
        return set()
    return set(df_effect["EFFECT_NAME"].astype(str).str.strip())

EFFECT_ALIAS: Dict[str, str] = {
    # 필요 시 별칭: "두피 클렌징": "두피 개선"
}

def normalize_effect_names(names: List[str], vocab: set) -> List[str]:
    out = []
    for n in names:
        n = EFFECT_ALIAS.get(str(n).strip(), str(n).strip())
        if n in vocab:
            out.append(n)
    return sorted(set(out))

# 헤어 전용 고민→효능
_RAW_HAIR_CONCERN_TO_EFFECTS: Dict[str, List[str]] = {
    "탈모":       ["탈모 개선","두피 개선"],
    "손상모":     ["손상모 개선"],
    "두피트러블": ["두피 개선"],
    "비듬":       ["두피 개선"],
}

# 날씨 룰(헤어)
_RAW_WEATHER_RULES = [
    {"cond": lambda w: _nz(w.get("humidity"), 100.0) <= 40.0,
     "effects": ["손상모 개선"], "bonus": 0.8},
    {"cond": lambda w: (_nz(w.get("temp"), 0.0) >= 28.0) and (_nz(w.get("humidity"), 0.0) >= 60.0),
     "effects": ["두피 개선"], "bonus": 0.6},
    {"cond": lambda w: _nz(w.get("pm25"), 0.0) >= 35.0,
     "effects": ["두피 개선"], "bonus": 0.7},
]

def build_query_text(member_id: int, frames: dict, prefer_category_name: str = None) -> dict:
    df_hair = frames["df_hair"]

    def pick_labels(df_one: pd.DataFrame, mapping: Dict[str,str]) -> List[str]:
        if df_one.empty: return []
        out=[]
        for k,label in mapping.items():
            if k in df_one.columns:
                try:
                    if int(df_one.iloc[0][k]) == 1:
                        out.append(label)
                except Exception:
                    pass
        return out

    hair_row = df_hair[df_hair["MEMBER_ID"]==member_id] if not df_hair.empty else pd.DataFrame()
    hair_list = pick_labels(hair_row, HAIR_FLAG_2_LABEL)

    vocab = get_effect_vocab(frames["df_effect"])
    concern_to_effects = {k: normalize_effect_names(v, vocab) for k, v in _RAW_HAIR_CONCERN_TO_EFFECTS.items()}
    target_effects = sorted(set(sum([concern_to_effects.get(x, []) for x in hair_list], [])))

    parts=[]
    if hair_list: parts.append(f"두피/모발고민={','.join(hair_list)}")
    if prefer_category_name: parts.append(f"카테고리={prefer_category_name}")

    return {
        "query_text": " | ".join(parts) if parts else "헤어 고민 없음",
        "target_effects": target_effects,
        "skin_type": None
    }

def normalize_rule_effects(rules: List[dict], vocab: set) -> List[dict]:
    out=[]
    for r in rules:
        effs = normalize_effect_names(r["effects"], vocab)
        out.append({**r, "effects": effs})
    return out

def build_prod_effects(frames: dict) -> Dict[int, List[str]]:
    eff = frames["df_effect"]
    pe  = frames["df_product_effect"]
    if eff.empty or pe.empty:
        return {}
    merged = (
        pe.merge(eff[["EFFECT_ID","EFFECT_NAME"]], on="EFFECT_ID", how="left")
          .groupby("PRODUCT_ID")["EFFECT_NAME"]
          .apply(lambda s: sorted(set(list(s.dropna().astype(str))))).to_dict()
    )
    return merged

def attach_product_text(dp: pd.DataFrame, prod_effects: Dict[int, List[str]]) -> pd.DataFrame:
    if dp.empty:
        raise ValueError("[df_product] is empty")
    for c in ["PRODUCT_ID","PRODUCT_NAME","CATEGORY_NAME"]:
        if c not in dp.columns:
            raise ValueError(f"[df_product] missing column: {c}")
    def build_text(row: pd.Series) -> str:
        effs = prod_effects.get(int(row["PRODUCT_ID"]), [])
        return f"이름:{row['PRODUCT_NAME']} | 카테고리:{row['CATEGORY_NAME']} | 효능:{','.join(effs)}"
    dp = dp.copy()
    dp["PRODUCT_TEXT"] = dp.apply(build_text, axis=1)
    return dp

def build_faiss_index(dp: pd.DataFrame):
    texts = dp["PRODUCT_TEXT"].tolist()
    vecs  = encode_passages(texts)
    ids   = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index, ids

def effect_match_score(product_id: int, target_effects: List[str], prod_effects: Dict[int, List[str]]) -> float:
    if not target_effects: return 0.0
    effs = set(prod_effects.get(int(product_id), []))
    return float(sum(1 for t in target_effects if t in effs))

def weather_bonus_for_product(product_id: int, weather: dict, prod_effects: Dict[int, List[str]], rules: List[dict]) -> float:
    effs = set(prod_effects.get(int(product_id), []))
    if not effs or not weather: return 0.0
    bonus = 0.0
    for r in rules:
        if r["cond"](weather):
            hit = effs.intersection(set(r["effects"]))
            if hit:
                ratio = len(hit) / max(1, len(r["effects"]))
                bonus += r["bonus"] * ratio
    return float(bonus)

# --------------------------------
# 메인
# --------------------------------
def recommend_hair(member_id: int,
                   frames: dict,
                   prefer_category_id: Optional[int] = None,
                   topk: int = TOPK,
                   final: int = FINAL,
                   weather_ctx: Optional[dict] = None) -> pd.DataFrame:
    dp = frames["df_product"]
    dc = frames["df_category"]

    # 카테고리명(선택)
    cat_name = None
    if prefer_category_id is not None and not dc.empty and {"CATEGORY_ID","CATEGORY_NAME"}.issubset(dc.columns):
        row = dc[dc["CATEGORY_ID"] == prefer_category_id]
        if not row.empty:
            cat_name = row.iloc[0]["CATEGORY_NAME"]

    q = build_query_text(member_id, frames, prefer_category_name=cat_name)

    prod_effects = build_prod_effects(frames)
    dp = attach_product_text(dp, prod_effects)
    dp = dp[dp["CATEGORY_ID"].between(6, 7)].reset_index(drop=True)
    if "STOCK" in dp.columns:
        dp = dp[pd.to_numeric(dp["STOCK"], errors="coerce").fillna(0) > 0]

    index, ids = build_faiss_index(dp)
    k = min(int(topk), len(ids)) if len(ids) > 0 else 0
    if k == 0:
        return pd.DataFrame(columns=["productId","name","category","sim","effMatch","finalScore"])

    qv = encode_queries([q["query_text"]])
    D, I = index.search(qv, k)

    vocab = get_effect_vocab(frames["df_effect"])
    weather_rules = normalize_rule_effects(_RAW_WEATHER_RULES, vocab)
    has_units = "UNITS_SOLD" in dp.columns
    wctx = weather_ctx or {}

    rows = []
    for idx, sim in zip(I[0], D[0]):
        if idx < 0 or not np.isfinite(sim): 
            continue
        pid = int(ids[idx])
        prow = dp[dp["PRODUCT_ID"]==pid].iloc[0]

        eff_score = effect_match_score(pid, q["target_effects"], prod_effects)
        pop_bonus = math.log1p(float(prow["UNITS_SOLD"])) if has_units and pd.notna(prow.get("UNITS_SOLD", None)) else 0.0
        w_bonus   = weather_bonus_for_product(pid, wctx, prod_effects, weather_rules)

        final_score = ALPHA*float(sim) + BETA*eff_score + LAMBDA*w_bonus + DELTA*pop_bonus

        rows.append({
            "productId": pid,
            "name": prow.get("PRODUCT_NAME"),
            "category": prow.get("CATEGORY_NAME"),
            "categoryId": int(prow.get("CATEGORY_ID")) if pd.notna(prow.get("CATEGORY_ID")) else None,
            "sim": float(sim),
            "effMatch": float(eff_score),
            "finalScore": float(final_score),
            "price": int(prow["PRICE"]) if "PRICE" in dp.columns and pd.notna(prow.get("PRICE")) else None,
            "brand": prow.get("BRAND_NAME"),
            "unitsSold": float(prow["UNITS_SOLD"]) if has_units else 0.0,
            "ordersSold": float(prow["ORDERS_SOLD"]) if "ORDERS_SOLD" in dp.columns else 0.0,
            "stock": int(prow["STOCK"]) if "STOCK" in dp.columns and pd.notna(prow.get("STOCK")) else None,
            "discountRate": int(prow["DISCOUNT_RATE"]) if "DISCOUNT_RATE" in dp.columns and pd.notna(prow.get("DISCOUNT_RATE")) else None,
            "thumbnailUrl": prow.get("THUMBNAIL_URL") if "THUMBNAIL_URL" in dp.columns else None
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res[np.isfinite(res["sim"]) & np.isfinite(res["finalScore"])]
    res = res.drop_duplicates(["productId"], keep="first")
    res = res.sort_values(["finalScore","sim"], ascending=False).head(final).reset_index(drop=True)
    return res
