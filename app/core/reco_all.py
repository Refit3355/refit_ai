import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import faiss

from app.core.db import safe_select, read_sql_df, qualify
from app.core.embedding import encode_queries, encode_passages

# ===== 파라미터 =====
ALPHA  = 0.5   # 임베딩 유사도
BETA   = 0.7   # 고민→효능 매칭
DELTA  = 0.2   # 인기도(log1p(UNITS_SOLD))
LAMBDA = 0.5   # 날씨 보정
TOPK   = 200
FINAL  = 10

# ===== 데이터 적재(전체) =====
def load_frames_all() -> dict:
    df_category = safe_select("CATEGORY", ["CATEGORY_ID","CATEGORY_NAME","DELETED_AT"])
    df_effect   = safe_select("EFFECT",   ["EFFECT_ID","EFFECT_NAME","BH_TYPE","DELETED_AT"])
    df_product  = safe_select("PRODUCT",  ["PRODUCT_ID","PRODUCT_NAME","CATEGORY_ID","PRICE","BRAND_NAME","DELETED_AT"])
    df_product_effect = safe_select("PRODUCT_EFFECT", ["PRODUCT_ID","EFFECT_ID"])

    if {"CATEGORY_ID","CATEGORY_NAME"}.issubset(df_category.columns):
        df_product = df_product.merge(
            df_category[["CATEGORY_ID","CATEGORY_NAME"]],
            on="CATEGORY_ID", how="left"
        )

    # 판매량(인기도)
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

    # 회원 상태/고민
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

# ===== 유틸/룰 =====
def _nz(x, default):
    try:
        if x is None: return default
        v = float(x)
        if np.isfinite(v): return v
    except Exception:
        pass
    return default

SKIN_TYPE_MAP = {1:"건성", 2:"지성", 3:"중성", 4:"복합성"}
SKIN_FLAG_2_LABEL = {
    "ATOPIC":"아토피","ACNE":"여드름","WHITENING":"미백","SEBUM":"피지",
    "INNER_DRYNESS":"속건조","WRINKLES":"주름","ENLARGED_PORES":"모공","REDNESS":"홍조","KERATIN":"각질"
}
HAIR_FLAG_2_LABEL = {
    "HAIR_LOSS":"탈모","DAMAGED_HAIR":"손상모","SCALP_TROUBLE":"두피트러블","DANDRUFF":"비듬"
}
HEALTH_FLAG_2_LABEL = {
    "EYE_HEALTH":"눈 건강","FATIGUE":"활력","SLEEP_STRESS":"수면/스트레스","IMMUNE_CARE":"면역력 증진",
    "MUSCLE_HEALTH":"근육/활력","GUT_HEALTH":"장 건강","BLOOD_CIRCULATION":"혈행 개선"
}

def get_effect_vocab(df_effect: pd.DataFrame) -> set:
    if df_effect.empty or "EFFECT_NAME" not in df_effect.columns:
        return set()
    return set(df_effect["EFFECT_NAME"].astype(str).str.strip())

EFFECT_ALIAS: Dict[str, str] = {}
def normalize_effect_names(names: List[str], vocab: set) -> List[str]:
    out = []
    for n in names:
        n = EFFECT_ALIAS.get(str(n).strip(), str(n).strip())
        if n in vocab:
            out.append(n)
    return sorted(set(out))

_RAW_CONCERN_TO_EFFECTS = {
    "여드름":["진정","여드름 완화"],
    "미백":["미백"],
    "피지":["진정"],
    "속건조":["보습"],
    "주름":["주름 개선","보습"],
    "모공":["진정"],
    "홍조":["진정"],
    "각질":[],
    "아토피":["진정","보습"],
    "탈모":["탈모 개선","두피 개선"],
    "손상모":["손상모 개선"],
    "두피트러블":["두피 개선"],
    "비듬":["두피 개선"],
    "눈 건강":["눈 건강"],
    "활력":["활력"],
    "면역력 증진":["면역력 증진"],
    "장 건강":["장 건강"],
    "혈행 개선":["혈행 개선"],
}

def build_query_text(member_id: int, frames: dict, prefer_category_name: str = None) -> dict:
    df_hi, df_skin, df_hair, df_hc = frames["df_hi"], frames["df_skin"], frames["df_hair"], frames["df_hc"]

    skin_type_txt = None
    if not df_hi.empty and {"MEMBER_ID","SKIN_TYPE"}.issubset(df_hi.columns):
        row = df_hi[df_hi["MEMBER_ID"] == member_id]
        if not row.empty:
            try:
                skin_type_txt = SKIN_TYPE_MAP.get(int(row.iloc[0]["SKIN_TYPE"]))
            except Exception:
                skin_type_txt = None

    def pick(df_one: pd.DataFrame, mapping: Dict[str,str]) -> List[str]:
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

    skin_row = df_skin[df_skin["MEMBER_ID"]==member_id] if not df_skin.empty else pd.DataFrame()
    hair_row = df_hair[df_hair["MEMBER_ID"]==member_id] if not df_hair.empty else pd.DataFrame()
    hc_row   = df_hc[df_hc["MEMBER_ID"]==member_id]     if not df_hc.empty   else pd.DataFrame()

    skin_list   = pick(skin_row, SKIN_FLAG_2_LABEL)
    hair_list   = pick(hair_row, HAIR_FLAG_2_LABEL)
    health_list = pick(hc_row,   HEALTH_FLAG_2_LABEL)

    vocab = get_effect_vocab(frames["df_effect"])
    concern_to_effects = {k: normalize_effect_names(v, vocab) for k, v in _RAW_CONCERN_TO_EFFECTS.items()}
    target_effects = sorted(set(sum([concern_to_effects.get(x, []) for x in (skin_list+hair_list+health_list)], [])))

    parts=[]
    if skin_type_txt: parts.append(f"피부타입={skin_type_txt}")
    if skin_list:     parts.append(f"피부고민={','.join(skin_list)}")
    if hair_list:     parts.append(f"두피/모발고민={','.join(hair_list)}")
    if health_list:   parts.append(f"건강고민={','.join(health_list)}")
    if prefer_category_name: parts.append(f"카테고리={prefer_category_name}")

    return {"query_text": " | ".join(parts) if parts else "고민 없음",
            "target_effects": target_effects,
            "skin_type": skin_type_txt}

_RAW_WEATHER_RULES = [
    {"cond": lambda w: _nz(w.get("uvi"), 0.0) >= 7.0,
     "effects": ["진정","미백","자외선 차단","항산화"], "bonus": 1.0},
    {"cond": lambda w: _nz(w.get("humidity"), 100.0) <= 40.0,
     "effects": ["보습"], "bonus": 0.8},
    {"cond": lambda w: (_nz(w.get("temp"), 0.0) >= 28.0) and (_nz(w.get("humidity"), 0.0) >= 60.0),
     "effects": ["진정"], "bonus": 0.6},
    {"cond": lambda w: _nz(w.get("pm25"), 0.0) >= 35.0,
     "effects": ["항산화","두피 개선"], "bonus": 0.7},
]

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
          .apply(lambda s: sorted(set(list(s.dropna().astype(str)))))
          .to_dict()
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

# ===== 메인(전체) =====
def recommend_all(member_id: int,
                  frames: dict,
                  prefer_category_id: Optional[int] = None,
                  topk: int = TOPK,
                  final: int = FINAL,
                  weather_ctx: Optional[dict] = None) -> pd.DataFrame:
    dp = frames["df_product"]
    dc = frames["df_category"]

    cat_name = None
    if prefer_category_id is not None and not dc.empty and {"CATEGORY_ID","CATEGORY_NAME"}.issubset(dc.columns):
        row = dc[dc["CATEGORY_ID"] == prefer_category_id]
        if not row.empty:
            cat_name = row.iloc[0]["CATEGORY_NAME"]

    q = build_query_text(member_id, frames, prefer_category_name=cat_name)

    prod_effects = build_prod_effects(frames)
    dp = attach_product_text(dp, prod_effects)

    if prefer_category_id is not None and "CATEGORY_ID" in dp.columns:
        dp = dp[dp["CATEGORY_ID"] == int(prefer_category_id)]

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
        # 날씨 보정 
        def weather_bonus_for_product(product_id: int) -> float:
            effs = set(prod_effects.get(int(product_id), []))
            if not effs or not wctx: return 0.0
            bonus = 0.0
            for r in weather_rules:
                if r["cond"](wctx):
                    hit = effs.intersection(set(r["effects"]))
                    if hit:
                        ratio = len(hit) / max(1, len(r["effects"]))
                        bonus += r["bonus"] * ratio
            return float(bonus)
        w_bonus = weather_bonus_for_product(pid)

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
            "ordersSold": float(prow["ORDERS_SOLD"]) if "ORDERS_SOLD" in dp.columns else 0.0
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res = res[np.isfinite(res["sim"]) & np.isfinite(res["finalScore"])]
    res = res.drop_duplicates(["productId"], keep="first")
    res = res.sort_values(["finalScore","sim"], ascending=False).head(final).reset_index(drop=True)
    return res
