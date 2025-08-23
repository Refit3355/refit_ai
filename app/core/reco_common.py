from typing import List
import numpy as np
import pandas as pd
import faiss

from app.core.db import safe_select
from app.core.embedding import encode_queries, encode_passages
from app.core.scheduler import global_index  

# ---- 최소 프레임 로더 ----
def load_frames() -> dict:
    df_category = safe_select("CATEGORY", ["CATEGORY_ID", "CATEGORY_NAME", "DELETED_AT"])
    df_product  = safe_select("PRODUCT",  ["PRODUCT_ID", "PRODUCT_NAME", "CATEGORY_ID", "PRICE", "BRAND_NAME", "STOCK", "DISCOUNT_RATE", "THUMBNAIL_URL", "DELETED_AT"])

    if {"CATEGORY_ID", "CATEGORY_NAME"}.issubset(df_category.columns):
        df_product = df_product.merge(
            df_category[["CATEGORY_ID", "CATEGORY_NAME"]],
            on="CATEGORY_ID", how="left"
        )

    df_hi   = safe_select("HEALTH_INFO", ["MEMBER_ID","STEPS","BLOOD_GLUCOSE","BLOOD_PRESSURE","TOTAL_CALORIES_BURNED","NUTRITION","SLEEPSESSION"])
    df_skin = safe_select("SKIN_CONCERN", ["MEMBER_ID","SKIN_TYPE"])

    return {"df_product": df_product, "df_category": df_category, "df_hi": df_hi, "df_skin": df_skin}


# ---- 상품 텍스트 생성 (임베딩 입력) ----
def attach_product_text(dp: pd.DataFrame) -> pd.DataFrame:
    if dp.empty:
        raise ValueError("[df_product] is empty")

    must = ["PRODUCT_ID", "PRODUCT_NAME", "CATEGORY_ID"]
    for c in must:
        if c not in dp.columns:
            raise ValueError(f"[df_product] missing column: {c}")

    def _build(row: pd.Series) -> str:
        cat = row.get("CATEGORY_NAME") if "CATEGORY_NAME" in dp.columns else str(row.get("CATEGORY_ID"))
        return f"이름:{row['PRODUCT_NAME']} | 카테고리:{cat}"

    out = dp.copy()
    out["PRODUCT_TEXT"] = out.apply(_build, axis=1)
    return out


# ---- FAISS 인덱스 ----
def build_faiss_index(dp: pd.DataFrame):
    texts = dp["PRODUCT_TEXT"].tolist()
    vecs  = encode_passages(texts)
    ids   = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)

    base = faiss.IndexFlatIP(vecs.shape[1])
    index = faiss.IndexIDMap(base)
    index.add_with_ids(vecs, ids)

    return index


# ---- 단순 공통 추천 (비상용/백업 로직) ----
def simple_recommend(
    query_text: str,
    frames: dict,
    topk=200,
    final=100,
    product_type: int = 0,
    prefer_category_id: int | None = None
) -> pd.DataFrame:
    dp = frames["df_product"]
    if dp.empty:
        return pd.DataFrame(columns=["productId", "name", "finalScore"])

    # 카테고리 선호 필터
    if prefer_category_id is not None and "CATEGORY_ID" in dp.columns:
        dp = dp[dp["CATEGORY_ID"] == int(prefer_category_id)]

    # 재고 필터
    if "STOCK" in dp.columns:
        dp = dp[pd.to_numeric(dp["STOCK"], errors="coerce").fillna(0) > 0]

    # 텍스트 부착
    dp = attach_product_text(dp)
    if global_index is None:
        index = build_faiss_index(dp)
    else:
        index = global_index

    import re
    def _nz(x, d):
        try:
            if x is None: return d
            v = float(x)
            if np.isfinite(v): return v
        except Exception:
            pass
        return d

    # member 추출
    member_id = None
    m = re.search(r"member:(\d+)", query_text)
    if m:
        try:
            member_id = int(m.group(1))
        except Exception:
            member_id = None

    # 건강 지표 기반 enrich
    enriched = query_text
    df_hi = frames.get("df_hi", pd.DataFrame())
    df_skin = frames.get("df_skin", pd.DataFrame())
    eff = []

    if member_id is not None and not df_skin.empty and {"MEMBER_ID","SKIN_TYPE"}.issubset(df_skin.columns):
        row = df_skin[df_skin["MEMBER_ID"] == member_id]
        if not row.empty:
            try:
                SKIN_TYPE_MAP = {1:"건성", 2:"중성", 3:"지성", 4:"복합성", 5:"수분 부족 지성"}
                skin_type_txt = SKIN_TYPE_MAP.get(int(row.iloc[0]["SKIN_TYPE"]))
                if skin_type_txt:
                    enriched = f"{enriched} | 피부타입={skin_type_txt}"
            except Exception:
                pass

    if member_id is not None and not df_hi.empty and "MEMBER_ID" in df_hi.columns:
        row = df_hi[df_hi["MEMBER_ID"] == member_id]
        if not row.empty:
            r = row.iloc[0]
            steps = _nz(r.get("STEPS"), np.nan)
            glu = _nz(r.get("BLOOD_GLUCOSE"), np.nan)
            bp = _nz(r.get("BLOOD_PRESSURE"), np.nan)
            kcal = _nz(r.get("TOTAL_CALORIES_BURNED"), np.nan)
            nutr = _nz(r.get("NUTRITION"), np.nan)
            sleep = _nz(r.get("SLEEPSESSION"), np.nan)

            if product_type == 1:  # 스킨케어
                if np.isfinite(nutr) and nutr <= 1: eff += ["보습"]
                if np.isfinite(sleep) and sleep < 420: eff += ["진정"]
                if np.isfinite(kcal) and kcal > 700: eff += ["진정"]
                eff = [e for e in eff if e in {"보습","진정","주름 개선","미백","자외선 차단","여드름 완화","가려움 개선","튼살 개선"}]
            elif product_type == 2:  # 헤어케어
                if np.isfinite(nutr) and nutr <= 1: eff += ["손상모 개선","탈모 개선"]
                if np.isfinite(steps) and steps < 5000: eff += ["탈모 개선"]
                if np.isfinite(sleep) and sleep < 420: eff += ["탈모 개선"]
                if np.isfinite(bp) and bp >= 130: eff += ["두피 개선"]
                if np.isfinite(glu) and glu >= 126: eff += ["두피 개선"]
                if np.isfinite(kcal) and kcal > 700: eff += ["두피 개선"]
                eff = [e for e in eff if e in {"손상모 개선","탈모 개선","두피 개선"}]
            elif product_type == 3:  # 건강기능식품
                if np.isfinite(sleep) and sleep < 420: eff += ["활력"]
                if np.isfinite(steps) and steps < 5000: eff += ["혈행 개선","활력"]
                if np.isfinite(glu) and glu >= 126: eff += ["장 건강"]
                if np.isfinite(bp) and bp >= 130: eff += ["혈행 개선"]
                if np.isfinite(nutr) and nutr <= 1: eff += ["면역력 증진"]
                if np.isfinite(kcal) and kcal > 700: eff += ["활력"]
                eff = [e for e in eff if e in {"혈행 개선","장 건강","면역력 증진","항산화","눈 건강","뼈 건강","활력","피부 건강"}]
            else:  # 혼합
                if np.isfinite(nutr) and nutr <= 1: eff += ["보습","면역력 증진"]
                if np.isfinite(sleep) and sleep < 420: eff += ["진정","활력"]
                if np.isfinite(steps) and steps < 5000: eff += ["탈모 개선","혈행 개선"]
                if np.isfinite(bp) and bp >= 130: eff += ["두피 개선","혈행 개선"]
                if np.isfinite(glu) and glu >= 126: eff += ["두피 개선","장 건강"]
                if np.isfinite(kcal) and kcal > 700: eff += ["진정","활력"]

            eff = sorted(set(eff))
            if eff:
                enriched = f"{enriched} | 건강지표효능={','.join(eff)}"

    # 검색
    qv = encode_queries([enriched])
    k = min(int(topk), len(dp))
    if k <= 0:
        return pd.DataFrame(columns=["productId", "name", "finalScore"])

    D, I = index.search(qv, k)

    # 결과 매핑
    rows: List[dict] = []
    for pid, sim in zip(I[0], D[0]):
        if pid < 0 or not np.isfinite(sim):
            continue
        row = dp[dp["PRODUCT_ID"] == pid].iloc[0]
        rows.append({
            "productId": int(row["PRODUCT_ID"]),
            "name": row.get("PRODUCT_NAME"),
            "categoryId": int(row.get("CATEGORY_ID")) if pd.notna(row.get("CATEGORY_ID")) else None,
            "sim": float(sim),
            "finalScore": float(sim),
            "price": int(row["PRICE"]) if "PRICE" in dp.columns and pd.notna(row.get("PRICE")) else None,
            "brand": row.get("BRAND_NAME"),
            "stock": int(row["STOCK"]) if "STOCK" in dp.columns and pd.notna(row.get("STOCK")) else None,
            "discountRate": int(row["DISCOUNT_RATE"]) if "DISCOUNT_RATE" in dp.columns and pd.notna(row.get("DISCOUNT_RATE")) else None,
            "thumbnailUrl": row.get("THUMBNAIL_URL") if "THUMBNAIL_URL" in dp.columns else None
        })

    res = pd.DataFrame(rows)
    return res.sort_values("finalScore", ascending=False).head(final).reset_index(drop=True)
