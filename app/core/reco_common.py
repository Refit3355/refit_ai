from typing import Optional, List
import numpy as np
import pandas as pd
import faiss

from app.core.db import safe_select
from app.core.embedding import encode_queries, encode_passages


# ---- 최소 프레임 로더 ----
def load_frames() -> dict:
    df_category = safe_select("CATEGORY", ["CATEGORY_ID", "CATEGORY_NAME", "DELETED_AT"])
    df_product  = safe_select("PRODUCT",  ["PRODUCT_ID", "PRODUCT_NAME", "CATEGORY_ID", "PRICE", "BRAND_NAME", "STOCK", "DISCOUNT_RATE", "THUMBNAIL_URL", "DELETED_AT"])

    if {"CATEGORY_ID", "CATEGORY_NAME"}.issubset(df_category.columns):
        df_product = df_product.merge(
            df_category[["CATEGORY_ID", "CATEGORY_NAME"]],
            on="CATEGORY_ID", how="left"
        )

    return {"df_product": df_product, "df_category": df_category}


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
    vecs  = encode_passages(texts)  # e5 passage 인코딩 (정규화 포함)
    ids   = dp["PRODUCT_ID"].to_numpy(dtype=np.int64)
    index = faiss.IndexFlatIP(vecs.shape[1])  # cosine(dot)용
    index.add(vecs)
    return index, ids


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

    # 카테고리 제한
    if prefer_category_id is not None and "CATEGORY_ID" in dp.columns:
        dp = dp[dp["CATEGORY_ID"] == int(prefer_category_id)]

    # 재고 필터
    if "STOCK" in dp.columns:
        dp = dp[pd.to_numeric(dp["STOCK"], errors="coerce").fillna(0) > 0]

    # 텍스트 생성 및 인덱스 빌드
    dp = attach_product_text(dp)
    index, ids = build_faiss_index(dp)

    # 쿼리 인코딩 & 검색
    qv = encode_queries([query_text])
    k = min(int(topk), len(ids))
    if k <= 0:
        return pd.DataFrame(columns=["productId", "name", "finalScore"])

    D, I = index.search(qv, k)

    rows: List[dict] = []
    for idx, sim in zip(I[0], D[0]):
        if idx < 0 or not np.isfinite(sim):
            continue
        row = dp.iloc[idx]
        rows.append({
            "productId": int(ids[idx]),
            "name": row.get("PRODUCT_NAME"),
            "categoryId": int(row.get("CATEGORY_ID")) if pd.notna(row.get("CATEGORY_ID")) else None,
            "sim": float(sim),
            "finalScore": float(sim),  # 유사도=최종점수
            "price": int(row["PRICE"]) if "PRICE" in dp.columns and pd.notna(row.get("PRICE")) else None,
            "brand": row.get("BRAND_NAME"),
            "stock": int(row["STOCK"]) if "STOCK" in dp.columns and pd.notna(row.get("STOCK")) else None,
            "discountRate": int(row["DISCOUNT_RATE"]) if "DISCOUNT_RATE" in dp.columns and pd.notna(row.get("DISCOUNT_RATE")) else None,
            "thumbnailUrl": row.get("THUMBNAIL_URL") if "THUMBNAIL_URL" in dp.columns else None
        })

    res = pd.DataFrame(rows)
    return res.sort_values("finalScore", ascending=False).head(final).reset_index(drop=True)
