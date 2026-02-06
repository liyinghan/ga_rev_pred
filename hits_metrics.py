#!/usr/bin/env python3
"""
Compute per-row metrics from the hits field (within hits of that row):
  - total_product_viewed
  - average_price_product_viewed
  - total_category_viewed
  - total_revenue (from hits when present; optionally from totals)

Hits are Python literal (ast.literal_eval). Totals are JSON (json.loads).

Usage:
  from hits_metrics import compute_hits_metrics
  row_metrics = compute_hits_metrics(hits_raw, totals_raw=None)
  # row_metrics = {"total_product_viewed": int, "average_price_product_viewed": float, ...}
"""

import ast
import json
from typing import Any, Dict, List, Optional

# GA stores revenue and price in micros (1e6 = 1 USD)
MICROS = 1e6

REVENUE_KEYS = ("transactionRevenue", "totalTransactionRevenue", "productRevenue")


def _safe_float(val: Any) -> float:
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return 0.0


def _sum_revenue_in_obj(obj: Any, acc: List[float]) -> None:
    """Recursively sum numeric values for REVENUE_KEYS (in micros)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in REVENUE_KEYS:
                acc.append(_safe_float(v))
            _sum_revenue_in_obj(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _sum_revenue_in_obj(v, acc)


def _is_purchase_hit(hit: Any) -> bool:
    """True if hit's eCommerceAction.action_type is '6' (Purchase)."""
    if not isinstance(hit, dict):
        return False
    ea = hit.get("eCommerceAction")
    if not isinstance(ea, dict):
        return False
    return ea.get("action_type") == "6"


def _last_leaf_category(v2_product_category: Optional[str]) -> str:
    """
    Last-leaf logic: strip trailing slashes, split by /, take last element.
    Fallback to 'unknown' if empty or (not set).
    """
    raw = (v2_product_category or "").strip().rstrip("/")
    parts = [s for s in raw.split("/") if s]
    leaf = parts[-1] if parts else ""
    if not leaf or leaf == "(not set)":
        return "unknown"
    return leaf


def compute_hits_metrics(
    hits_raw: Optional[str],
    totals_raw: Optional[str] = None,
    use_totals_for_revenue_if_empty: bool = True,
) -> Dict[str, Any]:
    """
    Compute four metrics from the hits (and optionally totals) of one row.

    Returns:
      total_product_viewed: int
      average_price_product_viewed: float (USD); 0 if no products
      total_category_viewed: int (unique categories)
      categories_viewed: list[str] (each unique category last-leaf name, sorted)
      total_revenue: float (USD)
    """
    total_product_viewed = 0
    prices_usd: List[float] = []
    categories: set = set()
    revenue_micros: List[float] = []

    if hits_raw is None or (isinstance(hits_raw, str) and not hits_raw.strip()):
        hits = []
    else:
        try:
            hits = ast.literal_eval(hits_raw)
        except (ValueError, SyntaxError):
            hits = []
        if not isinstance(hits, list):
            hits = []

    for hit in hits:
        if not isinstance(hit, dict):
            continue
        products = hit.get("product") or []
        if not isinstance(products, list):
            products = []

        total_product_viewed += len(products)

        for p in products:
            if not isinstance(p, dict):
                continue
            # Price: productPrice or localProductPrice, in micros
            raw_price = p.get("productPrice") or p.get("localProductPrice")
            if raw_price is not None and str(raw_price).strip():
                price_usd = _safe_float(raw_price) / MICROS
                if price_usd >= 0:
                    prices_usd.append(price_usd)
            # Category: last-leaf logic
            cat = _last_leaf_category(p.get("v2ProductCategory"))
            categories.add(cat)
            # Revenue: only from purchase hits (handled below per-hit)
            pass

        # Revenue: only count when hit is a Purchase (action_type == '6')
        if _is_purchase_hit(hit):
            _sum_revenue_in_obj(hit, revenue_micros)

    # Average price
    if prices_usd:
        average_price_product_viewed = sum(prices_usd) / len(prices_usd)
    else:
        average_price_product_viewed = 0.0

    # Total revenue from hits (micros -> USD)
    total_revenue_from_hits = sum(revenue_micros) / MICROS

    if use_totals_for_revenue_if_empty and total_revenue_from_hits == 0 and totals_raw:
        try:
            totals = json.loads(totals_raw) if isinstance(totals_raw, str) else totals_raw
            if isinstance(totals, dict):
                rev = _safe_float(totals.get("transactionRevenue") or totals.get("totalTransactionRevenue"))
                if rev > 0:
                    total_revenue_from_hits = rev / MICROS
        except (json.JSONDecodeError, TypeError):
            pass

    return {
        "total_product_viewed": total_product_viewed,
        "average_price_product_viewed": average_price_product_viewed,
        "total_category_viewed": len(categories),
        # Add this line to output the unique category names:
        "categories_viewed": sorted(list(categories)),
        "total_revenue": total_revenue_from_hits,
    }


if __name__ == "__main__":
    sample_hits = """[{'hitNumber': '1', 'product': [
        {'productPrice': '23990000', 'v2ProductCategory': 'Home/Drinkware/'},
        {'productPrice': '24990000', 'v2ProductCategory': 'Home/Drinkware/'}
    ]}]"""
    m = compute_hits_metrics(sample_hits, totals_raw=None)
    print("Sample metrics:", m)
    assert m["total_product_viewed"] == 2
    assert m["total_category_viewed"] == 1  # last-leaf: "Drinkware" once
    assert m["categories_viewed"] == ["Drinkware"]
    assert 23.0 < m["average_price_product_viewed"] < 25.0

    # Last-leaf: Home/Apparel/Mens/ -> Mens
    assert _last_leaf_category("Home/Apparel/Mens/") == "Mens"
    assert _last_leaf_category("") == "unknown"
    assert _last_leaf_category("(not set)") == "unknown"
