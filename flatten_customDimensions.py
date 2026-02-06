#!/usr/bin/env python3
"""
Flatten customDimensions: extract only the 'value' (e.g. "North America", "apac").

How it works:
1. Each cell is a string that looks like Python literal: "[{'index': '4', 'value': 'North America'}]"
2. Parse with ast.literal_eval() → list of dicts
3. Take the 'value' from each dict → list of strings
4. Join multiple values (if any) with ", " or take the first one; use "" for empty []

Usage:
  from flatten_customDimensions import flatten_custom_dimensions
  df["customDimensions_flat"] = df["customDimensions"].map(flatten_custom_dimensions)
"""

import ast


def flatten_custom_dimensions(raw: str, join_multiple: str = ", ") -> str:
    """
    Parse customDimensions string and return only the value(s).

    - "[{'index': '4', 'value': 'North America'}]"  →  "North America"
    - "[]" or null/empty                           →  ""
    - Multiple dicts: values are joined with join_multiple (default ", ").
    """
    if raw is None or (isinstance(raw, str) and raw.strip() in ("", "[]")):
        return ""

    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return ""

    if not isinstance(parsed, list):
        return ""

    values = []
    for d in parsed:
        if isinstance(d, dict) and "value" in d:
            values.append(str(d["value"]).strip())

    return join_multiple.join(values) if values else ""


# --- Example / test ---
if __name__ == "__main__":
    examples = [
        "[{'index': '4', 'value': 'North America'}]",
        "[{'index': '4', 'value': 'EMEA'}]",
        "[{'index': '4', 'value': 'apac'}]",
        "[]",
        "",
    ]
    print("Flatten customDimensions (values only):\n")
    for ex in examples:
        out = flatten_custom_dimensions(ex)
        print(f"  {repr(ex)[:50]:50}  →  {repr(out)}")
