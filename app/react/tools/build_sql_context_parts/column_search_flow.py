from __future__ import annotations

import json
import re
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from xml.sax.saxutils import escape as xml_escape

from app.smart_logger import SmartLogger

from .models import ColumnCandidate, TableCandidate
from .neo4j import (
    _neo4j_fetch_anchor_like_columns_for_tables,
    _neo4j_fetch_table_schemas,
    _neo4j_search_table_scoped_columns,
)
from .text import _ANCHOR_COLUMN_SUBSTRINGS


@dataclass(frozen=True)
class ColumnSearchResult:
    selected_tables: List[TableCandidate]
    per_table_columns: Dict[str, List[ColumnCandidate]]
    per_table_mode: str
    column_candidates: List[ColumnCandidate]


_RE_TEXT = re.compile(r"(char|text|varchar)", re.I)
_RE_NUM = re.compile(r"(int|numeric|decimal|real|double|float)", re.I)
_RE_TIME = re.compile(r"(date|time|timestamp)", re.I)
_RE_NAME_CODE = re.compile(r"(name|nm|title|code|cd|id|sn)$|(^|_)(name|nm|title|code|cd|id|sn)($|_)", re.I)
_RE_NAME_ONLY = re.compile(r"(name|nm|title)$|(^|_)(name|nm|title)($|_)", re.I)
_RE_CODE_ONLY = re.compile(r"(code|cd|id|sn)$|(^|_)(code|cd|id|sn)($|_)", re.I)
_RE_METRIC_NAME = re.compile(
    r"(val|value|amt|amount|qty|quantity)$|(^|_)(val|value|amt|amount|qty|quantity)($|_)", re.I
)
_RE_TIME_NAME = re.compile(r"(dt|date|time|tm|ts)$|(^|_)(dt|date|time|tm|ts)($|_)", re.I)


def _is_name_like(col_name: str) -> bool:
    n = (col_name or "").strip()
    if not n:
        return False
    return _RE_NAME_ONLY.search(n) is not None


def _is_code_like(col_name: str) -> bool:
    n = (col_name or "").strip()
    if not n:
        return False
    return _RE_CODE_ONLY.search(n) is not None


def _name_hint_bonus(col_name_l: str) -> float:
    s = (col_name_l or "").strip().lower()
    if not s:
        return 0.0
    if s == "nm" or s.endswith("_nm") or s.endswith("nm") or "name" in s:
        return 0.15
    if "title" in s:
        return 0.05
    return 0.02


def _code_hint_bonus(col_name_l: str) -> float:
    s = (col_name_l or "").strip().lower()
    if not s:
        return 0.0
    if s.endswith("_cd") or s.endswith("_code") or "code" in s:
        return 0.15
    if s.endswith("_id") or s == "id" or "id" in s:
        return 0.05
    if s.endswith("sn") or s == "sn":
        return 0.03
    return 0.02


def _tfqn_l(schema: str, table: str) -> str:
    s = (schema or "").strip().lower()
    t = (table or "").strip().lower()
    return f"{s}.{t}" if s else t


def _parse_columns_used(similar_queries: Sequence[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for sq in (similar_queries or [])[:5]:
        cu = sq.get("columns_used")
        if isinstance(cu, str):
            try:
                cu = json.loads(cu)
            except Exception:
                cu = None
        if isinstance(cu, (list, tuple)):
            for x in cu:
                s = str(x or "").strip().lower()
                if s and s.count(".") >= 2:
                    out.add(s)
    return out


def _join_columns_by_table(
    fk_relationships: Sequence[Dict[str, Any]],
) -> Dict[str, Set[str]]:
    """
    Build per-table set of join column names from FK relationships.
    Key: schema.table (lowercase)
    """
    out: Dict[str, Set[str]] = {}
    for rel in fk_relationships or []:
        fs = str(rel.get("from_schema") or "").strip()
        ft = str(rel.get("from_table") or "").strip()
        fc = str(rel.get("from_column") or "").strip()
        ts = str(rel.get("to_schema") or "").strip()
        tt = str(rel.get("to_table") or "").strip()
        tc = str(rel.get("to_column") or "").strip()
        if fs and ft and fc:
            out.setdefault(_tfqn_l(fs, ft), set()).add(fc.lower())
        if ts and tt and tc:
            out.setdefault(_tfqn_l(ts, tt), set()).add(tc.lower())
    return out


def _boost_for_meta(*, col_name: str, dtype: str, has_enum: bool, is_pk: bool) -> float:
    n = str(col_name or "")
    dt = str(dtype or "")
    b = 0.0
    if is_pk:
        b += 0.25
    if has_enum:
        b += 0.35
    if _RE_NAME_CODE.search(n):
        b += 0.20
    if _RE_METRIC_NAME.search(n) or _RE_NUM.search(dt):
        b += 0.15
    if _RE_TIME_NAME.search(n) or _RE_TIME.search(dt):
        b += 0.15
    if _RE_TEXT.search(dt):
        b += 0.05
    return float(b)


async def search_columns_per_table(
    *,
    context,
    schema_embedding: List[float],
    selected_tables: Sequence[TableCandidate],
    per_table_k: int,
    all_keywords: Sequence[str],
    fk_relationships: Sequence[Dict[str, Any]] | None = None,
    similar_queries: Sequence[Dict[str, Any]] | None = None,
    table_schemas: Sequence[Dict[str, Any]] | None = None,
    result_parts: List[str],
) -> ColumnSearchResult:
    """
    Per-table column retrieval (table-scoped) + safety expansion + role/slot selection.
    """
    per_table_k = max(int(per_table_k), 1)
    picked_tables = list(selected_tables)

    SmartLogger.log(
        "DEBUG",
        "react.build_sql_context.column_search.per_table.start",
        category="react.tool.detail.build_sql_context",
        params={
            "per_table_k": int(per_table_k),
            "tables_count": len(picked_tables),
            "schema_filter": getattr(context, "schema_filter", None),
        },
        max_inline_chars=0,
    )

    per_table_columns: Dict[str, List[ColumnCandidate]] = {}
    per_table_mode = "not_run"
    column_candidates: List[ColumnCandidate] = []
    try:
        per_table_columns, per_table_mode = await _neo4j_search_table_scoped_columns(
            context=context,
            embedding=schema_embedding,
            tables=picked_tables,
            per_table_k=per_table_k,
        )

        anchor_like_cols = await _neo4j_fetch_anchor_like_columns_for_tables(
            context=context,
            tables=picked_tables,
            name_substrings_lower=list(_ANCHOR_COLUMN_SUBSTRINGS),
            keywords_lower=[k.lower() for k in list(all_keywords)[:20]],
            per_table_limit=max(5, min(15, per_table_k)),
        )
        anchor_like_added = 0
        for c in anchor_like_cols:
            tfqn_l = (
                f"{(c.table_schema or '').lower()}.{(c.table_name or '').lower()}"
                if c.table_schema
                else (c.table_name or "").lower()
            )
            if not tfqn_l:
                continue
            existing = per_table_columns.get(tfqn_l, [])
            seen = {x.name.lower() for x in existing if x.name}
            if c.name and c.name.lower() not in seen:
                per_table_columns.setdefault(tfqn_l, []).append(c)
                anchor_like_added += 1

        # Fetch full schema metadata for selected tables (includes enum_values/cardinality/is_primary_key).
        # Allow caller to pass pre-fetched schemas to avoid duplicate Neo4j calls.
        if table_schemas is None:
            table_schemas = await _neo4j_fetch_table_schemas(context=context, tables=picked_tables)
        meta_by_table: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for t in table_schemas or []:
            tfqn = _tfqn_l(str(t.get("schema") or ""), str(t.get("name") or ""))
            cols = t.get("columns") or []
            m: Dict[str, Dict[str, Any]] = {}
            for c in cols:
                if not isinstance(c, dict):
                    continue
                nm = str(c.get("name") or "").strip()
                if not nm:
                    continue
                m[nm.lower()] = c
            meta_by_table[tfqn] = m

        join_cols_by_table = _join_columns_by_table(list(fk_relationships or []))
        prior_cols_used = _parse_columns_used(list(similar_queries or []))

        # Re-score and select columns per table (generic, not domain-hardcoded).
        per_table_selected: Dict[str, List[ColumnCandidate]] = {}
        for t in picked_tables:
            tfqn = _tfqn_l(t.schema, t.name)
            base_cols = list(per_table_columns.get(tfqn, []) or [])
            meta_cols = meta_by_table.get(tfqn, {})
            join_cols = join_cols_by_table.get(tfqn, set())

            # Candidate pool keyed by column name (lowercased).
            by_name: Dict[str, ColumnCandidate] = {}
            for c in base_cols:
                nm = (c.name or "").strip()
                if not nm:
                    continue
                key = nm.lower()
                prev = by_name.get(key)
                if prev is None or float(c.score or 0.0) > float(prev.score or 0.0):
                    by_name[key] = c

            # Ensure join columns are present (even if vector search didn't return them).
            for nm_l in sorted(join_cols):
                if nm_l in by_name:
                    continue
                meta = meta_cols.get(nm_l) or {}
                by_name[nm_l] = ColumnCandidate(
                    table_schema=str(t.schema or ""),
                    table_name=str(t.name or ""),
                    name=str(meta.get("name") or nm_l),
                    dtype=str(meta.get("dtype") or ""),
                    description=str(meta.get("description") or ""),
                    score=0.0,
                )

            # Add a few enum-like columns (cached) as filter candidates.
            enum_added = 0
            for nm_l, meta in meta_cols.items():
                if enum_added >= 12:
                    break
                if nm_l in by_name:
                    continue
                if meta.get("enum_values"):
                    by_name[nm_l] = ColumnCandidate(
                        table_schema=str(t.schema or ""),
                        table_name=str(t.name or ""),
                        name=str(meta.get("name") or nm_l),
                        dtype=str(meta.get("dtype") or ""),
                        description=str(meta.get("description") or ""),
                        score=0.0,
                    )
                    enum_added += 1

            # 1) forced join columns first (stable)
            selected: List[ColumnCandidate] = []
            seen: Set[str] = set()
            for nm_l in sorted(join_cols):
                c = by_name.get(nm_l)
                if not c:
                    continue
                key = (c.name or "").strip().lower()
                if not key or key in seen:
                    continue
                selected.append(c)
                seen.add(key)

            # 2) then fill by boosted score
            scored: List[Tuple[float, ColumnCandidate]] = []
            schema_l = (t.schema or "").strip().lower()
            table_l = (t.name or "").strip().lower()
            for nm_l, c in by_name.items():
                meta = meta_cols.get(nm_l) or {}
                dtype = str(meta.get("dtype") or c.dtype or "")
                has_enum = bool(meta.get("enum_values"))
                is_pk = bool(meta.get("is_primary_key"))
                base = float(c.score or 0.0)
                fqn = f"{schema_l}.{table_l}.{nm_l}"
                prior = 0.45 if fqn in prior_cols_used else 0.0
                join_boost = 1.25 if nm_l in join_cols else 0.0
                meta_boost = _boost_for_meta(
                    col_name=str(c.name or ""),
                    dtype=dtype,
                    has_enum=has_enum,
                    is_pk=is_pk,
                )
                final = base + prior + join_boost + meta_boost
                scored.append(
                    (
                        float(final),
                        ColumnCandidate(
                            table_schema=str(t.schema or ""),
                            table_name=str(t.name or ""),
                            name=str(c.name or ""),
                            dtype=str(dtype or ""),
                            description=str(meta.get("description") or c.description or ""),
                            score=float(final),
                        ),
                    )
                )

            scored.sort(key=lambda x: float(x[0]), reverse=True)
            hard_cap = min(30, max(per_table_k, len(selected)))
            for _, c in scored:
                if len(selected) >= hard_cap:
                    break
                key = (c.name or "").strip().lower()
                if not key or key in seen:
                    continue
                selected.append(c)
                seen.add(key)
                if len(selected) >= per_table_k:
                    break

            # 3) Pair completion: if we selected a CODE/ID-like column, try to also include a NAME/TITLE-like column
            # (and vice versa) for user-friendly output schemas. This is generic (not domain-hardcoded).
            #
            # NOTE: This does not guarantee the LLM will select these columns, but it makes them available
            # in Context.top_columns under the per_table_k budget.
            try:
                if int(per_table_k) > 0 and len(join_cols) < int(per_table_k):
                    has_code = any(_is_code_like(c.name) for c in selected)
                    has_name = any(_is_name_like(c.name) for c in selected)

                    def _best_meta_match(*, want: str) -> Optional[ColumnCandidate]:
                        """
                        Pick a good candidate from table schema metadata.
                        want in {"name", "code"}.
                        """
                        best: Tuple[float, str] = (-1.0, "")
                        for nm_l, meta in meta_cols.items():
                            if want == "name" and not _is_name_like(nm_l):
                                continue
                            if want == "code" and not _is_code_like(nm_l):
                                continue
                            base_score = float(getattr(by_name.get(nm_l), "score", 0.0) or 0.0)
                            bonus = _name_hint_bonus(nm_l) if want == "name" else _code_hint_bonus(nm_l)
                            sc = base_score + float(bonus)
                            if sc > best[0]:
                                best = (sc, nm_l)
                        if not best[1]:
                            return None
                        nm_l = best[1]
                        # Prefer the already-built candidate (keeps dtype/desc if present); else create from meta.
                        cand = by_name.get(nm_l)
                        if cand is not None:
                            return cand
                        meta = meta_cols.get(nm_l) or {}
                        nm = str(meta.get("name") or nm_l).strip()
                        if not nm:
                            return None
                        return ColumnCandidate(
                            table_schema=str(t.schema or ""),
                            table_name=str(t.name or ""),
                            name=nm,
                            dtype=str(meta.get("dtype") or ""),
                            description=str(meta.get("description") or ""),
                            score=float(best[0]),
                        )

                    to_add: List[ColumnCandidate] = []
                    if has_code and not has_name:
                        c = _best_meta_match(want="name")
                        if c is not None:
                            to_add.append(c)
                    elif has_name and not has_code:
                        c = _best_meta_match(want="code")
                        if c is not None:
                            to_add.append(c)

                    added_keys: Set[str] = set()
                    for c in to_add:
                        key = (c.name or "").strip().lower()
                        if not key or key in seen:
                            continue
                        selected.append(c)
                        seen.add(key)
                        added_keys.add(key)

                    # If we exceeded budget, drop the lowest-score non-join, non-added columns.
                    if len(selected) > int(per_table_k):
                        keep: Set[str] = set(nm.lower() for nm in join_cols)
                        keep.update(added_keys)
                        while len(selected) > int(per_table_k):
                            # Choose a removable candidate with the lowest score.
                            removable = [
                                (float(getattr(c, "score", 0.0) or 0.0), i)
                                for i, c in enumerate(selected)
                                if ((c.name or "").strip().lower() not in keep)
                            ]
                            if not removable:
                                break
                            removable.sort(key=lambda x: float(x[0]))
                            _sc, idx_to_drop = removable[0]
                            dropped = selected.pop(idx_to_drop)
                            seen.discard((dropped.name or "").strip().lower())

                    # Reorder: join columns first, then by score (desc) for stable top_columns readability.
                    join_set = set(nm.lower() for nm in join_cols)
                    join_first = [c for c in selected if (c.name or "").strip().lower() in join_set]
                    rest = [c for c in selected if (c.name or "").strip().lower() not in join_set]
                    rest.sort(key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)
                    selected = join_first + rest
            except Exception:
                # Never fail schema retrieval because of UX enrichment.
                pass

            per_table_selected[tfqn] = selected

        flat: List[ColumnCandidate] = []
        per_table_columns = {}
        for tfqn_l, cols in per_table_selected.items():
            per_table_columns[tfqn_l] = cols
            flat.extend(cols)
        column_candidates = sorted(flat, key=lambda x: float(x.score or 0.0), reverse=True)

        SmartLogger.log(
            "DEBUG",
            "react.build_sql_context.column_search.per_table.done",
            category="react.tool.detail.build_sql_context",
            params={
                "mode": per_table_mode,
                "tables_count": len(picked_tables),
                "per_table_k": int(per_table_k),
                "anchor_like_added": int(anchor_like_added),
                "flattened_columns_count": len(column_candidates),
                "has_fk_relationships": bool(fk_relationships),
                "has_similar_queries": bool(similar_queries),
            },
            max_inline_chars=0,
        )
    except Exception as exc:
        SmartLogger.log(
            "ERROR",
            "react.build_sql_context.column_search.error",
            category="react.tool.detail.build_sql_context",
            params={"error": str(exc), "traceback": traceback.format_exc()},
            max_inline_chars=0,
        )
        result_parts.append(f"<warning>column_vec_search_failed: {xml_escape(str(exc)[:160])}</warning>")

    return ColumnSearchResult(
        selected_tables=picked_tables,
        per_table_columns=per_table_columns,
        per_table_mode=str(per_table_mode),
        column_candidates=column_candidates,
    )


__all__ = ["ColumnSearchResult", "search_columns_per_table"]


