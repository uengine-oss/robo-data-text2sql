from __future__ import annotations

"""
ToolResult-backed metadata merge.

Goal:
- Parse build_sql_context tool_result XML (<build_sql_context_result>...) and merge it into
  ReactMetadata in an idempotent, key-deduped way.
- Keep LLM outputs (XML streaming) unchanged; this is purely server-side memory enrichment.
"""

import re
import xml.etree.ElementTree as ET
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.react.state import ReactMetadata


def _norm(s: Optional[str]) -> str:
    return (s or "").strip()


def _norm_l(s: Optional[str]) -> str:
    return _norm(s).lower()


def _table_key(schema: str, table: str) -> str:
    s = _norm_l(schema)
    t = _norm_l(table)
    return f"{s}.{t}" if s else t


def _column_key(schema: str, table: str, column: str) -> str:
    return f"{_table_key(schema, table)}.{_norm_l(column)}"


def _fk_key(
    *,
    from_schema: str,
    from_table: str,
    from_column: str,
    to_schema: str,
    to_table: str,
    to_column: str,
) -> str:
    left = _column_key(from_schema, from_table, from_column)
    right = _column_key(to_schema, to_table, to_column)
    return f"{left}->{right}"


def _value_key(schema: str, table: str, column: str, actual_value: str) -> str:
    return f"{_column_key(schema, table, column)}={_norm_l(actual_value)}"


def _truncate(text: str, limit: int) -> str:
    s = _norm(text)
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    return s[: max(0, int(limit) - 1)] + "…"


@dataclass
class MergeStats:
    added: Dict[str, int] = field(default_factory=dict)
    updated: Dict[str, int] = field(default_factory=dict)
    skipped: Dict[str, int] = field(default_factory=dict)
    invalid: Dict[str, int] = field(default_factory=dict)

    def bump(self, bucket: str, category: str, inc: int = 1) -> None:
        target = getattr(self, bucket)
        target[category] = int(target.get(category, 0)) + int(inc)


@dataclass
class ParsedBuildSqlContext:
    tables: List[Dict[str, str]] = field(default_factory=list)
    columns: List[Dict[str, str]] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)
    values: List[Dict[str, str]] = field(default_factory=list)


def _safe_fromstring(xml_text: str) -> Optional[ET.Element]:
    try:
        return ET.fromstring(xml_text)
    except ET.ParseError:
        return None


def _find_build_sql_context_result(root: ET.Element) -> Optional[ET.Element]:
    if root.tag == "build_sql_context_result":
        return root
    node = root.find("build_sql_context_result")
    if node is not None:
        return node
    return root.find(".//build_sql_context_result")


_FROM_RE = re.compile(r'FROM\s+"([^"]+)"\."([^"]+)"\s+([A-Za-z_][A-Za-z0-9_]*)', re.IGNORECASE)
_SELECT_ALIAS_COL_RE = re.compile(
    r'([A-Za-z_][A-Za-z0-9_]*)\."([^"]+)"'
)


def _extract_table_and_selected_cols_from_sql(sql: str) -> Optional[Tuple[str, str, Dict[str, str]]]:
    """
    Best-effort extraction:
    - Find first FROM "SCHEMA"."TABLE" alias
    - Find SELECT alias."COL" occurrences and return alias->col list (for that alias only)

    Returns (schema, table, {"name_col": "...", "code_col": "..."}) when pair-like,
    else None.
    """
    s = (sql or "").strip()
    if not s:
        return None
    m = _FROM_RE.search(s)
    if not m:
        return None
    schema, table, alias = m.group(1), m.group(2), m.group(3)
    cols = []
    for am in _SELECT_ALIAS_COL_RE.finditer(s):
        if am.group(1) != alias:
            continue
        cols.append(am.group(2))
    # Need at least 2 columns to form a pair
    if len(cols) < 2:
        return None
    # Try to pick one name-like and one code/id-like column.
    def score_name(c: str) -> int:
        cl = c.lower()
        return int("name" in cl or cl.endswith("_nm") or cl == "nm" or cl.endswith("nm"))

    def score_code(c: str) -> int:
        cl = c.lower()
        return int("code" in cl or "id" in cl or cl.endswith("_cd") or cl.endswith("_code") or cl.endswith("_id"))

    name_cands = sorted(cols, key=lambda x: score_name(x), reverse=True)
    code_cands = sorted(cols, key=lambda x: score_code(x), reverse=True)
    name_col = name_cands[0] if score_name(name_cands[0]) else ""
    code_col = code_cands[0] if score_code(code_cands[0]) else ""
    if not name_col or not code_col or _norm_l(name_col) == _norm_l(code_col):
        return None
    return (schema, table, {"name_col": name_col, "code_col": code_col})


def parse_build_sql_context_tool_result(
    tool_result_xml: str,
    *,
    description_limit: int = 600,
) -> ParsedBuildSqlContext:
    """
    Parse build_sql_context tool_result (<tool_result>...</tool_result>) into normalized items.
    """
    parsed = ParsedBuildSqlContext()
    root = _safe_fromstring((tool_result_xml or "").strip())
    if root is None:
        return parsed

    b = _find_build_sql_context_result(root)
    if b is None:
        return parsed

    # 1) Tables from schema_candidates/tables
    for t in b.findall(".//schema_candidates/tables/table"):
        schema = _norm(t.findtext("schema"))
        name = _norm(t.findtext("name"))
        if not name:
            continue
        desc = _truncate(t.findtext("description") or "", int(description_limit))
        parsed.tables.append(
            {
                "schema": schema,
                "name": name,
                "purpose": "tool_grounded",
                "key_columns": "",
                "description": desc,
            }
        )

    # 2) Columns from per_table_columns
    for tb in b.findall(".//schema_candidates/per_table_columns/table"):
        schema = _norm(tb.findtext("schema"))
        table = _norm(tb.findtext("name"))
        if not table:
            continue
        for c in tb.findall(".//columns/column"):
            col = _norm(c.findtext("name"))
            if not col:
                continue
            dtype = _norm(c.findtext("dtype"))
            parsed.columns.append(
                {
                    "schema": schema,
                    "table": table,
                    "name": col,
                    "data_type": dtype,
                    "purpose": "tool_candidate",
                }
            )

    # 3) FK relationships
    for fk in b.findall(".//fk_relationships/fk"):
        fs = _norm(fk.findtext("from_schema"))
        ft = _norm(fk.findtext("from_table"))
        fc = _norm(fk.findtext("from_column"))
        ts = _norm(fk.findtext("to_schema"))
        tt = _norm(fk.findtext("to_table"))
        tc = _norm(fk.findtext("to_column"))
        if not (ft and fc and tt and tc):
            continue
        left = f'{fs}.{ft}.{fc}' if fs else f"{ft}.{fc}"
        right = f'{ts}.{tt}.{tc}' if ts else f"{tt}.{tc}"
        parsed.relationships.append(
            {
                "type": "FK",
                "tables": f"{fs}.{ft} -> {ts}.{tt}".strip(),
                "condition": f"{left} = {right}",
            }
        )

    # 4) resolved_values -> identified_values
    for v in b.findall(".//resolved_values/value"):
        user_term = _norm(v.findtext("user_term"))
        actual_value = _norm(v.findtext("actual_value"))
        column_fqn = _norm(v.findtext("column_fqn"))
        if not actual_value or not column_fqn:
            continue
        # Expect schema.table.column (best-effort). If schema missing, keep schema empty.
        parts = [p for p in column_fqn.split(".") if p]
        if len(parts) < 2:
            continue
        if len(parts) == 2:
            schema, table, col = "", parts[0], parts[1]
        else:
            schema, table, col = parts[0], parts[1], parts[2]
        if not table or not col:
            continue
        parsed.values.append(
            {
                "schema": schema,
                "table": table,
                "column": col,
                "actual_value": actual_value,
                "user_term": user_term,
                "source": "resolved_values",
            }
        )

    # 5) light_queries preview -> values (heuristic: NAME <-> CODE/ID pair)
    for q in b.findall(".//light_queries/query"):
        verdict = _norm(q.findtext("verdict")).upper()
        if verdict != "PASS":
            continue
        sql = q.findtext("sql") or ""
        preview = q.find("preview")
        if preview is None:
            continue
        try:
            row_count = int(_norm(preview.findtext("row_count")) or "0")
        except Exception:
            row_count = 0
        if row_count <= 0 or row_count > 5:
            continue

        sel = _extract_table_and_selected_cols_from_sql(sql)
        if not sel:
            continue
        schema, table, pair = sel
        name_col = pair["name_col"]
        code_col = pair["code_col"]

        for row in preview.findall(".//rows/row"):
            cells = row.findall("./cell")
            if not cells:
                continue
            name_val = ""
            code_val = ""
            for cell in cells:
                col_name = _norm(cell.get("column"))
                val = "".join(cell.itertext()).strip()
                if _norm_l(col_name) == _norm_l(name_col):
                    name_val = val
                elif _norm_l(col_name) == _norm_l(code_col):
                    code_val = val
            if not name_val or not code_val:
                continue
            parsed.values.append(
                {
                    "schema": schema,
                    "table": table,
                    "column": code_col,
                    "actual_value": code_val,
                    "user_term": name_val,
                    "source": "light_queries_preview",
                }
            )

    return parsed


def merge_parsed_into_metadata(
    metadata: ReactMetadata,
    parsed: ParsedBuildSqlContext,
    *,
    description_limit: int = 600,
) -> MergeStats:
    """
    Merge parsed items into ReactMetadata in a key-deduped, idempotent way.
    """
    stats = MergeStats()

    # Build indexes to existing entries
    table_idx: Dict[str, Dict[str, Any]] = {}
    for e in metadata.identified_tables:
        k = _table_key(str(e.get("schema") or ""), str(e.get("name") or ""))
        if not k:
            continue
        table_idx.setdefault(k, e)

    col_idx: Dict[str, Dict[str, Any]] = {}
    for e in metadata.identified_columns:
        k = _column_key(str(e.get("schema") or ""), str(e.get("table") or ""), str(e.get("name") or ""))
        if not k:
            continue
        col_idx.setdefault(k, e)

    rel_idx: Dict[str, Dict[str, Any]] = {}
    for e in metadata.identified_relationships:
        ty = str(e.get("type") or "")
        cond = str(e.get("condition") or "")
        tables = str(e.get("tables") or "")
        k = _norm_l(f"{ty}|{tables}|{cond}")
        if not k:
            continue
        rel_idx.setdefault(k, e)

    val_idx: Dict[str, Dict[str, Any]] = {}
    for e in metadata.identified_values:
        k = _value_key(
            str(e.get("schema") or ""),
            str(e.get("table") or ""),
            str(e.get("column") or ""),
            str(e.get("actual_value") or ""),
        )
        if not k:
            continue
        val_idx.setdefault(k, e)

    # Tables
    for t in parsed.tables:
        schema = _norm(t.get("schema"))
        name = _norm(t.get("name"))
        if not name:
            stats.bump("invalid", "tables")
            continue
        k = _table_key(schema, name)
        incoming_desc = _truncate(t.get("description") or "", int(description_limit))
        incoming = {
            "schema": schema,
            "name": name,
            "purpose": _norm(t.get("purpose")) or "tool_grounded",
            "key_columns": _norm(t.get("key_columns")) or "",
            "description": incoming_desc,
        }
        existing = table_idx.get(k)
        if existing is None:
            metadata.identified_tables.append(incoming)
            table_idx[k] = metadata.identified_tables[-1]
            stats.bump("added", "tables")
            continue

        updated = False
        if incoming.get("schema") and not _norm(existing.get("schema")):
            existing["schema"] = incoming["schema"]
            updated = True
        if incoming_desc and len(incoming_desc) > len(_norm(existing.get("description"))):
            # Prefer longer (still truncated) description.
            if incoming_desc != _norm(existing.get("description")):
                existing["description"] = incoming_desc
                updated = True
        if not _norm(existing.get("purpose")) and incoming.get("purpose"):
            existing["purpose"] = incoming["purpose"]
            updated = True
        if updated:
            stats.bump("updated", "tables")
        else:
            stats.bump("skipped", "tables")

    # Columns
    for c in parsed.columns:
        schema = _norm(c.get("schema"))
        table = _norm(c.get("table"))
        name = _norm(c.get("name"))
        if not table or not name:
            stats.bump("invalid", "columns")
            continue
        k = _column_key(schema, table, name)
        incoming = {
            "schema": schema,
            "table": table,
            "name": name,
            "data_type": _norm(c.get("data_type")) or "",
            "purpose": _norm(c.get("purpose")) or "tool_candidate",
        }
        existing = col_idx.get(k)
        if existing is None:
            metadata.identified_columns.append(incoming)
            col_idx[k] = metadata.identified_columns[-1]
            stats.bump("added", "columns")
            continue

        updated = False
        if incoming.get("schema") and not _norm(existing.get("schema")):
            existing["schema"] = incoming["schema"]
            updated = True
        if incoming.get("data_type") and not _norm(existing.get("data_type")):
            existing["data_type"] = incoming["data_type"]
            updated = True
        if not _norm(existing.get("purpose")) and incoming.get("purpose"):
            existing["purpose"] = incoming["purpose"]
            updated = True
        if updated:
            stats.bump("updated", "columns")
        else:
            stats.bump("skipped", "columns")

    # Relationships
    for r in parsed.relationships:
        ty = _norm(r.get("type"))
        tables = _norm(r.get("tables"))
        cond = _norm(r.get("condition"))
        if not (ty and cond):
            stats.bump("invalid", "relationships")
            continue
        k = _norm_l(f"{ty}|{tables}|{cond}")
        incoming = {"type": ty, "tables": tables, "condition": cond}
        if k not in rel_idx:
            metadata.identified_relationships.append(incoming)
            rel_idx[k] = metadata.identified_relationships[-1]
            stats.bump("added", "relationships")
        else:
            stats.bump("skipped", "relationships")

    # Values
    for v in parsed.values:
        schema = _norm(v.get("schema"))
        table = _norm(v.get("table"))
        col = _norm(v.get("column"))
        actual_value = _norm(v.get("actual_value"))
        user_term = _norm(v.get("user_term"))
        if not (table and col and actual_value):
            stats.bump("invalid", "values")
            continue
        k = _value_key(schema, table, col, actual_value)
        incoming = {
            "schema": schema,
            "table": table,
            "column": col,
            "actual_value": actual_value,
            "user_term": user_term,
        }
        existing = val_idx.get(k)
        if existing is None:
            metadata.identified_values.append(incoming)
            val_idx[k] = metadata.identified_values[-1]
            stats.bump("added", "values")
            continue

        # Merge user_term (avoid noisy growth: keep up to 3 unique terms).
        updated = False
        if user_term:
            existing_ut = _norm(existing.get("user_term"))
            if not existing_ut:
                existing["user_term"] = user_term
                updated = True
            else:
                parts = [p.strip() for p in existing_ut.split("|") if p.strip()]
                if user_term not in parts and len(parts) < 3:
                    existing["user_term"] = existing_ut + "|" + user_term
                    updated = True
        if updated:
            stats.bump("updated", "values")
        else:
            stats.bump("skipped", "values")

    return stats


def merge_build_sql_context_tool_result_into_metadata(
    *,
    metadata: ReactMetadata,
    tool_result_xml: str,
    description_limit: int = 600,
) -> MergeStats:
    parsed = parse_build_sql_context_tool_result(
        tool_result_xml, description_limit=int(description_limit)
    )
    return merge_parsed_into_metadata(
        metadata, parsed, description_limit=int(description_limit)
    )


def merge_build_sql_context_tool_results(
    *,
    base_tool_result_xml: str,
    new_tool_result_xml: str,
) -> str:
    """
    Merge two build_sql_context <tool_result> XML strings into a single <tool_result>.

    This is a best-effort, idempotent merge to prevent ask_user loops when the initial
    build_sql_context missed critical tables/values and the user provides clarifying info.

    Merge focus (high-signal blocks used by the controller):
    - <schema_candidates> (tables + per_table_columns)
    - <column_value_hints>
    - <resolved_values>
    - <fk_relationships>
    """
    base_text = (base_tool_result_xml or "").strip()
    new_text = (new_tool_result_xml or "").strip()
    if not base_text:
        return new_text
    if not new_text:
        return base_text

    base_root = _safe_fromstring(base_text)
    new_root = _safe_fromstring(new_text)
    if base_root is None:
        return new_text
    if new_root is None:
        return base_text

    base_b = _find_build_sql_context_result(base_root)
    new_b = _find_build_sql_context_result(new_root)
    if base_b is None:
        return new_text
    if new_b is None:
        return base_text

    _merge_schema_candidates(base_b=base_b, new_b=new_b)
    _merge_column_value_hints(base_b=base_b, new_b=new_b)
    _merge_resolved_values(base_b=base_b, new_b=new_b)
    _merge_fk_relationships(base_b=base_b, new_b=new_b)
    _merge_light_queries(base_b=base_b, new_b=new_b)
    _promote_light_queries_preview_to_resolved_values(base_b=base_b)

    return ET.tostring(base_root, encoding="unicode")


def _find_or_create(parent: ET.Element, tag: str) -> ET.Element:
    node = parent.find(tag)
    if node is not None:
        return node
    node = ET.SubElement(parent, tag)
    return node


def _parse_float(text: Optional[str]) -> Optional[float]:
    s = (text or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _set_child_text(parent: ET.Element, tag: str, value: str) -> None:
    node = parent.find(tag)
    if node is None:
        node = ET.SubElement(parent, tag)
    node.text = value


def _merge_schema_candidates(*, base_b: ET.Element, new_b: ET.Element) -> None:
    new_sc = new_b.find("schema_candidates")
    if new_sc is None:
        return
    base_sc = base_b.find("schema_candidates")
    if base_sc is None:
        base_b.append(copy.deepcopy(new_sc))
        return

    # 1) tables
    new_tables = new_sc.find("tables")
    if new_tables is not None:
        base_tables = base_sc.find("tables")
        if base_tables is None:
            base_sc.append(copy.deepcopy(new_tables))
        else:
            idx: Dict[str, ET.Element] = {}
            for t in list(base_tables.findall("table")):
                schema = _norm(t.findtext("schema"))
                name = _norm(t.findtext("name"))
                if not name:
                    continue
                idx[_table_key(schema, name)] = t
            for t in list(new_tables.findall("table")):
                schema = _norm(t.findtext("schema"))
                name = _norm(t.findtext("name"))
                if not name:
                    continue
                key = _table_key(schema, name)
                existing = idx.get(key)
                if existing is None:
                    base_tables.append(copy.deepcopy(t))
                    idx[key] = base_tables.findall("table")[-1]
                    continue

                # Merge fields: schema fill, prefer longer description, max score.
                desc_new = _norm(t.findtext("description"))
                desc_old = _norm(existing.findtext("description"))
                if desc_new and len(desc_new) > len(desc_old):
                    _set_child_text(existing, "description", desc_new)

                if schema and not _norm(existing.findtext("schema")):
                    _set_child_text(existing, "schema", schema)

                sc_new = _parse_float(t.findtext("score"))
                sc_old = _parse_float(existing.findtext("score"))
                if sc_new is not None:
                    if (sc_old is None) or (sc_new > float(sc_old)):
                        _set_child_text(existing, "score", f"{sc_new:.4f}")

    # 2) per_table_columns
    new_pt = new_sc.find("per_table_columns")
    if new_pt is not None:
        base_pt = base_sc.find("per_table_columns")
        if base_pt is None:
            base_sc.append(copy.deepcopy(new_pt))
        else:
            tb_idx: Dict[str, ET.Element] = {}
            for tb in list(base_pt.findall("table")):
                schema = _norm(tb.findtext("schema"))
                name = _norm(tb.findtext("name"))
                if not name:
                    continue
                tb_idx[_table_key(schema, name)] = tb
            for tb in list(new_pt.findall("table")):
                schema = _norm(tb.findtext("schema"))
                name = _norm(tb.findtext("name"))
                if not name:
                    continue
                key = _table_key(schema, name)
                existing_tb = tb_idx.get(key)
                if existing_tb is None:
                    base_pt.append(copy.deepcopy(tb))
                    tb_idx[key] = base_pt.findall("table")[-1]
                    continue

                base_cols = _find_or_create(existing_tb, "columns")
                new_cols = tb.find("columns")
                if new_cols is None:
                    continue

                col_idx: Dict[str, ET.Element] = {}
                for c in list(base_cols.findall("column")):
                    cn = _norm(c.findtext("name"))
                    if cn:
                        col_idx[cn.lower()] = c
                for c in list(new_cols.findall("column")):
                    cn = _norm(c.findtext("name"))
                    if not cn:
                        continue
                    existing_c = col_idx.get(cn.lower())
                    if existing_c is None:
                        base_cols.append(copy.deepcopy(c))
                        col_idx[cn.lower()] = base_cols.findall("column")[-1]
                        continue

                    dtype_new = _norm(c.findtext("dtype"))
                    if dtype_new and not _norm(existing_c.findtext("dtype")):
                        _set_child_text(existing_c, "dtype", dtype_new)

                    desc_new = _norm(c.findtext("description"))
                    desc_old = _norm(existing_c.findtext("description"))
                    if desc_new and len(desc_new) > len(desc_old):
                        _set_child_text(existing_c, "description", desc_new)

                    sc_new = _parse_float(c.findtext("score"))
                    sc_old = _parse_float(existing_c.findtext("score"))
                    if sc_new is not None:
                        if (sc_old is None) or (sc_new > float(sc_old)):
                            _set_child_text(existing_c, "score", f"{sc_new:.4f}")


def _merge_column_value_hints(*, base_b: ET.Element, new_b: ET.Element) -> None:
    new_root = new_b.find("column_value_hints")
    if new_root is None:
        return
    base_root = base_b.find("column_value_hints")
    if base_root is None:
        base_b.append(copy.deepcopy(new_root))
        return

    def table_key(t: ET.Element) -> str:
        return _table_key(_norm(t.findtext("schema")), _norm(t.findtext("name")))

    base_tables: Dict[str, ET.Element] = {}
    for t in list(base_root.findall("table")):
        k = table_key(t)
        if k:
            base_tables[k] = t

    for t in list(new_root.findall("table")):
        k = table_key(t)
        if not k:
            continue
        existing_t = base_tables.get(k)
        if existing_t is None:
            base_root.append(copy.deepcopy(t))
            base_tables[k] = base_root.findall("table")[-1]
            continue

        base_cols = _find_or_create(existing_t, "columns")
        new_cols = t.find("columns")
        if new_cols is None:
            continue

        col_idx: Dict[str, ET.Element] = {}
        for c in list(base_cols.findall("column")):
            nm = _norm(c.findtext("name"))
            if nm:
                col_idx[nm.lower()] = c
        for c in list(new_cols.findall("column")):
            nm = _norm(c.findtext("name"))
            if not nm:
                continue
            existing_c = col_idx.get(nm.lower())
            if existing_c is None:
                base_cols.append(copy.deepcopy(c))
                col_idx[nm.lower()] = base_cols.findall("column")[-1]
                continue

            dtype_new = _norm(c.findtext("dtype"))
            if dtype_new and not _norm(existing_c.findtext("dtype")):
                _set_child_text(existing_c, "dtype", dtype_new)

            card_new = _norm(c.findtext("cardinality"))
            if card_new and not _norm(existing_c.findtext("cardinality")):
                _set_child_text(existing_c, "cardinality", card_new)

            base_vals = _find_or_create(existing_c, "values")
            new_vals = c.find("values")
            if new_vals is None:
                continue

            val_idx: Dict[str, ET.Element] = {}
            for v in list(base_vals.findall("value")):
                txt = _norm(v.findtext("text"))
                if txt:
                    val_idx[txt.lower()] = v
            for v in list(new_vals.findall("value")):
                txt = _norm(v.findtext("text"))
                if not txt:
                    continue
                existing_v = val_idx.get(txt.lower())
                if existing_v is None:
                    base_vals.append(copy.deepcopy(v))
                    val_idx[txt.lower()] = base_vals.findall("value")[-1]
                    continue
                # Keep best (max) count if available.
                cnt_new = _parse_float(v.findtext("count"))
                cnt_old = _parse_float(existing_v.findtext("count"))
                if cnt_new is not None:
                    if (cnt_old is None) or (cnt_new > float(cnt_old)):
                        _set_child_text(existing_v, "count", str(int(cnt_new)))


def _merge_resolved_values(*, base_b: ET.Element, new_b: ET.Element) -> None:
    new_root = new_b.find("resolved_values")
    if new_root is None:
        return
    base_root = base_b.find("resolved_values")
    if base_root is None:
        base_b.append(copy.deepcopy(new_root))
        return

    def rv_key(v: ET.Element) -> str:
        col_fqn = _norm(v.findtext("column_fqn")).lower()
        actual = _norm(v.findtext("actual_value")).lower()
        if col_fqn and actual:
            return f"{col_fqn}={actual}"
        # Fallback key (less stable, but avoids duplication explosions)
        ut = _norm(v.findtext("user_term")).lower()
        return f"{actual}|{ut}" if actual else ""

    idx: Dict[str, ET.Element] = {}
    for v in list(base_root.findall("value")):
        k = rv_key(v)
        if k:
            idx[k] = v

    for v in list(new_root.findall("value")):
        k = rv_key(v)
        if not k:
            continue
        existing = idx.get(k)
        if existing is None:
            base_root.append(copy.deepcopy(v))
            idx[k] = base_root.findall("value")[-1]
            continue

        # Fill missing user_term/column_fqn/source best-effort
        if _norm(v.findtext("user_term")) and not _norm(existing.findtext("user_term")):
            _set_child_text(existing, "user_term", _norm(v.findtext("user_term")))
        if _norm(v.findtext("column_fqn")) and not _norm(existing.findtext("column_fqn")):
            _set_child_text(existing, "column_fqn", _norm(v.findtext("column_fqn")))
        if _norm(v.findtext("source")) and not _norm(existing.findtext("source")):
            _set_child_text(existing, "source", _norm(v.findtext("source")))


def _merge_fk_relationships(*, base_b: ET.Element, new_b: ET.Element) -> None:
    new_root = new_b.find("fk_relationships")
    if new_root is None:
        return
    base_root = base_b.find("fk_relationships")
    if base_root is None:
        base_b.append(copy.deepcopy(new_root))
        return

    def fk_key(fk: ET.Element) -> str:
        fs = _norm(fk.findtext("from_schema"))
        ft = _norm(fk.findtext("from_table"))
        fc = _norm(fk.findtext("from_column"))
        ts = _norm(fk.findtext("to_schema"))
        tt = _norm(fk.findtext("to_table"))
        tc = _norm(fk.findtext("to_column"))
        if not (ft and fc and tt and tc):
            return ""
        return _fk_key(
            from_schema=fs,
            from_table=ft,
            from_column=fc,
            to_schema=ts,
            to_table=tt,
            to_column=tc,
        )

    idx: Dict[str, ET.Element] = {}
    for fk in list(base_root.findall("fk")):
        k = fk_key(fk)
        if k:
            idx[k] = fk

    for fk in list(new_root.findall("fk")):
        k = fk_key(fk)
        if not k:
            continue
        if k in idx:
            continue
        base_root.append(copy.deepcopy(fk))
        idx[k] = base_root.findall("fk")[-1]


def _merge_light_queries(*, base_b: ET.Element, new_b: ET.Element) -> None:
    """
    Merge <light_queries> blocks by deduping on SQL text (normalized).
    We keep base metadata (target_k/mode/timeout) but append unique <query> items from new.
    """
    new_lq = new_b.find("light_queries")
    if new_lq is None:
        return
    base_lq = base_b.find("light_queries")
    if base_lq is None:
        base_b.append(copy.deepcopy(new_lq))
        return

    def _sql_key(q: ET.Element) -> str:
        s = _norm(q.findtext("sql"))
        return " ".join(s.split()).strip().lower()

    existing: Dict[str, ET.Element] = {}
    for q in list(base_lq.findall("query")):
        k = _sql_key(q)
        if k:
            existing[k] = q

    added = 0
    for q in list(new_lq.findall("query")):
        k = _sql_key(q)
        if not k:
            continue
        if k in existing:
            continue
        base_lq.append(copy.deepcopy(q))
        added += 1

    # Renumber query indices for readability (best-effort)
    if added > 0:
        for idx, q in enumerate(list(base_lq.findall("query")), start=1):
            try:
                q.set("index", str(idx))
            except Exception:
                pass


def _promote_light_queries_preview_to_resolved_values(*, base_b: ET.Element) -> None:
    """
    C-2: Promote small, high-confidence light_queries preview results into <resolved_values>
    so downstream controller loops can reuse the mapping without re-running the same light queries.
    """
    lq = base_b.find("light_queries")
    if lq is None:
        return

    rv_root = base_b.find("resolved_values")
    if rv_root is None:
        rv_root = ET.SubElement(base_b, "resolved_values")

    # Build existing key set (align with _merge_resolved_values semantics)
    existing_keys = set()
    for v in list(rv_root.findall("value")):
        col_fqn = _norm(v.findtext("column_fqn")).lower()
        actual = _norm(v.findtext("actual_value")).lower()
        if col_fqn and actual:
            existing_keys.add(f"{col_fqn}={actual}")

    for q in list(lq.findall("query")):
        verdict = _norm(q.findtext("verdict")).upper()
        if verdict != "PASS":
            continue
        preview = q.find("preview")
        if preview is None:
            continue
        try:
            row_count = int(_norm(preview.findtext("row_count")) or "0")
        except Exception:
            row_count = 0
        # Conservative: only promote very small previews to avoid wrong bulk assumptions.
        if row_count <= 0 or row_count > 5:
            continue

        sql = q.findtext("sql") or ""
        sel = _extract_table_and_selected_cols_from_sql(sql)
        if not sel:
            continue
        schema, table, pair = sel
        name_col = pair.get("name_col") or ""
        code_col = pair.get("code_col") or ""
        if not (schema and table and name_col and code_col):
            continue

        for row in preview.findall(".//rows/row"):
            name_val = ""
            code_val = ""
            for cell in row.findall("./cell"):
                col_name = _norm(cell.get("column"))
                val = "".join(cell.itertext()).strip()
                if _norm_l(col_name) == _norm_l(name_col):
                    name_val = val
                elif _norm_l(col_name) == _norm_l(code_col):
                    code_val = val
            if not name_val or not code_val:
                continue
            col_fqn = f"{schema}.{table}.{code_col}".strip(".")
            key = f"{col_fqn.lower()}={code_val.strip().lower()}"
            if key in existing_keys:
                continue
            v = ET.SubElement(rv_root, "value")
            _set_child_text(v, "user_term", name_val)
            _set_child_text(v, "actual_value", code_val)
            _set_child_text(v, "source", "light_queries_preview")
            _set_child_text(v, "column_fqn", col_fqn)
            existing_keys.add(key)



