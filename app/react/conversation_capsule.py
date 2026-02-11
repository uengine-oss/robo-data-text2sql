from __future__ import annotations

import base64
import json
import re
import time
import zlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


# =============================================================================
# Conversation capsule (client-stored token)
# - Designed for 10+ turns: zlib + base64url
# - Stores per-turn: question, final_sql, result_preview, derived_filters, hints, evidence xml
# - The token is NOT signed (tamperable). This is intentional per current policy.
# =============================================================================


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8")


def _unb64url(text: str) -> bytes:
    return base64.urlsafe_b64decode(text.encode("utf-8"))


def _ws_norm(s: str) -> str:
    return " ".join(str(s or "").split()).strip()


def _truncate_list(seq: Sequence[Any], n: int) -> List[Any]:
    return list(seq or [])[: max(0, int(n))]


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _strip_sql_comments(sql: str) -> str:
    s = (sql or "").strip()
    if not s:
        return ""
    # Remove block comments and line comments (best-effort).
    s = re.sub(r"/\*[\s\S]*?\*/", "", s, flags=re.MULTILINE)
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    return s.strip().rstrip(";").strip()


# =============================================================================
# Turn schema
# =============================================================================


@dataclass
class TurnPreview:
    columns: List[str] = field(default_factory=list)
    rows: List[List[Any]] = field(default_factory=list)
    row_count: int = 0
    execution_time_ms: float = 0.0

    @classmethod
    def from_execution_result(cls, execution_result: Optional[Dict[str, Any]]) -> "TurnPreview":
        if not isinstance(execution_result, dict):
            return cls()
        cols = execution_result.get("columns") or []
        rows = execution_result.get("rows") or []
        return cls(
            columns=[str(x) for x in list(cols) if str(x).strip()],
            rows=[list(r) if isinstance(r, list) else [r] for r in list(rows) if r is not None],
            row_count=_safe_int(execution_result.get("row_count"), 0),
            execution_time_ms=_safe_float(execution_result.get("execution_time_ms"), 0.0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "columns": list(self.columns),
            "rows": list(self.rows),
            "row_count": int(self.row_count),
            "execution_time_ms": float(self.execution_time_ms),
        }

    def capped(self, *, max_cols: int, max_rows: int) -> "TurnPreview":
        cols = list(self.columns)[: max(0, int(max_cols))]
        # Truncate each row to the visible columns count
        rows_out: List[List[Any]] = []
        for r in list(self.rows)[: max(0, int(max_rows))]:
            rr = list(r)[: len(cols)]
            rows_out.append(rr)
        return TurnPreview(
            columns=cols,
            rows=rows_out,
            row_count=int(self.row_count),
            execution_time_ms=float(self.execution_time_ms),
        )


@dataclass
class TurnCapsule:
    question: str
    final_sql: str
    preview: TurnPreview = field(default_factory=TurnPreview)
    derived_filters: Dict[str, str] = field(default_factory=dict)
    important_hints: Dict[str, Any] = field(default_factory=dict)
    evidence_ctx_xml: str = ""  # filtered build_sql_context tool_result (optional)
    created_at_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "final_sql": self.final_sql,
            "preview": self.preview.to_dict(),
            "derived_filters": dict(self.derived_filters or {}),
            "important_hints": dict(self.important_hints or {}),
            "evidence_ctx_xml": self.evidence_ctx_xml or "",
            "created_at_ms": int(self.created_at_ms or 0),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurnCapsule":
        pv = data.get("preview") if isinstance(data, dict) else {}
        preview = TurnPreview(
            columns=[str(x) for x in list((pv or {}).get("columns") or [])][:50],
            rows=list((pv or {}).get("rows") or [])[:50],
            row_count=_safe_int((pv or {}).get("row_count"), 0),
            execution_time_ms=_safe_float((pv or {}).get("execution_time_ms"), 0.0),
        )
        df = data.get("derived_filters") if isinstance(data.get("derived_filters"), dict) else {}
        ih = data.get("important_hints") if isinstance(data.get("important_hints"), dict) else {}
        return cls(
            question=_ws_norm(str(data.get("question") or "")),
            final_sql=str(data.get("final_sql") or "").strip(),
            preview=preview,
            derived_filters={str(k): str(v) for k, v in dict(df or {}).items() if str(k).strip()},
            important_hints=dict(ih or {}),
            evidence_ctx_xml=str(data.get("evidence_ctx_xml") or "").strip(),
            created_at_ms=_safe_int(data.get("created_at_ms"), 0),
        )


@dataclass
class ConversationCapsule:
    v: int = 1
    dbms: str = ""
    schema_filter: Optional[List[str]] = None
    turns: List[TurnCapsule] = field(default_factory=list)
    created_at_ms: int = 0
    updated_at_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "v": int(self.v),
            "dbms": str(self.dbms or ""),
            "schema_filter": list(self.schema_filter) if self.schema_filter else None,
            "created_at_ms": int(self.created_at_ms or 0),
            "updated_at_ms": int(self.updated_at_ms or 0),
            "turns": [t.to_dict() for t in list(self.turns or [])],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationCapsule":
        if not isinstance(data, dict):
            return cls()
        turns_raw = data.get("turns") if isinstance(data.get("turns"), list) else []
        turns: List[TurnCapsule] = []
        for t in list(turns_raw)[:200]:
            if isinstance(t, dict):
                turns.append(TurnCapsule.from_dict(t))
        sf = data.get("schema_filter")
        schema_filter = None
        if isinstance(sf, list):
            schema_filter = [str(x).strip() for x in sf if str(x).strip()][:20]
        return cls(
            v=_safe_int(data.get("v"), 1),
            dbms=str(data.get("dbms") or ""),
            schema_filter=schema_filter,
            turns=turns,
            created_at_ms=_safe_int(data.get("created_at_ms"), 0),
            updated_at_ms=_safe_int(data.get("updated_at_ms"), 0),
        )


def new_capsule(*, dbms: str, schema_filter: Optional[List[str]] = None) -> ConversationCapsule:
    now = int(time.time() * 1000)
    return ConversationCapsule(
        v=1,
        dbms=str(dbms or ""),
        schema_filter=list(schema_filter) if schema_filter else None,
        turns=[],
        created_at_ms=now,
        updated_at_ms=now,
    )


def encode_conversation_state(capsule: ConversationCapsule, *, level: int = 6) -> str:
    payload = json.dumps(capsule.to_dict(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(payload, level=int(level))
    return _b64url(comp)


def decode_conversation_state(token: Optional[str]) -> ConversationCapsule:
    t = (token or "").strip()
    if not t:
        return ConversationCapsule()
    try:
        raw = _unb64url(t)
        dec = zlib.decompress(raw)
        obj = json.loads(dec.decode("utf-8"))
        return ConversationCapsule.from_dict(obj if isinstance(obj, dict) else {})
    except Exception:
        # Fail-open: treat as empty conversation.
        return ConversationCapsule()


# =============================================================================
# Derived filters / important hints
# =============================================================================


def extract_derived_filters_from_sql(sql: str, *, max_items: int = 16) -> Dict[str, str]:
    """
    Best-effort extraction of equality filters: "COL" = 'value' or COL = 'value'
    Used as a compact, high-signal memory for follow-up questions.
    """
    s = str(sql or "")
    out: Dict[str, str] = {}
    # MindsDB/MySQL-style identifiers: `COL` = 'value' or `T`.`COL` = 'value'
    for m in re.finditer(r"`(?P<col>[A-Za-z0-9_]+)`\s*=\s*'(?P<val>[^']+)'", s):
        col = (m.group("col") or "").strip()
        val = (m.group("val") or "").strip()
        if not col or not val:
            continue
        k = col.upper()
        if k not in out:
            out[k] = val
        if len(out) >= int(max_items):
            return out
    # Prefer quoted columns first.
    for m in re.finditer(r"\"(?P<col>[A-Za-z0-9_]+)\"\s*=\s*'(?P<val>[^']+)'", s):
        col = (m.group("col") or "").strip()
        val = (m.group("val") or "").strip()
        if not col or not val:
            continue
        k = col.upper()
        if k not in out:
            out[k] = val
        if len(out) >= int(max_items):
            return out
    # Fallback: unquoted columns (riskier)
    for m in re.finditer(r"\b(?P<col>[A-Za-z0-9_]{2,})\b\s*=\s*'(?P<val>[^']+)'", s):
        col = (m.group("col") or "").strip()
        val = (m.group("val") or "").strip()
        if not col or not val:
            continue
        k = col.upper()
        if k not in out:
            out[k] = val
        if len(out) >= int(max_items):
            break
    return out


def _parse_collected_metadata_xml(collected_metadata_xml: str) -> Dict[str, Any]:
    """
    Extract a few stable hints from <collected_metadata>:
    - identified_values (schema/table/column/actual_value/user_term)
    - identified_tables (schema/name)
    """
    text = (collected_metadata_xml or "").strip()
    if not text:
        return {}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return {}
    out: Dict[str, Any] = {}
    # values
    vals: List[Dict[str, str]] = []
    for v in root.findall(".//identified_values/value")[:50]:
        vals.append(
            {
                "schema": (v.findtext("schema") or "").strip(),
                "table": (v.findtext("table") or "").strip(),
                "column": (v.findtext("column") or "").strip(),
                "actual_value": (v.findtext("actual_value") or "").strip(),
                "user_term": (v.findtext("user_term") or "").strip(),
            }
        )
    if vals:
        out["identified_values"] = vals[:50]
    # tables
    tbs: List[Dict[str, str]] = []
    for t in root.findall(".//identified_tables/table")[:50]:
        tbs.append(
            {
                "schema": (t.findtext("schema") or "").strip(),
                "name": (t.findtext("name") or "").strip(),
            }
        )
    if tbs:
        out["identified_tables"] = tbs[:50]
    return out


# =============================================================================
# Evidence XML filtering (build_sql_context tool_result -> minimal evidence)
# =============================================================================


def _tfqn(schema: str, table: str) -> str:
    s = (schema or "").strip()
    t = (table or "").strip()
    return f"{s}.{t}".strip(".")


def _tfqn_l(schema: str, table: str) -> str:
    return _tfqn(schema, table).lower()


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


def extract_used_tables_and_columns(sql: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Returns:
    - used_tables: set of table_fqn_lower (schema.table or table)
    - used_cols: mapping table_fqn_lower -> set(column_name_lower)
    """
    s = str(sql or "")
    used_tables: Set[str] = set()

    # FROM/JOIN "SCHEMA"."TABLE"
    for m in re.finditer(
        r"\b(from|join)\s+\"(?:(?P<schema>[^\"]+)\"\.)?\"(?P<table>[^\"]+)\"",
        s,
        flags=re.IGNORECASE,
    ):
        schema = (m.group("schema") or "").strip()
        table = (m.group("table") or "").strip()
        if not table:
            continue
        used_tables.add(_tfqn_l(schema, table) if schema else table.lower())

    # Alias mapping for alias."COL"
    alias_to_table: Dict[str, str] = {}
    for m in re.finditer(
        r"\b(from|join)\s+\"(?:(?P<schema>[^\"]+)\"\.)?\"(?P<table>[^\"]+)\"\s+(?:as\s+)?(?P<alias>[A-Za-z_][A-Za-z0-9_]*)",
        s,
        flags=re.IGNORECASE,
    ):
        schema = (m.group("schema") or "").strip()
        table = (m.group("table") or "").strip()
        alias = (m.group("alias") or "").strip()
        if not table or not alias:
            continue
        alias_to_table[alias] = (_tfqn_l(schema, table) if schema else table.lower())

    used_cols: Dict[str, Set[str]] = {}
    for m in re.finditer(r"\b(?P<alias>[A-Za-z_][A-Za-z0-9_]*)\.\s*\"(?P<col>[^\"]+)\"", s):
        alias = (m.group("alias") or "").strip()
        col = (m.group("col") or "").strip()
        tf = alias_to_table.get(alias)
        if not tf or not col:
            continue
        used_cols.setdefault(tf, set()).add(col.lower())
    return used_tables, used_cols


def filter_build_sql_context_tool_result_xml(
    tool_result_xml: str, *, final_sql: str, max_values: int = 80
) -> str:
    """
    Keep only evidence useful for follow-ups:
    - schema_candidates/tables + per_table_columns filtered by used tables/cols
    - resolved_values filtered by used cols (column_fqn matches)
    - fk_relationships filtered to used tables
    - column_value_hints filtered to used tables/cols (optional)
    Drop everything else (hyde/similar/light_queries/suggestions/warnings).
    """
    root = _safe_fromstring((tool_result_xml or "").strip())
    if root is None:
        return "<tool_result><build_sql_context_result /></tool_result>"
    b = _find_build_sql_context_result(root)
    if b is None:
        return "<tool_result><build_sql_context_result /></tool_result>"

    used_tables, used_cols = extract_used_tables_and_columns(final_sql)

    out_root = ET.Element("tool_result")
    out_b = ET.SubElement(out_root, "build_sql_context_result")

    # schema_candidates
    sc = b.find("schema_candidates")
    if sc is not None:
        out_sc = ET.SubElement(out_b, "schema_candidates")

        # tables
        tables = sc.find("tables")
        if tables is not None:
            out_tables = ET.SubElement(out_sc, "tables")
            for t in list(tables.findall("table")):
                schema = (t.findtext("schema") or "").strip()
                name = (t.findtext("name") or "").strip()
                tf = _tfqn_l(schema, name) if schema else name.lower()
                if not name:
                    continue
                if used_tables and tf not in used_tables:
                    continue
                nt = ET.SubElement(out_tables, "table")
                for tag in ("schema", "name", "description", "score"):
                    v = (t.findtext(tag) or "").strip()
                    if v:
                        ET.SubElement(nt, tag).text = v

        # per_table_columns
        pt = sc.find("per_table_columns")
        if pt is not None:
            out_pt = ET.SubElement(out_sc, "per_table_columns")
            for tb in list(pt.findall("table")):
                schema = (tb.findtext("schema") or "").strip()
                name = (tb.findtext("name") or "").strip()
                tf = _tfqn_l(schema, name) if schema else name.lower()
                if not name:
                    continue
                if used_tables and tf not in used_tables:
                    continue
                cols_allowed = used_cols.get(tf, set())
                ntb = ET.SubElement(out_pt, "table")
                for tag in ("schema", "name", "mode"):
                    v = (tb.findtext(tag) or "").strip()
                    if v:
                        ET.SubElement(ntb, tag).text = v
                cols_el = ET.SubElement(ntb, "columns")
                cols_src = tb.find("columns")
                if cols_src is not None:
                    for c in list(cols_src.findall("column")):
                        cn = (c.findtext("name") or "").strip()
                        if not cn:
                            continue
                        if cols_allowed and cn.lower() not in cols_allowed:
                            continue
                        nc = ET.SubElement(cols_el, "column")
                        for tag in ("name", "dtype", "description", "score"):
                            v = (c.findtext(tag) or "").strip()
                            if v:
                                ET.SubElement(nc, tag).text = v

    # column_value_hints (optional)
    cvh = b.find("column_value_hints")
    if cvh is not None:
        out_cvh = ET.SubElement(out_b, "column_value_hints")
        for tb in list(cvh.findall("table")):
            schema = (tb.findtext("schema") or "").strip()
            name = (tb.findtext("name") or "").strip()
            tf = _tfqn_l(schema, name) if schema else name.lower()
            if not name:
                continue
            if used_tables and tf not in used_tables:
                continue
            cols_allowed = used_cols.get(tf, set())
            cols_in = tb.find("columns")
            if cols_in is None:
                continue
            ntb = ET.SubElement(out_cvh, "table")
            if schema:
                ET.SubElement(ntb, "schema").text = schema
            ET.SubElement(ntb, "name").text = name
            cols_out = ET.SubElement(ntb, "columns")
            for c in list(cols_in.findall("column")):
                cn = (c.findtext("name") or "").strip()
                if not cn:
                    continue
                if cols_allowed and cn.lower() not in cols_allowed:
                    continue
                nc = ET.SubElement(cols_out, "column")
                for tag in ("name", "dtype", "cardinality"):
                    v = (c.findtext(tag) or "").strip()
                    if v:
                        ET.SubElement(nc, tag).text = v
                vals_out = ET.SubElement(nc, "values")
                vals_in = c.find("values")
                if vals_in is None:
                    continue
                kept = 0
                for v_el in list(vals_in.findall("value")):
                    if kept >= 12:
                        break
                    txt = (v_el.findtext("text") or "").strip()
                    if not txt:
                        continue
                    nv = ET.SubElement(vals_out, "value")
                    ET.SubElement(nv, "text").text = txt
                    cnt = (v_el.findtext("count") or "").strip()
                    if cnt:
                        ET.SubElement(nv, "count").text = cnt
                    kept += 1

    # resolved_values
    rv = b.find("resolved_values")
    if rv is not None:
        out_rv = ET.SubElement(out_b, "resolved_values")
        kept = 0
        for v in list(rv.findall("value")):
            if kept >= max(1, int(max_values)):
                break
            col_fqn = (v.findtext("column_fqn") or "").strip().lower()
            if col_fqn:
                parts = [p for p in col_fqn.split(".") if p]
                if len(parts) >= 3:
                    tf = f"{parts[0]}.{parts[1]}".lower()
                    cn = parts[2].lower()
                elif len(parts) == 2:
                    tf = parts[0].lower()
                    cn = parts[1].lower()
                else:
                    tf = ""
                    cn = ""
                allowed_cols = used_cols.get(tf, set())
                if allowed_cols and cn and cn not in allowed_cols:
                    continue
                if used_tables and tf and tf not in used_tables:
                    continue
            else:
                # Without column_fqn, keep only if we have no filtering signal.
                if used_tables or used_cols:
                    continue
            nv = ET.SubElement(out_rv, "value")
            for tag in ("user_term", "actual_value", "source", "column_fqn"):
                t = (v.findtext(tag) or "").strip()
                if t:
                    ET.SubElement(nv, tag).text = t
            kept += 1

    # fk_relationships
    fk = b.find("fk_relationships")
    if fk is not None:
        out_fk = ET.SubElement(out_b, "fk_relationships")
        kept = 0
        for rel in list(fk.findall("fk")):
            if kept >= 120:
                break
            fs = (rel.findtext("from_schema") or "").strip()
            ft = (rel.findtext("from_table") or "").strip()
            ts = (rel.findtext("to_schema") or "").strip()
            tt = (rel.findtext("to_table") or "").strip()
            if ft and tt:
                l = _tfqn_l(fs, ft) if fs else ft.lower()
                r = _tfqn_l(ts, tt) if ts else tt.lower()
                if used_tables and (l not in used_tables or r not in used_tables):
                    continue
            nr = ET.SubElement(out_fk, "fk")
            for tag in (
                "from_schema",
                "from_table",
                "from_column",
                "to_schema",
                "to_table",
                "to_column",
                "constraint_name",
            ):
                vv = (rel.findtext(tag) or "").strip()
                if vv:
                    ET.SubElement(nr, tag).text = vv
            kept += 1

    return ET.tostring(out_root, encoding="unicode")


# =============================================================================
# Conversation context for candidate LLM (recent + relevant selection)
# =============================================================================


def _tokenize_for_overlap(text: str) -> Set[str]:
    toks = re.findall(r"[A-Za-z0-9가-힣_]{2,}", str(text or "").lower())
    stop = {
        "보여줘",
        "보여주세요",
        "알려줘",
        "알려주세요",
        "조회",
        "데이터",
        "정보",
        "결과",
        "내역",
        "기준",
        "기간",
        "최근",
        "전체",
        "모든",
        "같이",
        "방금",
    }
    return set([t for t in toks if t not in stop and not t.isdigit()])


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = max(1, len(a | b))
    return float(inter) / float(union)


def build_conversation_context(
    capsule: ConversationCapsule,
    *,
    followup_question: str,
    recent_n: int = 5,
    relevant_m: int = 3,
    max_preview_cols: int = 12,
    max_preview_rows: int = 3,
) -> Dict[str, Any]:
    """
    Returns a structured JSON-friendly payload:
    - mode, recent_n, relevant_m
    - selected_indices (0-based, chronological)
    - turns: list of {question, final_sql, preview, derived_filters, important_hints}
    """
    turns = list(capsule.turns or [])
    if not turns:
        return {"mode": "recent_plus_relevant", "selected_indices": [], "turns": []}

    n = max(1, int(recent_n))
    m = max(0, int(relevant_m))
    q = (followup_question or "").strip()
    q_toks = _tokenize_for_overlap(q)

    # recent indices
    recent_start = max(0, len(turns) - n)
    recent_idx = list(range(recent_start, len(turns)))
    # relevant scoring across all turns (including older than recent)
    scored: List[Tuple[float, int]] = []
    for idx, t in enumerate(turns):
        blob = "\n".join(
            [
                t.question,
                " ".join([f"{k}={v}" for k, v in (t.derived_filters or {}).items()]),
                (t.final_sql or "")[:800],
            ]
        )
        sc = _jaccard(q_toks, _tokenize_for_overlap(blob))
        scored.append((sc, idx))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    relevant_idx: List[int] = []
    if m > 0:
        for _sc, idx in scored:
            if idx in recent_idx:
                continue
            relevant_idx.append(idx)
            if len(relevant_idx) >= m:
                break

    # Merge + dedup by final_sql (keep latest occurrence)
    picked = sorted(set(recent_idx + relevant_idx))
    # Always keep latest turn
    if (len(turns) - 1) not in picked:
        picked.append(len(turns) - 1)
        picked = sorted(set(picked))

    # Dedup by final_sql: keep latest index per normalized final_sql
    latest_by_sql: Dict[str, int] = {}
    for idx in picked:
        sql_key = _ws_norm(_strip_sql_comments(turns[idx].final_sql)).lower()
        if not sql_key:
            continue
        latest_by_sql[sql_key] = max(latest_by_sql.get(sql_key, -1), idx)
    if latest_by_sql:
        keep_set = set(latest_by_sql.values())
        keep_set.add(len(turns) - 1)
        picked = sorted([i for i in picked if i in keep_set])

    turns_payload: List[Dict[str, Any]] = []
    for idx in picked:
        t = turns[idx]
        pv = t.preview.capped(max_cols=max_preview_cols, max_rows=max_preview_rows)
        turns_payload.append(
            {
                "index": int(idx),
                "question": t.question,
                "final_sql": (t.final_sql or "").strip(),
                "preview": pv.to_dict(),
                "derived_filters": dict(t.derived_filters or {}),
                "important_hints": dict(t.important_hints or {}),
            }
        )

    return {
        "mode": "recent_plus_relevant",
        "recent_n": int(n),
        "relevant_m": int(m),
        "selected_indices": picked,
        "turns": turns_payload,
    }


def append_completed_turn_to_capsule(
    capsule: ConversationCapsule,
    *,
    question: str,
    final_sql: str,
    execution_result: Optional[Dict[str, Any]],
    collected_metadata_xml: str,
    build_sql_context_xml: str,
    max_turns: int = 40,
    preview_max_cols: int = 12,
    preview_max_rows: int = 5,
) -> ConversationCapsule:
    now = int(time.time() * 1000)
    base = capsule if isinstance(capsule, ConversationCapsule) else ConversationCapsule()
    if not base.created_at_ms:
        base.created_at_ms = now
    base.updated_at_ms = now

    sql_clean = _strip_sql_comments(str(final_sql or ""))
    preview = TurnPreview.from_execution_result(execution_result).capped(
        max_cols=int(preview_max_cols), max_rows=int(preview_max_rows)
    )
    derived = extract_derived_filters_from_sql(sql_clean, max_items=16)
    hints = _parse_collected_metadata_xml(collected_metadata_xml)
    evidence = ""
    if (build_sql_context_xml or "").strip() and sql_clean:
        try:
            evidence = filter_build_sql_context_tool_result_xml(
                build_sql_context_xml, final_sql=sql_clean, max_values=80
            )
        except Exception:
            evidence = ""

    turn = TurnCapsule(
        question=_ws_norm(str(question or "")),
        final_sql=sql_clean,
        preview=preview,
        derived_filters=derived,
        important_hints=hints,
        evidence_ctx_xml=evidence,
        created_at_ms=now,
    )

    base.turns.append(turn)
    # Cap total turns
    cap = max(1, int(max_turns))
    if len(base.turns) > cap:
        base.turns = base.turns[-cap:]
    return base


__all__ = [
    "ConversationCapsule",
    "TurnCapsule",
    "TurnPreview",
    "new_capsule",
    "encode_conversation_state",
    "decode_conversation_state",
    "build_conversation_context",
    "append_completed_turn_to_capsule",
    "filter_build_sql_context_tool_result_xml",
]

