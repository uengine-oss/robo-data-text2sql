from __future__ import annotations

import json
import re
import traceback
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.react.generators.controller_repair_generator import get_controller_repair_generator
from app.react.generators.controller_sql_candidates_generator import (
    get_controller_sql_candidates_generator,
)
from app.react.generators.controller_triage_generator import get_controller_triage_generator
from app.react.rubric_judge import (
    compute_score_and_accept,
    create_rubric_llm,
    evaluate_candidate,
    extract_context_evidence,
    extract_requirements,
)
from app.react.tools import ToolContext, execute_tool
from app.react.utils.log_sanitize import sanitize_for_log
from app.smart_logger import SmartLogger

_SQL_LINE_COMMENT_RE = re.compile(r"--.*?$", flags=re.MULTILINE)
_SQL_BLOCK_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/", flags=re.MULTILINE)

# Heuristics for slot-presence detection (used only for ask_user dedupe guidance).
_RE_DATE_YYYYMMDD = re.compile(r"\b(19|20)\d{2}[01]\d[0-3]\d\b")
_RE_DATE_YYYY_MM_DD = re.compile(r"\b(19|20)\d{2}[-/.][01]?\d[-/.][0-3]?\d\b")
_RE_RELATIVE_PERIOD = re.compile(
    r"(최근\s*\d+\s*(일|주|개월|달)|지난\s*\d+\s*(일|주|개월|달)|최근\s*(일주일|한달|한\s*달|한\s*개월)|"
    r"이번\s*(주|달|월)|전\s*(주|달|월)|전기간|전체기간|전체|모든)",
    flags=re.IGNORECASE,
)
_RE_AGG = re.compile(r"(평균|합계|총합|건수|개수|카운트|avg|average|sum|count|max|min)", flags=re.IGNORECASE)
_RE_GRAIN = re.compile(r"(일별|일일|daily|월별|monthly|시간별|hourly)", flags=re.IGNORECASE)

_RE_TECHNICAL_USER_ASK = re.compile(
    # Technical / DB-internal terms that end-users should never be asked for.
    # Keep this generic (domain-agnostic) and prefer broad patterns over domain-specific tokens.
    r"(\b(table|column|schema|sql)\b|테이블|컬럼|스키마|SQL|태그|시리얼|serial|\btag\w*\b|\b[A-Z]{2,}_[A-Z0-9]{2,}\b|\b\w+_(CODE|ID|SN)\b)",
    flags=re.IGNORECASE,
)

def _emit_controller_stage(
    context: ToolContext,
    *,
    stage: str,
    status: str,
    seq: int,
    elapsed_ms: Optional[float] = None,
    counts: Optional[Dict[str, int]] = None,
    notes: Optional[List[str]] = None,
    error: Optional[str] = None,
) -> None:
    """
    Best-effort: emit controller progress event to the router's NDJSON stream.
    Must never raise.
    """
    try:
        ts_ms = int(time.time() * 1000)
        payload: Dict[str, Any] = {
            "event": "pipeline_stage",
            "pipeline": "controller",
            "stage": str(stage or ""),
            "status": str(status or ""),
            "seq": int(seq),
            "iteration": 0,
            "ts_ms": ts_ms,
        }
        if elapsed_ms is not None:
            payload["elapsed_ms"] = float(elapsed_ms)
        if counts:
            payload["counts"] = {str(k): int(v) for k, v in dict(counts).items() if isinstance(v, int)}
        if notes:
            payload["notes"] = [str(x) for x in list(notes) if str(x).strip()][:12]
        if error:
            payload["error"] = str(error)[:500]
        context.emit(payload)
    except Exception:
        return


def _emit_controller_item(
    context: ToolContext,
    *,
    stage: str,
    item_type: str,
    index: int,
    total: int,
    verdict: Optional[str] = None,
    row_count: Optional[int] = None,
    elapsed_ms: Optional[float] = None,
) -> None:
    """
    Best-effort: emit controller item-level progress event (e.g., candidate i/N).
    Must never raise.
    """
    try:
        ts_ms = int(time.time() * 1000)
        payload: Dict[str, Any] = {
            "event": "pipeline_item",
            "pipeline": "controller",
            "stage": str(stage or ""),
            "item_type": str(item_type or ""),
            "iteration": 0,
            "index": int(index),
            "total": int(total),
            "ts_ms": ts_ms,
        }
        if verdict is not None:
            payload["verdict"] = str(verdict)
        if row_count is not None:
            payload["row_count"] = int(row_count)
        if elapsed_ms is not None:
            payload["elapsed_ms"] = float(elapsed_ms)
        context.emit(payload)
    except Exception:
        return


def sanitize_sql(sql: str) -> str:
    """
    Remove comments and trailing semicolons to avoid SQLGuard 'dangerous pattern' false negatives.
    """
    s = (sql or "").strip()
    if not s:
        return ""
    s = _SQL_BLOCK_COMMENT_RE.sub("", s)
    s = _SQL_LINE_COMMENT_RE.sub("", s)
    s = s.strip().rstrip(";").strip()
    return s


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return None
    cand = s[start : end + 1]
    try:
        obj = json.loads(cand)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _looks_like_technical_user_request(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return bool(_RE_TECHNICAL_USER_ASK.search(s))


def _extract_key_terms_for_enrichment(text: str, *, limit: int = 6) -> List[str]:
    """
    Extract a few generic high-signal tokens from the user's question for enrichment.
    Domain-agnostic: no special-casing for specific facilities/metrics.
    """
    s = (text or "").strip()
    if not s:
        return []
    raw = re.findall(r"[A-Za-z0-9가-힣]{2,}", s)
    stop = {
        # Common request words (Korean)
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
        # English
        "show",
        "list",
        "query",
        "data",
        "info",
        "result",
    }
    out: List[str] = []
    seen = set()
    for t in raw:
        tt = t.strip()
        if not tt:
            continue
        if tt in stop:
            continue
        if tt.isdigit():
            continue
        k = tt.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(tt)
        if len(out) >= max(1, int(limit)):
            break
    return out


def _default_triage_enrichment_queries(*, user_question: str) -> List[str]:
    """
    Deterministic fallback enrichment when we want context_refresh but the LLM did not provide any.
    IMPORTANT: These are NOT shown to the end-user; they are internal hints for build_sql_context.
    Keep them business-oriented (avoid asking the user for DB artifacts).
    """
    q = (user_question or "").strip()
    if not q:
        return []
    terms = _extract_key_terms_for_enrichment(q, limit=6)
    base = " ".join(terms) if terms else q

    # Use only generic hints (no domain-specific entity assumptions).
    parts: List[str] = [
        f"{base} 관련 명칭/동의어/표현을 확장해 문맥을 보강",
        f"{base} 관련 대상(엔티티) 후보를 추가로 탐색",
        f"{base} 관련 측정 항목(지표) 정의/단위를 추가로 탐색",
        f"{base} 관련 시간 기준(일/월/시간)과 집계 힌트를 추가로 탐색",
    ]
    # Dedup + cap
    uniq: List[str] = []
    seen = set()
    for p in parts:
        s = (p or "").strip()
        if not s:
            continue
        k = s.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(s[:200])
    return uniq[:4]


def _compute_build_sql_context_stats(tool_result_xml: str) -> Dict[str, Any]:
    """
    Best-effort stats from build_sql_context tool_result XML.
    Used only as an LLM input signal (triage); must be robust and never raise.
    """
    text = (tool_result_xml or "").strip()
    stats: Dict[str, Any] = {
        "has_build_sql_context": bool(text and "<build_sql_context_result" in text),
        "candidate_tables_count": 0,
        "per_table_columns_table_count": 0,
        "per_table_columns_total_count": 0,
        "resolved_values_count": 0,
        "value_hints_columns_count": 0,
        "value_hints_values_count": 0,
        "light_queries_total": 0,
        "light_queries_pass": 0,
        "light_queries_pass_with_rows": 0,
    }
    if not text:
        return stats
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # Fallback: cheap substring heuristics (very rough).
        stats["candidate_tables_count"] = int(text.count("<schema_candidates>") > 0)
        stats["resolved_values_count"] = text.count("<resolved_values>")  # 0/1 typically
        stats["light_queries_total"] = text.count("<query ")
        stats["light_queries_pass"] = text.count("<verdict>PASS</verdict>")
        return stats
    b = root.find(".//build_sql_context_result")
    if b is None:
        b = root
    try:
        stats["candidate_tables_count"] = len(b.findall(".//schema_candidates/tables/table"))
    except Exception:
        pass
    try:
        tbs = b.findall(".//schema_candidates/per_table_columns/table")
        stats["per_table_columns_table_count"] = len(tbs)
        total_cols = 0
        for tb in tbs:
            total_cols += len(tb.findall(".//columns/column"))
        stats["per_table_columns_total_count"] = int(total_cols)
    except Exception:
        pass
    try:
        stats["resolved_values_count"] = len(b.findall(".//resolved_values/value"))
    except Exception:
        pass
    try:
        cols = b.findall(".//column_value_hints/table/columns/column")
        stats["value_hints_columns_count"] = len(cols)
        vcount = 0
        for c in cols:
            vcount += len(c.findall(".//values/value"))
        stats["value_hints_values_count"] = int(vcount)
    except Exception:
        pass
    try:
        qs = b.findall(".//light_queries/query")
        stats["light_queries_total"] = len(qs)
        pass_cnt = 0
        pass_rows = 0
        for q in qs:
            verdict = (q.findtext("verdict") or "").strip().upper()
            if verdict == "PASS":
                pass_cnt += 1
                try:
                    rc = int((q.findtext(".//preview/row_count") or "0").strip() or "0")
                except Exception:
                    rc = 0
                if rc > 0:
                    pass_rows += 1
        stats["light_queries_pass"] = int(pass_cnt)
        stats["light_queries_pass_with_rows"] = int(pass_rows)
    except Exception:
        pass
    return stats


def parse_validate_sql(tool_result_xml: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "verdict": "",
        "selected_sql": "",
        "fail_reason": "",
        "preview": {},
        "suggested_fixes": [],
        "auto_rewrite": {},
    }
    text = (tool_result_xml or "").strip()
    if not text:
        return out
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return out
    node = root.find(".//validate_sql_result")
    if node is None:
        return out
    out["verdict"] = (node.findtext("verdict") or "").strip().upper()
    out["fail_reason"] = (node.findtext("fail_reason") or "").strip()
    out["selected_sql"] = (node.findtext("selected_sql") or "").strip()

    # Optional: suggested fixes on FAIL (or best-effort on other stages)
    try:
        fixes: List[str] = []
        fixes_el = node.find("suggested_fixes")
        if fixes_el is not None:
            for fx in fixes_el.findall("./fix"):
                t = (fx.text or "").strip()
                if t:
                    fixes.append(t[:400])
        out["suggested_fixes"] = fixes[:8]
    except Exception:
        out["suggested_fixes"] = []

    # Optional: auto rewrite details
    try:
        ar_el = node.find("auto_rewrite")
        if ar_el is not None:
            ar: Dict[str, Any] = {}
            ar["applied"] = ((ar_el.findtext("applied") or "").strip().lower() == "true")
            ar["source"] = (ar_el.findtext("source") or "").strip()[:80]
            reasons: List[str] = []
            for r in ar_el.findall("./reason"):
                t = (r.text or "").strip()
                if t:
                    reasons.append(t[:240])
            if reasons:
                ar["reasons"] = reasons[:8]
            out["auto_rewrite"] = ar
    except Exception:
        out["auto_rewrite"] = {}

    preview_el = node.find("preview")
    if preview_el is None:
        return out
    cols = [c.text or "" for c in preview_el.findall("./columns/column")]
    cols = [c.strip() for c in cols if c and c.strip()]
    rows_out: List[List[Any]] = []
    for row_el in preview_el.findall("./rows/row"):
        cells = row_el.findall("./cell")
        if not cells:
            continue
        row_vals: List[Any] = []
        for cell_el in cells:
            cell_text = "".join(cell_el.itertext()).strip()
            row_vals.append(cell_text if cell_text != "" else None)
        rows_out.append(row_vals)
    try:
        row_count = int((preview_el.findtext("row_count") or "0").strip() or "0")
    except Exception:
        row_count = 0
    try:
        exec_ms = float((preview_el.findtext("preview_execution_time_ms") or "0").strip() or "0")
    except Exception:
        exec_ms = 0.0
    out["preview"] = {
        "columns": cols,
        "rows": rows_out,
        "row_count": row_count,
        "execution_time_ms": exec_ms,
    }
    return out


def _norm_sql_key(sql: str) -> str:
    """Collapse whitespace + lowercase for weak similarity checks."""
    return " ".join(str(sql or "").split()).strip().lower()


def _same_fail_set(a: Sequence[str], b: Sequence[str]) -> bool:
    sa = set([str(x or "").strip() for x in (a or []) if str(x or "").strip()])
    sb = set([str(x or "").strip() for x in (b or []) if str(x or "").strip()])
    return sa == sb


def _failed_checks_payload(
    *,
    requirements: Sequence[Any],
    checks: Sequence[Any],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Build structured failed-checks payload for repair prompts.

    Returns:
    - failed_checks: list of {id,must,type,text,status,why}
    - passed_must_ids: list of MUST requirement ids where status==PASS
    """
    reqs = list(requirements or [])
    chks = list(checks or [])
    check_map: Dict[str, Any] = {}
    for c in chks:
        cid = str(getattr(c, "id", "") or "").strip()
        if cid and cid not in check_map:
            check_map[cid] = c
    failed: List[Dict[str, Any]] = []
    passed_must: List[str] = []
    for r in reqs:
        rid = str(getattr(r, "id", "") or "").strip()
        if not rid:
            continue
        must = bool(getattr(r, "must", True))
        typ = str(getattr(r, "type", "") or "").strip()
        txt = str(getattr(r, "text", "") or "").strip()
        c = check_map.get(rid)
        st = str(getattr(c, "status", "") or "UNKNOWN").strip().upper()
        if st not in ("PASS", "FAIL", "UNKNOWN"):
            st = "UNKNOWN"
        why = str(getattr(c, "why", "") or "").strip()
        if must and st == "PASS":
            passed_must.append(rid)
        if st != "PASS":
            failed.append(
                {
                    "id": rid[:24],
                    "must": bool(must),
                    "type": typ[:24],
                    "text": txt[:220],
                    "status": st,
                    "why": why[:140],
                }
            )
    return failed[:32], passed_must[:32]


def _has_latest_point_in_time_filter(sql: str) -> bool:
    """
    Heuristic: does SQL clearly target a *single latest point in time* (e.g., latest day)?

    This matters because user phrases like "가장 최근 일별 ..." often mean:
    - "daily table" + "latest day record(s)" (single day), NOT necessarily a multi-day time series.
    """
    s = (sql or "").strip()
    if not s:
        return False
    sl = s.lower()
    if "log_time" not in sl and "date" not in sl and "dt" not in sl:
        return False

    # Pattern A: ORDER BY <time> DESC LIMIT 1 / FETCH FIRST 1
    if re.search(r"\border\s+by\b[\s\S]{0,250}\b(log_time|date|dt)\b[\s\S]{0,80}\bdesc\b", sl, flags=re.IGNORECASE):
        if re.search(r"\blimit\s+1\b", sl, flags=re.IGNORECASE) or re.search(
            r"\bfetch\s+first\s+1\b", sl, flags=re.IGNORECASE
        ):
            return True

    # Pattern B: WHERE <time> = (SELECT MAX(<time>) ...)
    if re.search(
        r"\bwhere\b[\s\S]{0,800}\b(log_time|date|dt)\b[\s\S]{0,80}=\s*\(\s*select\s+max\(\s*\"?(log_time|date|dt)\"?\s*\)",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True

    # Pattern C: WHERE <time> IN (SELECT MAX(<time>) ...)
    if re.search(
        r"\bwhere\b[\s\S]{0,800}\b(log_time|date|dt)\b[\s\S]{0,80}\bin\s*\(\s*select\s+max\(\s*\"?(log_time|date|dt)\"?\s*\)",
        s,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        return True

    return False


def semantic_mismatch_reasons(*, user_query: str, sql: str) -> List[str]:
    uq = (user_query or "").strip().lower()
    s = (sql or "").strip().lower()
    if not uq or not s:
        return []
    reasons: List[str] = []
    if any(k in uq for k in ("평균", "average", "avg")) and "avg(" not in s:
        reasons.append("missing AVG()")
    if any(k in uq for k in ("합계", "총합", "sum", "total")) and "sum(" not in s:
        reasons.append("missing SUM()")
    if any(k in uq for k in ("건수", "개수", "count")) and "count(" not in s:
        reasons.append("missing COUNT()")
    if any(k in uq for k in ("최대", "max")) and "max(" not in s:
        reasons.append("missing MAX()")
    if any(k in uq for k in ("최소", "min")) and "min(" not in s:
        reasons.append("missing MIN()")
    # Grain / grouping
    if ("일일" in uq) or ("일별" in uq) or ("daily" in uq):
        # "일별"이 항상 time-series(group by) 요구사항을 의미하지는 않는다.
        # 특히 "가장 최근"과 같이 최신 1일(1시점)만 조회하는 질의는 GROUP BY 없이도 정답일 수 있다.
        if re.search(r"\bgroup\s+by\b", s, flags=re.IGNORECASE) is None and not _has_latest_point_in_time_filter(sql):
            reasons.append("missing GROUP BY for daily grain")
    return reasons


def _preview_non_null_stats(validate_preview: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    """
    Returns (cells_total, cells_non_null) from validate_sql preview.
    We treat None/""/"null" as null-ish.
    """
    if not isinstance(validate_preview, dict):
        return (0, 0)
    rows = validate_preview.get("rows")
    if not isinstance(rows, list):
        return (0, 0)
    total = 0
    non_null = 0
    for r in rows:
        if not isinstance(r, list):
            continue
        for c in r:
            total += 1
            if c is None:
                continue
            s = str(c).strip()
            if not s:
                continue
            if s.lower() == "null":
                continue
            non_null += 1
    return (total, non_null)


def _extract_value_hints(tool_result_xml: str) -> List[Dict[str, Any]]:
    """
    Extract small-cardinality value hints from <column_value_hints>.
    Regex-based extraction to tolerate occasionally non-XML-safe text elsewhere.
    """
    text = (tool_result_xml or "")
    if not text.strip():
        return []
    start = text.find("<column_value_hints>")
    end = text.find("</column_value_hints>")
    if start < 0 or end < 0 or end <= start:
        return []
    section = text[start : end + len("</column_value_hints>")]

    out: List[Dict[str, Any]] = []
    for tb in re.findall(r"<table>[\s\S]*?</table>", section):
        m_schema = re.search(r"<schema>([\s\S]*?)</schema>", tb)
        m_name = re.search(r"<name>([\s\S]*?)</name>", tb)
        schema = (m_schema.group(1) if m_schema else "").strip()
        name = (m_name.group(1) if m_name else "").strip()
        if not name:
            continue

        cols_section_m = re.search(r"<columns>([\s\S]*?)</columns>", tb)
        cols_section = cols_section_m.group(1) if cols_section_m else ""
        for cb in re.findall(r"<column>[\s\S]*?</column>", cols_section):
            m_cn = re.search(r"<name>([\s\S]*?)</name>", cb)
            cn = (m_cn.group(1) if m_cn else "").strip()
            if not cn:
                continue
            m_dtype = re.search(r"<dtype>([\s\S]*?)</dtype>", cb)
            dtype = (m_dtype.group(1) if m_dtype else "").strip()
            card = None
            try:
                card = int((re.search(r"<cardinality>([\s\S]*?)</cardinality>", cb).group(1) or "").strip())  # type: ignore[union-attr]
            except Exception:
                card = None
            vals: List[str] = []
            vals_section_m = re.search(r"<values>([\s\S]*?)</values>", cb)
            vals_section = vals_section_m.group(1) if vals_section_m else ""
            for vb in re.findall(r"<value>[\s\S]*?</value>", vals_section):
                m_txt = re.search(r"<text>([\s\S]*?)</text>", vb)
                v = (m_txt.group(1) if m_txt else "").strip()
                if v:
                    vals.append(v)
            out.append(
                {
                    "schema": schema,
                    "table": name,
                    "column": cn,
                    "dtype": dtype,
                    "cardinality": card,
                    "values": vals,
                }
            )
    return out


def _build_value_hints_map(tool_result_xml: str, *, max_columns: int = 30, max_values_per_col: int = 20) -> Dict[str, set]:
    """
    Build a loose mapping: column_name(lower) -> set(values_lower).
    This is used as a *general* sanity-check: if SQL uses col='literal' where col has
    enumerated hints and 'literal' is not among them, treat as suspicious.
    """
    hints = _extract_value_hints(tool_result_xml)
    out: Dict[str, set] = {}
    for h in hints:
        cn = str(h.get("column") or "").strip()
        if not cn:
            continue
        card = h.get("cardinality")
        # Prefer truly small-cardinality columns; still allow unknown card if values are few.
        if isinstance(card, int) and card > 200:
            continue
        vals = h.get("values")
        if not isinstance(vals, list) or not vals:
            continue
        key = cn.lower()
        s = out.setdefault(key, set())
        for v in vals[: max(1, int(max_values_per_col))]:
            vv = str(v or "").strip().lower()
            if vv:
                s.add(vv)
        if len(out) >= max(1, int(max_columns)):
            break
    return out


def score_sql(
    *,
    user_query: str,
    sql: str,
    validate_preview: Optional[Dict[str, Any]] = None,
    value_hints: Optional[Dict[str, set]] = None,
) -> float:
    """
    Deterministic score in [0, 1] using only:
    - question text (intent/grain keywords)
    - SQL text structure
    - validate preview signals (row_count)
    """
    uq = (user_query or "").strip().lower()
    s = (sql or "").strip().lower()
    if not uq or not s:
        return 0.0

    score = 0.0

    if re.search(r"\bselect\s+\*\b", s, flags=re.IGNORECASE):
        score -= 0.35
    if re.search(r"\bfrom\b", s, flags=re.IGNORECASE) is None:
        score -= 0.6

    miss = semantic_mismatch_reasons(user_query=user_query, sql=sql)
    if not miss:
        score += 0.55
    else:
        if any(x in s for x in ("avg(", "sum(", "count(", "max(", "min(")):
            score += 0.15

    if ("일일" in uq) or ("일별" in uq) or ("daily" in uq):
        if re.search(r"\bgroup\s+by\b", s, flags=re.IGNORECASE) and any(x in s for x in ("log_time", "date", "dt", "day")):
            score += 0.20
        elif _has_latest_point_in_time_filter(sql):
            # Allow "latest daily value" queries without GROUP BY.
            score += 0.08
        else:
            # Penalize "daily" questions answered without grouping.
            score -= 0.25
    if re.search(r"\blimit\b", s, flags=re.IGNORECASE):
        score += 0.05

    # If the user explicitly asks to include units, mildly reward selecting a unit/description field.
    try:
        if ("단위" in uq) or (re.search(r"\bunit\b", uq, flags=re.IGNORECASE) is not None):
            if (
                re.search(r"\bunit_desc\b", s, flags=re.IGNORECASE) is not None
                or re.search(r"\btag_unit\b", s, flags=re.IGNORECASE) is not None
            ):
                score += 0.04
    except Exception:
        pass

    # User-friendly output heuristic:
    # If the SQL selects a code/id-like identifier (e.g., *_CODE, *_ID), prefer including a name/title-like
    # column too (e.g., *_NAME, *_NM) when available. This is a mild preference only.
    try:
        _re_name = re.compile(r"(^|_)(name|nm|title)($|_)", re.IGNORECASE)
        _re_code = re.compile(r"(^|_)(code|cd|id|sn)($|_)", re.IGNORECASE)
        cols = [m.group(1) for m in re.finditer(r'\.\s*\"([^\"]+)\"', sql)]
        cols_l = [(c or "").strip().lower() for c in cols if c and str(c).strip()]
        has_name = any(_re_name.search(c) is not None for c in cols_l)
        has_code = any(_re_code.search(c) is not None for c in cols_l)
        if has_name and has_code:
            score += 0.06
        elif has_code and not has_name:
            score -= 0.06
        elif has_name and not has_code:
            score -= 0.03
    except Exception:
        pass

    try:
        rc = None
        if isinstance(validate_preview, dict):
            rc = validate_preview.get("row_count")
        cells_total, cells_non_null = _preview_non_null_stats(validate_preview)
        if isinstance(rc, int) and rc <= 0:
            score -= 0.30
        # Strong penalty for "PASS but empty/all-null" preview, which is usually a wrong filter.
        # Example: aggregate over empty input returns 1 row with NULL.
        if cells_total > 0 and cells_non_null == 0:
            score -= 0.60
        # Only reward preview if it contains some non-null signal.
        if isinstance(rc, int) and rc > 0 and cells_non_null > 0:
            score += 0.20
    except Exception:
        pass

    # Enum/value-hints sanity check: col='literal' where literal is not among hinted values.
    if isinstance(value_hints, dict) and value_hints:
        bad_eq = 0
        for m in re.finditer(r"\.\s*\"(?P<col>[^\"]+)\"\s*=\s*'(?P<val>[^']*)'", sql):
            col = (m.group("col") or "").strip().lower()
            val = (m.group("val") or "").strip().lower()
            if not col or not val:
                continue
            allowed = value_hints.get(col)
            if isinstance(allowed, set) and allowed and val not in allowed:
                bad_eq += 1
        if bad_eq > 0:
            score -= min(0.45, 0.25 + 0.10 * float(bad_eq))

    return max(0.0, min(1.0, float(score)))


def _extract_per_table_columns(tool_result_xml: str) -> List[Dict[str, Any]]:
    """
    Regex-based extraction to tolerate occasionally non-XML-safe text in other parts of tool_result.
    We only need schema/name and column names under <per_table_columns>.
    """
    text = (tool_result_xml or "")
    if not text.strip():
        return []
    start = text.find("<per_table_columns>")
    end = text.find("</per_table_columns>")
    if start < 0 or end < 0 or end <= start:
        return []
    section = text[start : end + len("</per_table_columns>")]
    out: List[Dict[str, Any]] = []
    for tb in re.findall(r"<table>[\s\S]*?</table>", section):
        m_schema = re.search(r"<schema>([\s\S]*?)</schema>", tb)
        m_name = re.search(r"<name>([\s\S]*?)</name>", tb)
        schema = (m_schema.group(1) if m_schema else "").strip()
        name = (m_name.group(1) if m_name else "").strip()
        if not name:
            continue
        cols: List[Dict[str, str]] = []
        cols_section_m = re.search(r"<columns>([\s\S]*?)</columns>", tb)
        cols_section = cols_section_m.group(1) if cols_section_m else ""
        for cb in re.findall(r"<column>[\s\S]*?</column>", cols_section):
            m_cn = re.search(r"<name>([\s\S]*?)</name>", cb)
            cn = (m_cn.group(1) if m_cn else "").strip()
            if not cn:
                continue
            m_dt = re.search(r"<dtype>([\s\S]*?)</dtype>", cb)
            dt = (m_dt.group(1) if m_dt else "").strip()
            m_desc = re.search(r"<description>([\s\S]*?)</description>", cb)
            desc = (m_desc.group(1) if m_desc else "").strip()
            cols.append({"name": cn, "dtype": dt, "description": desc})
        out.append({"schema": schema, "name": name, "cols": cols})
    return out


def _extract_resolved_values(tool_result_xml: str) -> List[Dict[str, str]]:
    """
    Regex-based extraction to tolerate occasionally non-XML-safe text in other parts of tool_result.
    """
    text = (tool_result_xml or "")
    if not text.strip():
        return []
    start = text.find("<resolved_values>")
    end = text.find("</resolved_values>")
    if start < 0 or end < 0 or end <= start:
        return []
    section = text[start : end + len("</resolved_values>")]
    out: List[Dict[str, str]] = []
    for vb in re.findall(r"<value>[\s\S]*?</value>", section):
        def _get(tag: str) -> str:
            m = re.search(rf"<{tag}>([\s\S]*?)</{tag}>", vb)
            return (m.group(1) if m else "").strip()
        out.append(
            {
                "schema": _get("schema"),
                "table": _get("table"),
                "column": _get("column"),
                "user_term": _get("user_term"),
                "actual_value": _get("actual_value"),
            }
        )
    return out


async def draft_llm_candidates(
    *,
    question: str,
    dbms: str,
    max_sql_seconds: int,
    compact_context: str,
    conversation_context: Optional[Dict[str, Any]] = None,
    n_candidates: int,
    temperature: Optional[float] = None,
    diversity_hints: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> List[str]:
    gen = get_controller_sql_candidates_generator()
    cands, _mode = await gen.generate(
        question=question,
        dbms=dbms,
        max_sql_seconds=max_sql_seconds,
        context_xml=compact_context,
        conversation_context=conversation_context,
        n_candidates=int(n_candidates),
        temperature=temperature,
        diversity_hints=diversity_hints,
        seed=seed,
        react_run_id=None,
    )
    return list(cands or [])[: max(1, int(n_candidates))]


def compact_build_sql_context(tool_result_xml: str, *, max_tables: int = 12, max_cols: int = 20) -> str:
    """
    Legacy: historically compacted build_sql_context for prompt size.

    New behavior (A-1): return the original tool_result XML as-is so the controller can use
    the full context (including <schema_candidates>, <column_value_hints>, <fk_relationships>,
    and <light_queries> previews).
    """
    _ = max_tables
    _ = max_cols
    return (tool_result_xml or "").strip()


@dataclass
class ControllerConfig:
    n_candidates: int = 2
    score_threshold: float = 0.75
    allow_one_revision: bool = True
    # C-pipeline (hybrid): explore a few candidates then converge with repair loop.
    initial_candidates: int = 3
    max_repairs_per_candidate: int = 4
    stall_rounds: int = 2
    candidate_switch_enabled: bool = True
    # Generation temperatures (generators may choose to honor these).
    candidate_temperature: float = 0.30
    repair_temperature: float = 0.00


@dataclass
class ControllerAttempt:
    candidate_index: int
    sql: str
    verdict: str
    score: float
    fail_reason: str = ""
    preview: Dict[str, Any] = field(default_factory=dict)
    tool_result_xml: str = ""
    # Optional metadata for debugging/repair loops (backward compatible for consumers).
    phase: str = ""  # initial|repair|revision
    base_candidate: int = 0
    repair_round: int = 0
    hard_reject: bool = False
    hard_reason: str = ""
    rubric_fail_must_ids: List[str] = field(default_factory=list)
    rubric_missing_must: List[str] = field(default_factory=list)
    rubric_failed_checks: List[Dict[str, Any]] = field(default_factory=list)
    validate_suggested_fixes: List[str] = field(default_factory=list)
    validate_auto_rewrite: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ControllerResult:
    status: str  # submit_sql|ask_user|error
    final_sql: str = ""
    validated_sql: str = ""
    preview: Dict[str, Any] = field(default_factory=dict)
    attempts: List[ControllerAttempt] = field(default_factory=list)
    question_to_user: str = ""
    build_sql_context_xml: str = ""


@dataclass
class ControllerTriageResult:
    """
    Decision returned by a single triage LLM call after the controller fails to find an acceptable SQL.

    decision:
    - ask_user: ask the user a clarifying question (should avoid re-asking slots already present).
    - context_refresh: generate enrichment queries to re-run build_sql_context and retry the controller.
    - give_best_effort: only allowed when tool-call budget is exhausted; reuse best PASS attempt even if score<threshold.
    """

    decision: str  # ask_user|context_refresh|give_best_effort
    why: List[str] = field(default_factory=list)
    ask_user_question: str = ""
    enrichment_queries: List[str] = field(default_factory=list)


def _slots_present(text: str) -> Dict[str, bool]:
    s = (text or "").strip()
    if not s:
        return {"period": False, "aggregation": False, "grain": False}
    has_period = bool(_RE_DATE_YYYYMMDD.search(s) or _RE_DATE_YYYY_MM_DD.search(s) or _RE_RELATIVE_PERIOD.search(s))
    has_agg = bool(_RE_AGG.search(s))
    has_grain = bool(_RE_GRAIN.search(s))
    return {"period": has_period, "aggregation": has_agg, "grain": has_grain}


def _extract_ask_user_suggestions(tool_result_xml: str, *, max_items: int = 6) -> List[str]:
    text = (tool_result_xml or "")
    if not text.strip():
        return []
    start = text.find("<ask_user_suggestions>")
    end = text.find("</ask_user_suggestions>")
    if start < 0 or end < 0 or end <= start:
        return []
    section = text[start : end + len("</ask_user_suggestions>")]
    out: List[str] = []
    for m in re.finditer(r"<suggestion>([\s\S]*?)</suggestion>", section):
        v = (m.group(1) or "").strip()
        if not v:
            continue
        out.append(v)
        if len(out) >= max(1, int(max_items)):
            break
    return out


def _summarize_attempts_for_triage(attempts: Sequence[ControllerAttempt], *, max_fail_reasons: int = 3) -> Dict[str, Any]:
    total = len(list(attempts or []))
    pass_attempts = [a for a in list(attempts or []) if str(getattr(a, "verdict", "") or "").upper() == "PASS"]
    fail_attempts = [a for a in list(attempts or []) if str(getattr(a, "verdict", "") or "").upper() != "PASS"]
    best_pass = None
    if pass_attempts:
        best_pass = sorted(pass_attempts, key=lambda x: float(getattr(x, "score", 0.0) or 0.0), reverse=True)[0]
    # Top FAIL reasons (counted)
    reason_counts: Dict[str, int] = {}
    for a in fail_attempts:
        r = str(getattr(a, "fail_reason", "") or "").strip()
        if not r:
            continue
        key = r[:240]
        reason_counts[key] = int(reason_counts.get(key, 0)) + 1
    top_fail_reasons = [k for k, _v in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))][: max(1, int(max_fail_reasons))]

    best_pass_preview = getattr(best_pass, "preview", {}) if best_pass is not None else {}
    row_count = None
    try:
        if isinstance(best_pass_preview, dict):
            rc = best_pass_preview.get("row_count")
            if rc is not None:
                row_count = int(rc)
    except Exception:
        row_count = None

    return {
        "total": int(total),
        "pass_count": int(len(pass_attempts)),
        "fail_count": int(len(fail_attempts)),
        "best_pass_score": float(getattr(best_pass, "score", 0.0) or 0.0) if best_pass is not None else 0.0,
        "best_pass_row_count": row_count,
        "top_fail_reasons": top_fail_reasons,
    }


def _parse_triage_json(obj_or_text: Any) -> Optional[ControllerTriageResult]:
    obj: Optional[Dict[str, Any]]
    if isinstance(obj_or_text, dict):
        obj = obj_or_text
    else:
        obj = _extract_first_json(str(obj_or_text or ""))
    if not obj:
        return None
    decision = str(obj.get("decision") or "").strip()
    if not decision:
        return None
    why_raw = obj.get("why")
    why: List[str] = []
    if isinstance(why_raw, list):
        for x in why_raw:
            s = str(x or "").strip()
            if s:
                why.append(s[:200])
    elif isinstance(why_raw, str):
        s = why_raw.strip()
        if s:
            why.append(s[:200])
    ask_user_question = str(obj.get("ask_user_question") or "").strip()
    eq_raw = obj.get("enrichment_queries")
    enrichment_queries: List[str] = []
    if isinstance(eq_raw, list):
        for x in eq_raw:
            s = str(x or "").strip()
            if s:
                enrichment_queries.append(s[:800])
    elif isinstance(eq_raw, str):
        s = eq_raw.strip()
        if s:
            enrichment_queries.append(s[:800])
    # Normalize decision
    d = decision.strip().lower()
    if d not in {"ask_user", "context_refresh", "give_best_effort"}:
        return None
    return ControllerTriageResult(
        decision=d,
        why=why[:8],
        ask_user_question=ask_user_question[:600],
        enrichment_queries=enrichment_queries[:8],
    )


async def triage_no_acceptable_sql(
    *,
    user_question: str,
    dbms: str,
    remaining_tool_calls: int,
    attempts: Sequence[ControllerAttempt],
    build_sql_context_xml: str,
    allow_give_best_effort: bool,
) -> ControllerTriageResult:
    """
    Single LLM call that decides what to do when the controller fails to find an acceptable SQL.
    This function must be called at most once per request.
    """
    q = (user_question or "").strip()
    ctx_xml = build_sql_context_xml or ""
    compact_ctx = compact_build_sql_context(ctx_xml)
    suggestions = _extract_ask_user_suggestions(ctx_xml)
    slots = _slots_present(q)
    attempts_summary = _summarize_attempts_for_triage(attempts)
    context_stats = _compute_build_sql_context_stats(ctx_xml)

    payload: Dict[str, Any] = {
        "user_question": q,
        "remaining_tool_calls": int(remaining_tool_calls),
        "allow_give_best_effort": bool(allow_give_best_effort),
        "slots_present": slots,
        "attempts_summary": attempts_summary,
        "ask_user_suggestions": suggestions,
        "context_summary": compact_ctx,
        "context_stats": context_stats,
    }
    try:
        gen = get_controller_triage_generator()
        triage_obj = await gen.generate(payload=payload, react_run_id=None)
        triage = _parse_triage_json(triage_obj or {})
        if triage is None:
            raise ValueError("triage_parse_failed")
        # Enforce policy: give_best_effort only on exhaustion + allow flag.
        if triage.decision == "give_best_effort":
            if not bool(allow_give_best_effort) or int(remaining_tool_calls) > 0:
                triage.decision = "ask_user"
                triage.why = (triage.why or []) + ["policy: give_best_effort not allowed"]
        # Enforce policy: if context_refresh is chosen but empty, fill a deterministic fallback.
        if triage.decision == "context_refresh" and not triage.enrichment_queries:
            triage.enrichment_queries = _default_triage_enrichment_queries(user_question=q)
            triage.why = (triage.why or []) + ["fallback: filled enrichment_queries"]
            if not triage.enrichment_queries:
                triage.decision = "ask_user"
                triage.why = (triage.why or []) + ["fallback: no enrichment_queries provided"]

        # Guardrail: never ask end-users for technical/DB-internal artifacts.
        if triage.decision == "ask_user":
            uq = (triage.ask_user_question or "").strip()
            if _looks_like_technical_user_request(uq):
                # Prefer auto context refresh when we still have tool-call budget.
                if int(remaining_tool_calls) > 0:
                    triage.decision = "context_refresh"
                    triage.ask_user_question = ""
                    if not triage.enrichment_queries:
                        triage.enrichment_queries = _default_triage_enrichment_queries(user_question=q)
                    triage.why = (triage.why or []) + ["policy: blocked technical ask_user_question -> context_refresh"]
                    try:
                        SmartLogger.log(
                            "INFO",
                            "react.controller.triage_no_acceptable_sql.policy_block.technical_ask",
                            category="react.controller.triage_no_acceptable_sql.policy",
                            params=sanitize_for_log(
                                {
                                    "user_question_preview": q[:500],
                                    "blocked_question_preview": uq[:500],
                                    "remaining_tool_calls": int(remaining_tool_calls),
                                    "enrichment_queries": list(triage.enrichment_queries or [])[:6],
                                }
                            ),
                            max_inline_chars=0,
                        )
                    except Exception:
                        pass
                else:
                    # Tool budget exhausted: replace with a safe business-oriented question.
                    triage.ask_user_question = (
                        "정확한 조회를 위해 추가 확인이 필요합니다. "
                        "조회 대상과 "
                        "원하시는 기준(예: 평균/합계/건수, 일/월/시간 단위), "
                        "기간(예: 최근 1주/1개월 또는 특정 날짜 범위)을 알려주세요."
                    )
                    triage.why = (triage.why or []) + ["policy: blocked technical ask_user_question -> safe business question"]
        if triage.decision == "ask_user" and not triage.ask_user_question:
            triage.ask_user_question = (
                "추가 확인이 필요합니다. 질문의 대상(예: 특정 지점/지역/제품/조직)과 "
                "집계 기준(예: 평균/합계/건수, 그룹핑 단위: 일/월/시간)을 더 구체적으로 알려주세요."
            )
        try:
            SmartLogger.log(
                "INFO",
                "react.controller.triage_no_acceptable_sql.decision",
                category="react.controller.triage_no_acceptable_sql",
                params=sanitize_for_log(
                    {
                        "dbms": dbms,
                        "remaining_tool_calls": int(remaining_tool_calls),
                        "allow_give_best_effort": bool(allow_give_best_effort),
                        "decision": triage.decision,
                        "why": list(triage.why or [])[:8],
                        "ask_user_question": (triage.ask_user_question or "")[:600],
                        "enrichment_queries": list(triage.enrichment_queries or [])[:8],
                        "attempts_summary": attempts_summary,
                        "context_stats": context_stats,
                    }
                ),
                max_inline_chars=0,
            )
        except Exception:
            pass
        return triage
    except Exception as exc:
        # Log error context for triage failures; keep payload compact to avoid excessive logs.
        try:
            SmartLogger.log(
                "ERROR",
                "react.controller.triage_no_acceptable_sql.error",
                category="react.controller.triage_no_acceptable_sql.error",
                params=sanitize_for_log(
                    {
                        "dbms": dbms,
                        "remaining_tool_calls": int(remaining_tool_calls),
                        "allow_give_best_effort": bool(allow_give_best_effort),
                        "user_question_preview": q[:2000],
                        "slots_present": slots,
                        "attempts_summary": attempts_summary,
                        "ask_user_suggestions": list(suggestions or [])[:8],
                        "context_summary_len": len(compact_ctx or ""),
                        "context_summary_preview": (compact_ctx or "")[:2000],
                        "build_sql_context_xml_len": len(ctx_xml or ""),
                        "llm_model": getattr(get_controller_triage_generator().llm_handle, "model", None),
                        "exception": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                ),
                max_inline_chars=0,
            )
        except Exception:
            # Never fail the fallback path due to logging issues.
            pass
        # Safe fallback: ask_user with a generic but non-overly prescriptive question.
        return ControllerTriageResult(
            decision="ask_user",
            why=["triage: exception fallback"],
            ask_user_question=(
                "정확한 조회를 위해 추가 확인이 필요합니다. 조회 대상(예: 특정 엔티티/카테고리)과 "
                "원하시는 집계 기준(예: 평균/합계/건수, 그룹핑 단위: 일/월/시간)을 알려주세요."
            ),
            enrichment_queries=[],
        )


async def run_controller(
    *,
    question: str,
    dbms: str,
    tool_context: ToolContext,
    max_sql_seconds: int,
    config: ControllerConfig,
    prebuilt_context_xml: Optional[str] = None,
    conversation_context: Optional[Dict[str, Any]] = None,
) -> ControllerResult:
    # Always build context unless caller provides one.
    if prebuilt_context_xml and prebuilt_context_xml.strip():
        ctx_xml = prebuilt_context_xml
    else:
        ctx_xml = await execute_tool("build_sql_context", tool_context, {"question": question})
    compact_ctx = compact_build_sql_context(ctx_xml)
    # Deterministic scoring has been replaced by LLM rubric judge.
    # Keep hard preview rules deterministic (empty/meaningless result must be rejected).

    # C-pipeline:
    # - Phase 1: Explore a few candidates (ideally diverse)
    # - Phase 2: Converge via repair loop using rubric failed-checks (why 포함)
    # - Phase 3: Escape via switching to 2nd best candidate on stall

    attempts: List[ControllerAttempt] = []
    debug_attempts: List[Dict[str, Any]] = []

    # Controller progress (best-effort). Router may interleave these into NDJSON stream.
    controller_seq = 1

    # Prepare rubric judge inputs once per controller run.
    context_evidence = extract_context_evidence(ctx_xml)
    rubric_llm = None
    requirements = []
    try:
        req_started = time.perf_counter()
        _emit_controller_stage(tool_context, stage="controller_requirements", status="start", seq=controller_seq)
        rubric_llm = create_rubric_llm()
        requirements = await extract_requirements(llm=rubric_llm, question=question)
        _emit_controller_stage(
            tool_context,
            stage="controller_requirements",
            status="done",
            seq=controller_seq,
            elapsed_ms=(time.perf_counter() - req_started) * 1000.0,
            counts={"requirements_total": int(len(requirements or []))},
        )
    except Exception as exc:
        # If the rubric judge cannot be constructed/called, fail-closed later per candidate.
        try:
            _emit_controller_stage(
                tool_context,
                stage="controller_requirements",
                status="done",
                seq=controller_seq,
                elapsed_ms=(time.perf_counter() - req_started) * 1000.0 if "req_started" in locals() else None,
                counts={"requirements_total": 0},
                error=f"requirements init failed: {exc}",
            )
        except Exception:
            pass
        try:
            SmartLogger.log(
                "WARNING",
                "react.controller.rubric_judge.init_failed",
                category="react.controller",
                params=sanitize_for_log({"error": repr(exc)}),
                max_inline_chars=0,
            )
        except Exception:
            pass

    def _hard_preview_reject(preview_any: Any) -> Tuple[bool, str]:
        hard_reject = False
        hard_reason = ""
        preview = preview_any if isinstance(preview_any, dict) else {}
        try:
            rc = preview.get("row_count")
            cols = preview.get("columns")
            if isinstance(rc, int) and rc <= 0:
                hard_reject = True
                hard_reason = "preview row_count==0"
            if isinstance(cols, list) and len(cols) == 0:
                hard_reject = True
                hard_reason = hard_reason or "preview columns empty"
            cells_total, cells_non_null = _preview_non_null_stats(preview if isinstance(preview, dict) else None)
            if cells_total > 0 and cells_non_null == 0:
                hard_reject = True
                hard_reason = hard_reason or "preview result empty/all-null"
        except Exception:
            # Fail-open for hard rule evaluation errors (but still require rubric accept later)
            return (False, "")
        return (bool(hard_reject), str(hard_reason or ""))

    async def _evaluate_once(
        *,
        sql_in: str,
        attempt_no: int,
        phase: str,
        base_candidate: int,
        repair_round: int,
    ) -> Tuple[ControllerAttempt, float, bool, List[str], List[str], List[Dict[str, Any]], List[str]]:
        tool_result = await execute_tool("validate_sql", tool_context, {"sql": sql_in})
        parsed = parse_validate_sql(tool_result)
        verdict = str(parsed.get("verdict") or "").strip().upper()
        selected_sql = sanitize_sql(str(parsed.get("selected_sql") or "") or sql_in)
        preview = parsed.get("preview") if isinstance(parsed.get("preview"), dict) else {}
        suggested_fixes = parsed.get("suggested_fixes") if isinstance(parsed.get("suggested_fixes"), list) else []
        auto_rewrite = parsed.get("auto_rewrite") if isinstance(parsed.get("auto_rewrite"), dict) else {}

        sc = 0.0
        accept = False
        hard_reject = False
        hard_reason = ""
        missing_must: List[str] = []
        fail_must_ids: List[str] = []
        failed_payload: List[Dict[str, Any]] = []
        passed_must_ids: List[str] = []

        if verdict == "PASS":
            hard_reject, hard_reason = _hard_preview_reject(preview)
            if hard_reject:
                sc = 0.0
                accept = False
                missing_must = [hard_reason or "preview empty/meaningless"]
                fail_must_ids = ["__hard_preview__"]
                failed_payload = [
                    {
                        "id": "__hard_preview__",
                        "must": True,
                        "type": "other",
                        "text": hard_reason or "preview empty/meaningless",
                        "status": "FAIL",
                        "why": hard_reason[:140],
                    }
                ]
            else:
                if rubric_llm is None or not requirements:
                    sc = 0.0
                    accept = False
                    missing_must = ["rubric judge unavailable"]
                    fail_must_ids = ["__rubric_unavailable__"]
                    failed_payload = [
                        {
                            "id": "__rubric_unavailable__",
                            "must": True,
                            "type": "other",
                            "text": "rubric judge unavailable",
                            "status": "FAIL",
                            "why": "",
                        }
                    ]
                else:
                    checks = []
                    try:
                        checks = await evaluate_candidate(
                            llm=rubric_llm,
                            question=question,
                            sql=selected_sql,
                            preview=dict(preview) if isinstance(preview, dict) else {},
                            context_evidence=context_evidence,
                            requirements=requirements,
                        )
                        sc, accept, missing_must, fail_must_ids = compute_score_and_accept(
                            requirements=requirements,
                            checks=checks,
                        )
                    except Exception:
                        sc = 0.0
                        accept = False
                        missing_must = ["rubric judge failed"]
                        fail_must_ids = ["__rubric_failed__"]
                    try:
                        failed_payload, passed_must_ids = _failed_checks_payload(
                            requirements=requirements,
                            checks=checks,
                        )
                    except Exception:
                        failed_payload = []
                        passed_must_ids = []

        attempt = ControllerAttempt(
            candidate_index=int(attempt_no),
            sql=selected_sql,
            verdict=verdict,
            score=float(sc),
            fail_reason=str(parsed.get("fail_reason") or ""),
            preview=dict(preview) if isinstance(preview, dict) else {},
            tool_result_xml=str(tool_result or ""),
            phase=str(phase or ""),
            base_candidate=int(base_candidate or 0),
            repair_round=int(repair_round or 0),
            hard_reject=bool(hard_reject),
            hard_reason=str(hard_reason or "")[:120],
            rubric_fail_must_ids=list(fail_must_ids or [])[:8],
            rubric_missing_must=list(missing_must or [])[:8],
            rubric_failed_checks=list(failed_payload or [])[:24],
            validate_suggested_fixes=[str(x or "")[:240] for x in (suggested_fixes or []) if str(x or "").strip()][:8],
            validate_auto_rewrite=dict(auto_rewrite) if isinstance(auto_rewrite, dict) else {},
        )
        try:
            debug_attempts.append(
                {
                    "attempt": int(attempt_no),
                    "phase": str(phase),
                    "base_candidate": int(base_candidate or 0),
                    "repair_round": int(repair_round or 0),
                    "verdict": str(verdict),
                    "score": float(sc),
                    "accept": bool(accept),
                    "hard_reject": bool(hard_reject),
                    "hard_reason": str(hard_reason)[:120],
                    "row_count": int(preview.get("row_count") or 0) if isinstance(preview, dict) else None,
                    "fail_must_ids": list(fail_must_ids or [])[:8],
                    "sql_preview": selected_sql[:260],
                }
            )
        except Exception:
            pass
        return attempt, float(sc), bool(accept), list(missing_must or [])[:8], list(fail_must_ids or [])[:8], list(failed_payload or [])[:32], list(passed_must_ids or [])[:32]

    # Phase 1) Explore: evaluate a few initial candidates.
    init_n = int(getattr(config, "initial_candidates", 0) or 0)
    if init_n <= 0:
        init_n = int(getattr(config, "n_candidates", 2) or 2)
    init_n = max(1, init_n)
    init_n = int(min(init_n, max(1, int(getattr(config, "n_candidates", 2) or 2))))

    # Candidate diversity hints (generic, domain-agnostic).
    diversity_hints = [
        "Strategy A: Use a straight JOIN path from the primary fact table to the entity table; filter entity by NAME if available; GROUP BY day.",
        "Strategy B: Use a CTE to resolve entity CODE/ID from entity NAME first, then join fact tables by CODE/ID; keep aggregates and GROUP BY day.",
        "Strategy C: Use EXISTS subquery (instead of JOIN filter) for entity constraint; keep output minimal and correct grain.",
    ]
    cand_temp = float(getattr(config, "candidate_temperature", 0.0) or 0.0)
    controller_seq += 1
    cand_started = time.perf_counter()
    _emit_controller_stage(tool_context, stage="controller_candidates", status="start", seq=controller_seq)
    candidates_raw = await draft_llm_candidates(
        question=question,
        dbms=dbms,
        max_sql_seconds=max_sql_seconds,
        compact_context=compact_ctx,
        conversation_context=conversation_context,
        n_candidates=int(init_n),
        temperature=cand_temp,
        diversity_hints=diversity_hints[: max(1, int(init_n))],
        seed=None,
    )
    candidates: List[str] = []
    seen = set()
    for x in candidates_raw[: max(1, int(init_n))]:
        sx = sanitize_sql(x)
        if not sx:
            continue
        k = sx.lower()
        if k in seen:
            continue
        seen.add(k)
        candidates.append(sx)

    _emit_controller_stage(
        tool_context,
        stage="controller_candidates",
        status="done",
        seq=controller_seq,
        elapsed_ms=(time.perf_counter() - cand_started) * 1000.0,
        counts={"candidates_total": int(len(candidates))},
    )

    best_sql = ""
    best_score = 0.0
    best_fail_ids: List[str] = []
    best_failed_payload: List[Dict[str, Any]] = []
    best_passed_must_ids: List[str] = []

    second_sql = ""
    second_score = 0.0
    second_fail_ids: List[str] = []
    second_failed_payload: List[Dict[str, Any]] = []
    second_passed_must_ids: List[str] = []

    best_preview: Dict[str, Any] = {}
    best_attempt: Optional[ControllerAttempt] = None
    second_attempt: Optional[ControllerAttempt] = None

    attempt_no = 0
    controller_seq += 1
    validate_started = time.perf_counter()
    _emit_controller_stage(
        tool_context,
        stage="controller_validate",
        status="start",
        seq=controller_seq,
        counts={"candidates_total": int(len(candidates))},
    )
    for base_idx, sql0 in enumerate(candidates, start=1):
        attempt_no += 1
        one_started = time.perf_counter()
        att, sc, accept, _miss, fail_ids, failed_payload, passed_must_ids = await _evaluate_once(
            sql_in=sql0,
            attempt_no=attempt_no,
            phase="initial",
            base_candidate=base_idx,
            repair_round=0,
        )
        attempts.append(att)
        try:
            rc = None
            if isinstance(getattr(att, "preview", None), dict):
                raw_rc = att.preview.get("row_count")
                if raw_rc is not None:
                    rc = int(raw_rc)
            _emit_controller_item(
                tool_context,
                stage="controller_validate",
                item_type="candidate",
                index=int(base_idx),
                total=int(max(1, len(candidates))),
                verdict=str(getattr(att, "verdict", "") or ""),
                row_count=rc,
                elapsed_ms=(time.perf_counter() - one_started) * 1000.0,
            )
        except Exception:
            pass

        if att.verdict == "PASS":
            if not best_sql or float(sc) > float(best_score):
                # shift down to second
                if best_sql:
                    second_sql = best_sql
                    second_score = float(best_score)
                    second_fail_ids = list(best_fail_ids or [])
                    second_failed_payload = list(best_failed_payload or [])
                    second_passed_must_ids = list(best_passed_must_ids or [])
                    second_attempt = best_attempt
                best_sql = att.sql
                best_score = float(sc)
                best_fail_ids = list(fail_ids or [])
                best_failed_payload = list(failed_payload or [])
                best_passed_must_ids = list(passed_must_ids or [])
                best_preview = dict(att.preview) if isinstance(att.preview, dict) else {}
                best_attempt = att
            elif not second_sql or float(sc) > float(second_score):
                second_sql = att.sql
                second_score = float(sc)
                second_fail_ids = list(fail_ids or [])
                second_failed_payload = list(failed_payload or [])
                second_passed_must_ids = list(passed_must_ids or [])
                second_attempt = att

        if att.verdict == "PASS" and bool(accept) and float(sc) >= float(config.score_threshold):
            try:
                SmartLogger.log(
                    "INFO",
                    "react.controller.accepted",
                    category="react.controller",
                    params=sanitize_for_log(
                        {
                            "score_threshold": float(config.score_threshold),
                            "accepted": {"candidate_index": int(att.candidate_index), "score": float(sc), "accept": True},
                            "attempts": debug_attempts[-12:],
                        }
                    ),
                    max_inline_chars=0,
                )
            except Exception:
                pass
            try:
                _emit_controller_stage(
                    tool_context,
                    stage="controller_validate",
                    status="done",
                    seq=controller_seq,
                    elapsed_ms=(time.perf_counter() - validate_started) * 1000.0,
                )
            except Exception:
                pass
            return ControllerResult(
                status="submit_sql",
                final_sql=att.sql,
                validated_sql=att.sql,
                preview=dict(att.preview) if isinstance(att.preview, dict) else {},
                attempts=attempts,
                build_sql_context_xml=ctx_xml,
            )

    # If nothing passed validate_sql, we cannot run rubric-based convergence.
    if not best_sql:
        # No acceptable SQL (no PASS candidates)
        try:
            SmartLogger.log(
                "INFO",
                "react.controller.no_acceptable_sql",
                category="react.controller",
                params=sanitize_for_log(
                    {
                        "score_threshold": float(config.score_threshold),
                        "attempts_total": int(len(attempts)),
                        "best_pass_score": 0.0,
                        "best_pass_sql_preview": "",
                        "attempts": debug_attempts[-12:],
                    }
                ),
                max_inline_chars=0,
            )
        except Exception:
            pass
        try:
            _emit_controller_stage(
                tool_context,
                stage="controller_validate",
                status="done",
                seq=controller_seq,
                elapsed_ms=(time.perf_counter() - validate_started) * 1000.0,
            )
        except Exception:
            pass
        return ControllerResult(
            status="ask_user",
            final_sql="",
            validated_sql="",
            preview={},
            attempts=attempts,
            question_to_user=(
                "질문을 실행 가능한 SQL로 확정하려면 몇 가지 확인이 필요합니다. "
                "원하시는 기간(예: 최근 7일/1개월)과 집계 기준(예: 평균/합계/건수, 그룹핑 단위: 일/월/시간), "
                "그리고 필터 조건(예: 특정 엔티티/카테고리/코드 등)을 알려주세요."
            ),
            build_sql_context_xml=ctx_xml,
        )

    # Phase 2/3) Converge + Escape
    active_sql = best_sql
    active_score = float(best_score)
    active_fail_ids = list(best_fail_ids or [])
    active_failed_payload = list(best_failed_payload or [])
    active_passed_must_ids = list(best_passed_must_ids or [])
    active_suggested_fixes: List[str] = list(getattr(best_attempt, "validate_suggested_fixes", []) or []) if best_attempt is not None else []
    active_auto_rewrite: Dict[str, Any] = dict(getattr(best_attempt, "validate_auto_rewrite", {}) or {}) if best_attempt is not None else {}
    active_base = 1

    fallback_sql = second_sql
    fallback_score = float(second_score)
    fallback_fail_ids = list(second_fail_ids or [])
    fallback_failed_payload = list(second_failed_payload or [])
    fallback_passed_must_ids = list(second_passed_must_ids or [])
    fallback_suggested_fixes: List[str] = list(getattr(second_attempt, "validate_suggested_fixes", []) or []) if second_attempt is not None else []
    fallback_auto_rewrite: Dict[str, Any] = dict(getattr(second_attempt, "validate_auto_rewrite", {}) or {}) if second_attempt is not None else {}
    switched = False

    stall = 0
    last_key = _norm_sql_key(active_sql)

    max_repairs = int(getattr(config, "max_repairs_per_candidate", 0) or 0)
    if max_repairs <= 0 and bool(getattr(config, "allow_one_revision", True)):
        max_repairs = 1

    controller_seq += 1
    repair_stage_seq = controller_seq
    repair_stage_started = time.perf_counter()
    _emit_controller_stage(
        tool_context,
        stage="controller_repair",
        status="start",
        seq=repair_stage_seq,
        counts={"repair_rounds_total": int(max(1, int(max_repairs)))},
    )
    for rr in range(1, max(1, int(max_repairs)) + 1):
        # Build repair hints: prefer structured failed_checks; fall back to legacy ids if needed.
        legacy_missing: List[str] = []
        if not active_failed_payload and active_fail_ids:
            legacy_missing = [str(x or "").strip() for x in (active_fail_ids or []) if str(x or "").strip()][:12]

        repair_temp = float(getattr(config, "repair_temperature", 0.0) or 0.0)
        gen = get_controller_repair_generator()
        revised_raw, _mode = await gen.generate(
            question=question,
            missing_requirements=legacy_missing,
            failed_checks=list(active_failed_payload or [])[:48],
            passed_must_ids=list(active_passed_must_ids or [])[:48],
            suggested_fixes=list(active_suggested_fixes or [])[:12],
            auto_rewrite=dict(active_auto_rewrite or {}) if isinstance(active_auto_rewrite, dict) else {},
            context_xml=compact_ctx,
            conversation_context=conversation_context,
            current_sql=active_sql,
            temperature=repair_temp,
            react_run_id=None,
        )
        revised = sanitize_sql(str(revised_raw or ""))
        if not revised:
            stall += 1
        else:
            attempt_no += 1
            rr_started = time.perf_counter()
            att, sc, accept, _miss, fail_ids, failed_payload, passed_must_ids = await _evaluate_once(
                sql_in=revised,
                attempt_no=attempt_no,
                phase="repair" if rr > 0 else "revision",
                base_candidate=int(active_base),
                repair_round=int(rr),
            )
            attempts.append(att)
            try:
                rc = None
                if isinstance(getattr(att, "preview", None), dict):
                    raw_rc = att.preview.get("row_count")
                    if raw_rc is not None:
                        rc = int(raw_rc)
                _emit_controller_item(
                    tool_context,
                    stage="controller_repair",
                    item_type="repair_round",
                    index=int(rr),
                    total=int(max(1, int(max_repairs))),
                    verdict=str(getattr(att, "verdict", "") or ""),
                    row_count=rc,
                    elapsed_ms=(time.perf_counter() - rr_started) * 1000.0,
                )
            except Exception:
                pass

            new_key = _norm_sql_key(att.sql)
            improved_score = float(sc) > float(active_score) + 0.01
            reduced_fail = len(list(fail_ids or [])) < len(list(active_fail_ids or []))
            same_fail = _same_fail_set(active_fail_ids, fail_ids)
            if new_key == last_key:
                stall += 1
            elif (not improved_score) and (not reduced_fail) and same_fail:
                stall += 1
            else:
                stall = 0
            last_key = new_key

            # Accept immediately if threshold met.
            if att.verdict == "PASS" and bool(accept) and float(sc) >= float(config.score_threshold):
                try:
                    SmartLogger.log(
                        "INFO",
                        "react.controller.accepted",
                        category="react.controller",
                        params=sanitize_for_log(
                            {
                                "score_threshold": float(config.score_threshold),
                                "accepted": {"candidate_index": int(att.candidate_index), "score": float(sc), "accept": True, "repair_round": int(rr)},
                                "attempts": debug_attempts[-12:],
                            }
                        ),
                        max_inline_chars=0,
                    )
                except Exception:
                    pass
                try:
                    _emit_controller_stage(
                        tool_context,
                        stage="controller_repair",
                        status="done",
                        seq=repair_stage_seq,
                        elapsed_ms=(time.perf_counter() - repair_stage_started) * 1000.0,
                    )
                    _emit_controller_stage(
                        tool_context,
                        stage="controller_validate",
                        status="done",
                        seq=controller_seq - 1,
                        elapsed_ms=(time.perf_counter() - validate_started) * 1000.0,
                    )
                except Exception:
                    pass
                return ControllerResult(
                    status="submit_sql",
                    final_sql=att.sql,
                    validated_sql=att.sql,
                    preview=dict(att.preview) if isinstance(att.preview, dict) else {},
                    attempts=attempts,
                    build_sql_context_xml=ctx_xml,
                )

            # Update active candidate only when it doesn't regress too much.
            if att.verdict == "PASS" and (improved_score or reduced_fail or float(sc) >= float(active_score)):
                active_sql = att.sql
                active_score = float(sc)
                active_fail_ids = list(fail_ids or [])
                active_failed_payload = list(failed_payload or [])
                active_passed_must_ids = list(passed_must_ids or [])
                active_suggested_fixes = list(getattr(att, "validate_suggested_fixes", []) or [])
                active_auto_rewrite = dict(getattr(att, "validate_auto_rewrite", {}) or {}) if isinstance(getattr(att, "validate_auto_rewrite", None), dict) else {}

        # Phase 3) Escape: switch to fallback candidate on stall (once).
        if int(stall) >= max(1, int(getattr(config, "stall_rounds", 2) or 2)):
            if (
                (not switched)
                and bool(getattr(config, "candidate_switch_enabled", True))
                and fallback_sql
                and _norm_sql_key(fallback_sql) != _norm_sql_key(active_sql)
            ):
                switched = True
                stall = 0
                active_sql = fallback_sql
                active_score = float(fallback_score)
                active_fail_ids = list(fallback_fail_ids or [])
                active_failed_payload = list(fallback_failed_payload or [])
                active_passed_must_ids = list(fallback_passed_must_ids or [])
                active_suggested_fixes = list(fallback_suggested_fixes or [])
                active_auto_rewrite = dict(fallback_auto_rewrite or {}) if isinstance(fallback_auto_rewrite, dict) else {}
                active_base += 1
                last_key = _norm_sql_key(active_sql)
            else:
                break

    # No acceptable SQL
    try:
        _emit_controller_stage(
            tool_context,
            stage="controller_repair",
            status="done",
            seq=repair_stage_seq,
            elapsed_ms=(time.perf_counter() - repair_stage_started) * 1000.0,
        )
        _emit_controller_stage(
            tool_context,
            stage="controller_validate",
            status="done",
            seq=controller_seq - 1,
            elapsed_ms=(time.perf_counter() - validate_started) * 1000.0,
        )
    except Exception:
        pass
    try:
        SmartLogger.log(
            "INFO",
            "react.controller.no_acceptable_sql",
            category="react.controller",
            params=sanitize_for_log(
                {
                    "score_threshold": float(config.score_threshold),
                    "attempts_total": int(len(attempts)),
                    "best_pass_score": float(best_score or 0.0),
                    "best_pass_sql_preview": (best_sql or "")[:260],
                    "attempts": debug_attempts[-12:],
                }
            ),
            max_inline_chars=0,
        )
    except Exception:
        pass
    return ControllerResult(
        status="ask_user",
        final_sql="",
        validated_sql="",
        preview={},
        attempts=attempts,
        question_to_user=(
            "질문을 실행 가능한 SQL로 확정하려면 몇 가지 확인이 필요합니다. "
            "원하시는 기간(예: 최근 7일/1개월)과 집계 기준(예: 평균/합계/건수, 그룹핑 단위: 일/월/시간), "
            "그리고 필터 조건(예: 특정 엔티티/카테고리/코드 등)을 알려주세요."
        ),
        build_sql_context_xml=ctx_xml,
    )


