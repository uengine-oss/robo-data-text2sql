from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from app.config import settings

_GENERIC_TERMS = {
    "전체",
    "모든",
    "조회",
    "검색",
    "보여줘",
    "알려줘",
    "해주세요",
    "데이터",
    "값",
    "결과",
    "현황",
    "리스트",
}

# Korean postpositions/particles frequently attached to entity names.
# Used for normalization in token extraction to improve matching (e.g., 청주정수장의 -> 청주정수장).
_KOREAN_TRAILING_PARTICLES = (
    "의",
    "을",
    "를",
    "은",
    "는",
    "이",
    "가",
    "과",
    "와",
    "에",
    "에서",
    "으로",
    "로",
    "부터",
    "까지",
    "및",
)

# Hard anchors for common join/filter/value patterns (domain + generic).
# 현재는 비어있지만, 필요하면 런타임에 서브스트링을 추가할 수 있다.
_ANCHOR_COLUMN_SUBSTRINGS: set[str] = set()


def _dedupe_keep_order(values: Sequence[str], *, limit: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values or []:
        s = str(v or "").strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= int(limit):
            break
    return out


def _truncate(text: str, max_len: int) -> str:
    t = str(text or "")
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


def _compact_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _pack_table_match_for_log(m: Any, *, desc_max_len: int = 200) -> Dict[str, Any]:
    """
    Pack a GraphSearcher table match object into a small log-friendly dict.
    Keep it stable and size-bounded to avoid noisy logs.
    """
    try:
        schema = str(getattr(m, "schema", "") or "")
        name = str(getattr(m, "name", "") or "")
        score = float(getattr(m, "score", 0.0) or 0.0)
        desc = _truncate(_compact_ws(str(getattr(m, "description", "") or "")), int(desc_max_len))
        analyzed = _truncate(
            _compact_ws(str(getattr(m, "analyzed_description", "") or "")), int(desc_max_len)
        )
        return {
            "schema": schema,
            "name": name,
            "fqn": f"{schema}.{name}" if schema else name,
            "score": score,
            "description": desc,
            "analyzed_description": analyzed,
        }
    except Exception:
        return {"schema": "", "name": "", "fqn": "", "score": 0.0}


def _regex_terms(question: str, *, limit: int = 20) -> List[str]:
    text = (question or "").strip()
    if not text:
        return []
    tokens = re.findall(r"[가-힣]{2,}|[A-Za-z0-9_]{2,}", text)
    uniq: List[str] = []
    seen = set()
    for t in tokens:
        raw = t.strip()
        if not raw:
            continue

        # Expand simple Korean variants by stripping common trailing particles.
        cand: List[str] = [raw]
        if re.fullmatch(r"[가-힣]{2,}", raw):
            for suf in _KOREAN_TRAILING_PARTICLES:
                if raw.endswith(suf) and len(raw) >= 3:
                    stripped = raw[: -len(suf)].strip()
                    if len(stripped) >= 2:
                        cand.append(stripped)

        for term in cand:
            term = term.strip()
            if not term:
                continue
            if term in _GENERIC_TERMS:
                continue
            if term.endswith("코드"):
                continue
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(term)
            if len(uniq) >= int(limit):
                break
        if len(uniq) >= int(limit):
            break
    return uniq[:limit]


_RE_TMP = re.compile(r"(^|_)(tmp|test|back|bk|old|drop|bf|bak)(_|$)", re.IGNORECASE)
_RE_VIEW = re.compile(r"^(q_|calc_|vw_|view_)", re.IGNORECASE)
_RE_DATE = re.compile(r"(19|20)\d{6}")  # yyyymmdd-like


def _table_name_penalty(name: str) -> float:
    """
    Generic penalty to downweight temporary/test/view-like artifacts without domain hardcoding.
    """
    n = (name or "").strip().lower()
    pen = 0.0
    try:
        if _RE_TMP.search(n):
            pen += float(getattr(settings, "table_penalty_tmp", 0.03) or 0.03)
        if _RE_VIEW.search(n):
            pen += float(getattr(settings, "table_penalty_view", 0.015) or 0.015)
        if _RE_DATE.search(n):
            pen += float(getattr(settings, "table_penalty_date", 0.01) or 0.01)
    except Exception:
        # never fail retrieval due to tuning values
        pass
    return float(pen)


def _base_name_candidates(name: str) -> List[str]:
    """
    Generic sibling candidates (domain-agnostic).
    """
    raw = (name or "").strip()
    if not raw:
        return []
    n = raw
    for pref in ("TMP_", "TEST_", "Q_", "CALC_", "VW_", "VIEW_"):
        if n.upper().startswith(pref):
            n = n[len(pref) :]
            break
    n2 = re.sub(r"_TB\d+$", "_TB", n, flags=re.IGNORECASE)
    n3 = re.sub(r"(DEL)?LOG(_TB)?$", "_TB", n2, flags=re.IGNORECASE)
    n4 = re.sub(r"(DEL)?LOG", "", n2, flags=re.IGNORECASE)
    n4 = re.sub(r"__+", "_", n4).strip("_")
    if not re.search(r"_TB$", n4, flags=re.IGNORECASE) and re.search(r"_TB", n2, flags=re.IGNORECASE):
        n4 = n4 + "_TB"
    out = [n2, n3, n4]
    uniq: List[str] = []
    seen = set()
    for x in out:
        x = (x or "").strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        uniq.append(x)
    return uniq[:3]


def _guess_unbounded_scope(question: str) -> bool:
    q = (question or "").strip()
    if not q:
        return False
    return any(token in q for token in ("전체", "모든", "전기간", "전체기간"))


__all__ = [
    "_GENERIC_TERMS",
    "_KOREAN_TRAILING_PARTICLES",
    "_ANCHOR_COLUMN_SUBSTRINGS",
    "_dedupe_keep_order",
    "_truncate",
    "_compact_ws",
    "_pack_table_match_for_log",
    "_regex_terms",
    "_table_name_penalty",
    "_base_name_candidates",
    "_guess_unbounded_scope",
]


