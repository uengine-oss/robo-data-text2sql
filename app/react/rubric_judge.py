from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.react.llm_factory import create_react_llm
from app.react.generators.rubric_evaluate_candidate_generator import (
    get_rubric_evaluate_candidate_generator,
)
from app.react.generators.rubric_extract_requirements_generator import (
    get_rubric_extract_requirements_generator,
)


@dataclass(frozen=True, slots=True)
class RubricRequirement:
    id: str
    must: bool
    type: str
    text: str


@dataclass(frozen=True, slots=True)
class RubricCheck:
    id: str
    status: str  # PASS|FAIL|UNKNOWN
    why: str = ""


def _is_falsey_env(v: Optional[str]) -> bool:
    t = (v or "").strip()
    return t in ("0", "false", "False", "no", "NO", "off", "OFF")


def create_rubric_llm():
    """
    Rubric judge용 LLM 핸들 생성.

    - 기본: light LLM (비용/지연 최소)
    - env: REACT_RUBRIC_USE_LIGHT=0 이면 일반 LLM 사용
    """
    use_light = not _is_falsey_env(os.environ.get("REACT_RUBRIC_USE_LIGHT"))
    max_tokens = 800 if use_light else 1500
    return create_react_llm(
        purpose="rubric_judge",
        thinking_level=None,
        include_thoughts=False,
        temperature=0.0,
        max_output_tokens=max_tokens,
        use_light=use_light,
    ).llm


def extract_context_evidence(build_sql_context_xml: str, *, max_items: int = 10) -> Dict[str, Any]:
    """
    build_sql_context tool_result XML에서 LLM judge에 줄 '증거'를 가볍게 추출한다.
    (XML 파서 대신 regex를 사용해, 일부 비정상 텍스트에도 최대한 견고하게 동작하도록 한다.)

    현재는 실전에서 가장 도움이 컸던 코드<->명 매핑 위주로 최소만 제공한다:
    - facility_mappings: SUJ_CODE <-> SUJ_NAME
    - br_code_hints: BR_CODE <-> BR_NAME (가능하면)
    """
    text = (build_sql_context_xml or "")
    if not text.strip():
        return {}

    out: Dict[str, Any] = {"facility_mappings": [], "br_code_hints": []}

    try:
        # Prefer light_queries section if present
        m_light = re.search(r"<light_queries>[\s\S]*?</light_queries>", text)
        block = m_light.group(0) if m_light else text

        facilities: List[Dict[str, str]] = []
        brs: List[Dict[str, str]] = []

        for row_m in re.finditer(r"<row index=\"\d+\">([\s\S]*?)</row>", block):
            row = row_m.group(1)
            c_suj_code = re.search(r"<cell column=\"SUJ_CODE\"><!\[CDATA\[([^\]]+)\]\]></cell>", row)
            c_suj_name = re.search(r"<cell column=\"SUJ_NAME\"><!\[CDATA\[([^\]]+)\]\]></cell>", row)
            if c_suj_code and c_suj_name:
                facilities.append({"SUJ_CODE": c_suj_code.group(1).strip(), "SUJ_NAME": c_suj_name.group(1).strip()})

            c_br_code = re.search(r"<cell column=\"BR_CODE\"><!\[CDATA\[([^\]]+)\]\]></cell>", row)
            c_br_name = re.search(r"<cell column=\"BR_NAME\"><!\[CDATA\[([^\]]+)\]\]></cell>", row)
            if c_br_code and c_br_name:
                brs.append({"BR_CODE": c_br_code.group(1).strip(), "BR_NAME": c_br_name.group(1).strip()})

        # Dedup
        seen_f = set()
        uniq_f: List[Dict[str, str]] = []
        for x in facilities:
            k = (x.get("SUJ_CODE") or "") + "|" + (x.get("SUJ_NAME") or "")
            if not k.strip() or k in seen_f:
                continue
            seen_f.add(k)
            uniq_f.append(x)

        seen_b = set()
        uniq_b: List[Dict[str, str]] = []
        for x in brs:
            k = (x.get("BR_CODE") or "") + "|" + (x.get("BR_NAME") or "")
            if not k.strip() or k in seen_b:
                continue
            seen_b.add(k)
            uniq_b.append(x)

        out["facility_mappings"] = uniq_f[: max(1, int(max_items))]
        out["br_code_hints"] = uniq_b[: max(1, int(max_items))]
    except Exception:
        # Fail-open: evidence missing is acceptable
        return {}

    return out


def _normalize_requirements(obj: Any) -> List[RubricRequirement]:
    reqs_raw = None
    if isinstance(obj, dict):
        reqs_raw = obj.get("requirements")
    if not isinstance(reqs_raw, list):
        return []

    out: List[RubricRequirement] = []
    for i, r in enumerate(reqs_raw, start=1):
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id") or f"R{i}").strip() or f"R{i}"
        must = bool(r.get("must")) if "must" in r else True
        typ = str(r.get("type") or "").strip() or "other"
        txt = str(r.get("text") or "").strip()
        if not txt:
            continue
        out.append(RubricRequirement(id=rid[:24], must=must, type=typ[:24], text=txt[:220]))

    # Ensure stable order and avoid duplicates by id
    seen = set()
    uniq: List[RubricRequirement] = []
    for r in out:
        if r.id in seen:
            continue
        seen.add(r.id)
        uniq.append(r)
    return uniq[:24]


def _normalize_checks(obj: Any) -> List[RubricCheck]:
    checks_raw = None
    if isinstance(obj, dict):
        checks_raw = obj.get("checks")
    if not isinstance(checks_raw, list):
        return []

    out: List[RubricCheck] = []
    for c in checks_raw:
        if not isinstance(c, dict):
            continue
        cid = str(c.get("id") or "").strip()
        if not cid:
            continue
        st = str(c.get("status") or "").strip().upper()
        if st not in ("PASS", "FAIL", "UNKNOWN"):
            st = "UNKNOWN"
        why = str(c.get("why") or "").strip()
        out.append(RubricCheck(id=cid[:24], status=st, why=why[:140]))

    # Dedup by id
    seen = set()
    uniq: List[RubricCheck] = []
    for c in out:
        if c.id in seen:
            continue
        seen.add(c.id)
        uniq.append(c)
    return uniq[:48]


async def extract_requirements(*, llm, question: str) -> List[RubricRequirement]:
    """
    질문에서 요구사항 체크리스트를 추출한다. (LLM 1회)

    LLM 응답은 반드시 JSON 하나만 반환하도록 강제한다.
    """
    gen = get_rubric_extract_requirements_generator()
    obj = await gen.generate(llm=llm, question=question)
    return _normalize_requirements(obj or {})


async def evaluate_candidate(
    *,
    llm,
    question: str,
    sql: str,
    preview: Dict[str, Any],
    context_evidence: Dict[str, Any],
    requirements: List[RubricRequirement],
) -> List[RubricCheck]:
    """
    (SQL + preview + evidence)로 각 요구사항 충족 여부를 판정한다. (LLM 1회)
    """
    req_payload = [{"id": r.id, "must": r.must, "type": r.type, "text": r.text} for r in requirements]
    gen = get_rubric_evaluate_candidate_generator()
    obj = await gen.generate(
        llm=llm,
        question=question,
        sql=sql,
        preview=preview,
        context_evidence=context_evidence,
        requirements_payload=req_payload,
    )
    return _normalize_checks(obj or {})


def compute_score_and_accept(
    *,
    requirements: List[RubricRequirement],
    checks: List[RubricCheck],
) -> Tuple[float, bool, List[str], List[str]]:
    """
    LLM이 준 checks를 기반으로 (결정론적으로) score/accept를 계산한다.

    Returns:
      - score: [0,1]
      - accept: MUST requirement가 전부 PASS이면 True
      - missing_must: MUST 중 FAIL/UNKNOWN인 요구사항 텍스트 요약
      - fail_must_ids: FAIL/UNKNOWN인 MUST requirement ids
    """
    if not requirements:
        return 0.0, False, ["no requirements extracted"], ["__no_requirements__"]

    check_map: Dict[str, RubricCheck] = {c.id: c for c in checks or []}

    pass_w = 0.0
    total_w = 0.0
    accept = True
    missing_must: List[str] = []
    fail_must_ids: List[str] = []

    for r in requirements:
        c = check_map.get(r.id)
        st = (c.status if c is not None else "UNKNOWN").upper()

        total_w += 1.0
        if st == "PASS":
            pass_w += 1.0
        elif st == "UNKNOWN":
            pass_w += 0.5
        else:
            pass_w += 0.0

        if r.must and st != "PASS":
            accept = False
            fail_must_ids.append(r.id)
            missing_must.append(f"{r.text} ({st})")

    score = (pass_w / total_w) if total_w > 0 else 0.0
    return max(0.0, min(1.0, float(score))), bool(accept), missing_must[:8], fail_must_ids[:8]

