from __future__ import annotations

import re
from app.react.state import ReactSessionState


AUTO_ADDED_PURPOSE = "Referenced in reasoning (auto-added from search_tables)"


def auto_enrich_tables_from_reasoning(
    reasoning: str,
    state: ReactSessionState,
) -> None:
    """
    Reasoning 텍스트에 search_tables 후보 테이블명이 언급되면
    해당 테이블을 자동으로 metadata.identified_tables 에 추가한다.
    """
    if not reasoning or not state.search_table_candidates:
        return

    for candidate in state.search_table_candidates:
        candidate_name = (candidate.get("name") or "").strip()
        candidate_description = (candidate.get("description") or "").strip()
        candidate_schema = (candidate.get("schema") or "").strip()
        if not candidate_name:
            continue
        if state.metadata.has_table(candidate_name, candidate_schema or None):
            continue

        pattern = rf"\b{re.escape(candidate_name)}\b"
        if not re.search(pattern, reasoning, flags=re.IGNORECASE):
            continue

        state.metadata.add_table_if_missing(
            table_name=candidate_name,
            schema=candidate_schema,
            purpose=AUTO_ADDED_PURPOSE,
            key_columns="",
            description=candidate_description,
        )
