from typing import Any, Dict, List, Optional, Set

from app.core.embedding import EmbeddingClient
from app.core.graph_search import GraphSearcher, TableMatch
from app.react.tools.context import ToolContext
from app.react.tools.neo4j_utils import (
    get_table_importance_scores,
    get_table_relationship_details,
)


async def execute(
    context: ToolContext,
    keywords: List[str],
) -> str:
    """
    Neo4j 스키마 그래프에서 키워드와 유사한 테이블을 검색한다.
    결과는 prompt.xml 지시에 맞춰 XML 문자열로 반환한다.
    """
    search_table_keyword_limit = max(int(context.scaled(context.search_table_keyword_limit)), 1)
    keywords = keywords[:search_table_keyword_limit]

    table_top_k = max(int(context.scaled(context.table_top_k)), 1)
    relation_limit = max(int(context.scaled(context.table_relation_limit)), 0)
    table_fetch_limit = max(table_top_k * 5, table_top_k)

    embedding_client = EmbeddingClient(context.openai_client)
    query_embeddings = await embedding_client.embed_batch(keywords)
    searcher = GraphSearcher(context.neo4j_session)
    importance_map = await get_table_importance_scores(context.neo4j_session)
    max_importance_score = max(
        (item.get("importance_score", 0) or 0) for item in importance_map.values()
    ) if importance_map else 0

    result_parts: List[str] = ["<tool_result>"]
    output_table_names: Set[str] = set()

    def _table_key(name: str, schema: Optional[str]) -> str:
        return f"{schema}.{name}" if schema else name

    for keyword, embedding in zip(keywords, query_embeddings):
        matches = await searcher.search_tables(embedding, k=table_fetch_limit)
        filtered_matches = [
            match
            for match in matches
            if _table_key(match.name, match.schema) not in output_table_names
        ]
        selected_matches = _select_table_matches(
            filtered_matches,
            table_top_k,
            importance_map,
            max_importance_score,
        )

        result_parts.append(f'<related_tables used_keyword="{keyword}">')

        for match in selected_matches:
            table_key = _table_key(match.name, match.schema)
            if table_key in output_table_names:
                continue

            result_parts.append("<table>")
            result_parts.append(f"<schema>{match.schema or ''}</schema>")
            result_parts.append(f"<name>{match.name}</name>")
            if match.description:
                result_parts.append(f"<description>{match.description}</description>")

            relationship_details = await get_table_relationship_details(
                context.neo4j_session,
                match.name,
                relation_limit=relation_limit,
            )

            fk_relationships = relationship_details["fk_relationships"]
            if fk_relationships:
                fk_relationships = sorted(
                    fk_relationships,
                    key=lambda r: (
                        r.get("related_table_schema") or "",
                        r.get("related_table") or "",
                        r.get("from_column") or "",
                        r.get("to_column") or "",
                        r.get("relation_type") or "",
                    ),
                )
                result_parts.append("<fk_relationships>")
                for rel in fk_relationships:
                    result_parts.append("<fk_relationship>")
                    if "related_table_schema" in rel:
                        result_parts.append(
                            f"<related_table_schema>{rel.get('related_table_schema') or ''}</related_table_schema>"
                        )
                    result_parts.append(f"<related_table>{rel['related_table']}</related_table>")
                    if rel.get("related_table_description"):
                        result_parts.append(
                            f"<related_table_description>{rel['related_table_description']}</related_table_description>"
                        )
                    result_parts.append(f"<relation_type>{rel['relation_type']}</relation_type>")
                    if rel.get("from_column"):
                        result_parts.append(f"<from_column>{rel['from_column']}</from_column>")
                    if rel.get("from_column_description"):
                        result_parts.append(
                            f"<from_column_description>{rel['from_column_description']}</from_column_description>"
                        )
                    if rel.get("to_column"):
                        result_parts.append(f"<to_column>{rel['to_column']}</to_column>")
                    if rel.get("to_column_description"):
                        result_parts.append(
                            f"<to_column_description>{rel['to_column_description']}</to_column_description>"
                        )
                    result_parts.append("</fk_relationship>")
                result_parts.append("</fk_relationships>")

            relationships_to_output = relationship_details["additional_relationships"]
            if relationships_to_output:
                relationships_to_output = sorted(
                    relationships_to_output,
                    key=lambda r: (
                        r.get("related_table_schema") or "",
                        r.get("related_table") or "",
                    ),
                )
                result_parts.append("<relationships>")
                for rel in relationships_to_output:
                    result_parts.append("<relationship>")
                    result_parts.append("<table>")
                    result_parts.append(
                        f"<schema>{rel.get('related_table_schema') or ''}</schema>"
                    )
                    result_parts.append(f"<name>{rel['related_table']}</name>")
                    if rel.get("related_table_description"):
                        result_parts.append(
                            f"<description>{rel['related_table_description']}</description>"
                        )
                    if rel.get("relationship_type"):
                        result_parts.append(
                            f"<relationship_type>{rel['relationship_type']}</relationship_type>"
                        )
                    result_parts.append("</table>")
                    result_parts.append("</relationship>")
                result_parts.append("</relationships>")

            result_parts.append("</table>")
            output_table_names.add(table_key)

        result_parts.append("</related_tables>")

    result_parts.append("</tool_result>")
    return "\n".join(result_parts)


def _select_table_matches(
    candidates: List[TableMatch],
    table_top_k: int,
    importance_map: Dict[str, Dict[str, Any]],
    max_importance_score: float,
) -> List[TableMatch]:
    """table_top_k 절반은 유사도순, 나머지는 조화평균 점수 기반으로 선택한다."""
    if table_top_k <= 0 or not candidates:
        return []

    unique_candidates: List[TableMatch] = []
    seen_names: Set[str] = set()
    for candidate in candidates:
        if candidate.name in seen_names:
            continue
        seen_names.add(candidate.name)
        unique_candidates.append(candidate)

    if len(unique_candidates) <= table_top_k:
        return unique_candidates

    primary_count = table_top_k // 2
    if table_top_k > 0 and primary_count == 0:
        primary_count = 1
    secondary_count = table_top_k - primary_count

    similarity_selected = unique_candidates[:primary_count]
    if secondary_count <= 0:
        return similarity_selected

    remaining_candidates = unique_candidates[primary_count:]
    if not remaining_candidates:
        return similarity_selected

    max_similarity = max(candidate.score for candidate in unique_candidates) or 0
    similarity_normalizer = max_similarity if max_similarity > 0 else 1

    scored_candidates = []
    for candidate in remaining_candidates:
        importance_score = (
            importance_map.get(candidate.name, {}).get("importance_score", 0) or 0
        )
        importance_norm = (
            importance_score / max_importance_score if max_importance_score else 0
        )
        similarity_norm = candidate.score / similarity_normalizer
        harmonic_score = 0.0
        if similarity_norm > 0 and importance_norm > 0:
            harmonic_score = (2 * similarity_norm * importance_norm) / (
                similarity_norm + importance_norm
            )

        scored_candidates.append(
            {
                "match": candidate,
                "harmonic_score": harmonic_score,
                "similarity_score": candidate.score,
            }
        )

    scored_candidates.sort(
        key=lambda item: (
            -item["harmonic_score"],
            -item["similarity_score"],
            item["match"].name,
        )
    )

    secondary_selection = [
        item["match"] for item in scored_candidates[:secondary_count]
    ]

    if len(secondary_selection) < secondary_count:
        already_selected = {match.name for match in secondary_selection}
        fallback_candidates = [
            candidate
            for candidate in remaining_candidates
            if candidate.name not in already_selected
        ]
        needed = secondary_count - len(secondary_selection)
        secondary_selection.extend(fallback_candidates[:needed])

    result = similarity_selected + secondary_selection
    result.sort(key=lambda m: (m.schema or "", m.name))
    return result
