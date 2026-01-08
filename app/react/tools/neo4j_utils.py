from typing import Any, Dict, List, Optional, Set, Tuple

from neo4j import AsyncSession


RELATIONSHIP_RANKS = {
    "HAS_COLUMN → FK_TO → HAS_COLUMN": 100,
}
DEFAULT_RELATIONSHIP_SCORE = 1

RELATIONSHIP_TYPE_MAP = {
    "HAS_COLUMN → FK_TO → HAS_COLUMN": "외래키 관계",
}


async def get_table_importance_scores(
    neo4j_session: AsyncSession,
) -> Dict[str, Dict[str, Any]]:
    """Neo4j에서 모든 테이블의 중요도를 미리 계산한다."""
    query = """
    MATCH (t:Table)
    OPTIONAL MATCH (t)-[r]-()
    WITH t, count(r) AS total_relations
    RETURN t.name AS table_name,
           t.schema AS schema,
           total_relations AS importance_score,
           t.description AS description
    ORDER BY importance_score DESC
    """

    result = await neo4j_session.run(query)
    records = await result.data()

    importance_map: Dict[str, Dict[str, Any]] = {}
    for record in records:
        table_name = record["table_name"]
        importance_map[table_name] = {
            "schema": record.get("schema"),
            "description": record.get("description"),
            "importance_score": record.get("importance_score", 0),
        }

    return importance_map


async def get_table_fk_relationships(
    neo4j_session: AsyncSession,
    table_name: str,
    limit: int,
    schema: Optional[str] = None,
) -> List[Dict]:
    """
    특정 테이블과 연관된 다른 테이블 정보를 조회한다.
    기존 run_mocked_tools 로직을 재사용한다.
    """
    query = """
    MATCH (t:Table)
    WHERE (
      (t.name IS NOT NULL AND toLower(t.name) = toLower($table_name))
      OR (t.original_name IS NOT NULL AND toLower(t.original_name) = toLower($table_name))
    )
      AND ($schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = toLower($schema)))
    MATCH (t)-[:HAS_COLUMN]->(c1:Column)-[fk:FK_TO]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
    RETURN DISTINCT COALESCE(t2.original_name, t2.name) AS related_table,
           t2.schema AS related_table_schema,
           t2.description AS related_table_description,
           'foreign_key' AS relation_type,
           c1.name AS from_column,
           c1.description AS from_column_description,
           c2.name AS to_column,
           c2.description AS to_column_description
    ORDER BY related_table, c1.name, c2.name
    LIMIT $limit
    """

    result = await neo4j_session.run(
        query, table_name=table_name, schema=schema, limit=limit
    )
    records = await result.data()

    relationships: List[Dict] = []
    for record in records:
        rel_info: Dict = {
            "related_table": record["related_table"],
            "related_table_schema": record.get("related_table_schema"),
            "relation_type": record["relation_type"],
            "from_column": record.get("from_column"),
            "to_column": record.get("to_column"),
        }
        if record.get("related_table_description"):
            rel_info["related_table_description"] = record["related_table_description"]
        if record.get("from_column_description"):
            rel_info["from_column_description"] = record["from_column_description"]
        if record.get("to_column_description"):
            rel_info["to_column_description"] = record["to_column_description"]
        relationships.append(rel_info)

    if len(relationships) < limit:
        reverse_query = """
        MATCH (t:Table)
        WHERE (
          (t.name IS NOT NULL AND toLower(t.name) = toLower($table_name))
          OR (t.original_name IS NOT NULL AND toLower(t.original_name) = toLower($table_name))
        )
          AND ($schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = toLower($schema)))
        MATCH (t2:Table)-[:HAS_COLUMN]->(c2:Column)-[fk:FK_TO]->(c1:Column)<-[:HAS_COLUMN]-(t)
        RETURN DISTINCT COALESCE(t2.original_name, t2.name) AS related_table,
               t2.schema AS related_table_schema,
               t2.description AS related_table_description,
               'referenced_by' AS relation_type,
               c1.name AS from_column,
               c1.description AS from_column_description,
               c2.name AS to_column,
               c2.description AS to_column_description
        ORDER BY related_table, c1.name, c2.name
        LIMIT $limit
        """

        reverse_result = await neo4j_session.run(
            reverse_query,
            table_name=table_name,
            schema=schema,
            limit=limit - len(relationships),
        )
        reverse_records = await reverse_result.data()

        for record in reverse_records:
            rel_info = {
                "related_table": record["related_table"],
                "related_table_schema": record.get("related_table_schema"),
                "relation_type": record["relation_type"],
                "from_column": record.get("from_column"),
                "to_column": record.get("to_column"),
            }
            if record.get("related_table_description"):
                rel_info["related_table_description"] = record["related_table_description"]
            if record.get("from_column_description"):
                rel_info["from_column_description"] = record["from_column_description"]
            if record.get("to_column_description"):
                rel_info["to_column_description"] = record["to_column_description"]
            relationships.append(rel_info)

    relationships.sort(
        key=lambda r: (
            r.get("related_table_schema") or "",
            r.get("related_table") or "",
            r.get("from_column") or "",
            r.get("to_column") or "",
            r.get("relation_type") or "",
        )
    )
    return relationships


async def get_table_any_relationships(
    neo4j_session: AsyncSession,
    table_name: str,
    schema: Optional[str] = None,
) -> List[Dict]:
    """특정 테이블과 최대 세 단계 이내에 연결된 다양한 관계를 조회한다."""
    query = """
    MATCH (t1:Table)
    WHERE (
      (t1.name IS NOT NULL AND toLower(t1.name) = toLower($table_name))
      OR (t1.original_name IS NOT NULL AND toLower(t1.original_name) = toLower($table_name))
    )
      AND ($schema IS NULL OR (t1.schema IS NOT NULL AND toLower(t1.schema) = toLower($schema)))
    MATCH path = (t1)-[*1..3]-(t2:Table)
    WHERE t1 <> t2
    WITH t2,
         collect(DISTINCT [rel IN relationships(path) | type(rel)]) AS relationship_paths
    RETURN COALESCE(t2.original_name, t2.name) AS related_table,
           t2.schema AS related_table_schema,
           COALESCE(t2.comment, t2.description, '설명 없음') AS related_table_description,
           [path_types IN relationship_paths |
               REDUCE(
                   acc = '',
                   rel IN path_types |
                   CASE
                       WHEN acc = '' THEN rel
                       ELSE acc + ' → ' + rel
                   END
               )
           ] AS relationship_paths
    ORDER BY related_table
    LIMIT 100
    """

    result = await neo4j_session.run(
        query,
        table_name=table_name,
        schema=schema,
    )
    records = await result.data()

    relationships: List[Dict] = []
    for record in records:
        rel_info: Dict = {
            "related_table": record["related_table"],
            "related_table_schema": record.get("related_table_schema"),
            "relationship_paths": record.get("relationship_paths") or [],
        }
        if record.get("related_table_description"):
            rel_info["related_table_description"] = record["related_table_description"]
        relationships.append(rel_info)

    return relationships


async def get_table_relationship_details(
    neo4j_session: AsyncSession,
    table_name: str,
    relation_limit: int,
    schema: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    """특정 테이블과 연관된 FK 및 기타 관계 정보를 묶어서 반환한다."""
    if relation_limit <= 0:
        return {"fk_relationships": [], "additional_relationships": []}

    fk_relationships = await get_table_fk_relationships(
        neo4j_session,
        table_name,
        schema=schema,
        limit=relation_limit,
    )
    fk_related_tables: Set[Tuple[str, str]] = set()
    for rel in fk_relationships:
        related_table = rel.get("related_table")
        if not related_table:
            continue
        related_schema = rel.get("related_table_schema") or ""
        fk_related_tables.add((related_schema, related_table))

    remaining_relationship_slots = max(relation_limit - len(fk_relationships), 0)
    additional_relationships: List[Dict[str, Any]] = []

    if remaining_relationship_slots:
        fallback_candidates = await get_table_any_relationships(
            neo4j_session,
            table_name,
            schema=schema,
        )
        scored_candidates: List[Dict[str, Any]] = []
        for candidate in fallback_candidates:
            candidate_name = candidate.get("related_table")
            candidate_schema = candidate.get("related_table_schema") or ""
            relationship_paths = candidate.get("relationship_paths") or []
            if (
                not candidate_name
                or (candidate_schema, candidate_name) in fk_related_tables
                or not relationship_paths
            ):
                continue

            score = 0
            relationship_type_labels: List[str] = []
            seen_type_labels: Set[str] = set()
            for path in relationship_paths:
                score += RELATIONSHIP_RANKS.get(path, DEFAULT_RELATIONSHIP_SCORE)
                type_label = RELATIONSHIP_TYPE_MAP.get(path)
                if type_label and type_label not in seen_type_labels:
                    seen_type_labels.add(type_label)
                    relationship_type_labels.append(type_label)

            scored_candidates.append(
                {
                    "related_table": candidate_name,
                    "related_table_schema": candidate.get("related_table_schema"),
                    "related_table_description": candidate.get("related_table_description"),
                    "relationship_type": ", ".join(relationship_type_labels)
                    if relationship_type_labels
                    else None,
                    "score": score,
                }
            )

        scored_candidates.sort(
            key=lambda item: (-item["score"], item["related_table"])
        )
        additional_relationships = scored_candidates[:remaining_relationship_slots]

    return {
        "fk_relationships": fk_relationships,
        "additional_relationships": additional_relationships,
    }


async def get_column_fk_relationships(
    neo4j_session: AsyncSession,
    table_name: str,
    column_name: str,
    limit: int,
    schema: Optional[str] = None,
) -> List[Dict]:
    """특정 컬럼의 외래키 관계를 조회한다."""
    query = """
    MATCH (t:Table)
    WHERE (
      (t.name IS NOT NULL AND toLower(t.name) = toLower($table_name))
      OR (t.original_name IS NOT NULL AND toLower(t.original_name) = toLower($table_name))
    )
      AND ($schema IS NULL OR (t.schema IS NOT NULL AND toLower(t.schema) = toLower($schema)))
    MATCH (t)-[:HAS_COLUMN]->(c1:Column {name: $column_name})-[fk:FK_TO]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
    RETURN COALESCE(t2.original_name, t2.name) AS referenced_table,
           t2.schema AS referenced_table_schema,
           t2.description AS referenced_table_description,
           c2.name AS referenced_column,
           c2.description AS referenced_column_description,
           fk.constraint AS constraint_name
    ORDER BY referenced_table, c2.name
    LIMIT $limit
    """

    result = await neo4j_session.run(
        query,
        table_name=table_name,
        column_name=column_name,
        schema=schema,
        limit=limit,
    )
    records = await result.data()

    fk_relationships: List[Dict] = []
    for record in records:
        fk_info: Dict = {
            "referenced_table": record["referenced_table"],
            "referenced_column": record["referenced_column"],
        }
        if record.get("referenced_table_schema"):
            fk_info["referenced_table_schema"] = record["referenced_table_schema"]
        if record.get("referenced_table_description"):
            fk_info["referenced_table_description"] = record["referenced_table_description"]
        if record.get("referenced_column_description"):
            fk_info["referenced_column_description"] = record["referenced_column_description"]
        if record.get("constraint_name"):
            fk_info["constraint_name"] = record["constraint_name"]
        fk_relationships.append(fk_info)

    return fk_relationships

