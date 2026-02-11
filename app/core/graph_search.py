"""Neo4j graph search for schema retrieval"""
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from app.config import settings


@dataclass
class TableMatch:
    """Table match result from vector search"""
    name: str
    schema: str
    db: str
    description: str
    analyzed_description: str = ""
    score: float = 0.0
    columns: List[Dict[str, Any]] = None


@dataclass
class ColumnMatch:
    """Column match result from vector search"""
    name: str
    table_name: str
    table_schema: str
    db: str
    dtype: str
    description: str
    score: float
    nullable: bool = True


@dataclass
class SubSchema:
    """Subschema for SQL generation"""
    tables: List[TableMatch]
    columns: List[ColumnMatch]
    fk_relationships: List[Dict[str, Any]]
    join_hints: List[str]


class GraphSearcher:
    """Search Neo4j graph for relevant schema elements"""
    
    def __init__(self, session):
        self.session = session
        self.top_k = settings.vector_top_k
        self.max_hops = settings.max_fk_hops
    
    async def search_tables(
        self,
        query_embedding: List[float],
        k: int = None,
        schema_filter: List[str] = None,
        datasource: str | None = None,
    ) -> List[TableMatch]:
        """Search for relevant tables using vector similarity
        
        Args:
            query_embedding: Vector embedding of the query
            k: Number of results to return
            schema_filter: Optional list of schemas to include (e.g., ["dw"] for OLAP only)
        """
        k = k or self.top_k
        
        # Use a higher k for initial fetch if filtering
        fetch_k = k * 3 if schema_filter else k
        
        query = """
        CALL db.index.vector.queryNodes('table_vec_index', $k, $embedding)
        YIELD node, score
        RETURN COALESCE(node.original_name, node.name) AS name,
               node.schema AS schema,
               node.db AS db,
               node.description AS description,
               COALESCE(node.analyzed_description, '') AS analyzed_description,
               score
        ORDER BY score DESC, node.schema ASC, name ASC
        """
        
        result = await self.session.run(query, k=fetch_k, embedding=query_embedding)
        records = await result.data()
        
        matches = [
            TableMatch(
                name=r["name"],
                schema=r["schema"],
                db=r["db"],
                description=r.get("description", ""),
                analyzed_description=r.get("analyzed_description", "") or "",
                score=r["score"]
            )
            for r in records
        ]
        
        # Apply schema filter if provided
        if schema_filter:
            schema_filter_lower = [s.lower() for s in schema_filter]
            matches = [
                m for m in matches
                if m.schema and m.schema.lower() in schema_filter_lower
            ]

        # Apply datasource filter if provided (Neo4j Table.db is treated as datasource key)
        ds = (datasource or "").strip()
        if ds:
            ds_l = ds.lower()
            matches = [m for m in matches if (m.db or "").strip().lower() == ds_l]
        
        return matches[:k]
    
    async def search_columns(
        self,
        query_embedding: List[float],
        k: int = None,
        schema_filter: List[str] = None,
        datasource: str | None = None,
    ) -> List[ColumnMatch]:
        """Search for relevant columns using vector similarity
        
        Args:
            query_embedding: Vector embedding of the query
            k: Number of results to return
            schema_filter: Optional list of schemas to include
        """
        k = k or self.top_k
        
        # Use a higher k for initial fetch if filtering
        fetch_k = k * 3 if schema_filter else k
        
        query = """
        CALL db.index.vector.queryNodes('column_vec_index', $k, $embedding)
        YIELD node, score
        MATCH (t:Table)-[:HAS_COLUMN]->(node)
        RETURN node.name AS name,
               t.name AS table_name,
               t.schema AS table_schema,
               t.db AS db,
               node.dtype AS dtype,
               node.description AS description,
               node.nullable AS nullable,
               score
        ORDER BY score DESC, t.name ASC, node.name ASC
        """
        
        result = await self.session.run(query, k=fetch_k, embedding=query_embedding)
        records = await result.data()
        
        schema_filter_lower = [s.lower() for s in (schema_filter or [])]
        ds = (datasource or "").strip()
        ds_l = ds.lower() if ds else ""

        matches: List[ColumnMatch] = []
        for r in records:
            table_schema = r.get("table_schema")
            db = r.get("db")
            if schema_filter_lower:
                if not table_schema or str(table_schema).lower() not in schema_filter_lower:
                    continue
            if ds_l:
                if not db or str(db).strip().lower() != ds_l:
                    continue
            matches.append(
                ColumnMatch(
                    name=r["name"],
                    table_name=r["table_name"],
                    table_schema=r.get("table_schema") or "",
                    db=r.get("db") or "",
                    dtype=r["dtype"],
                    description=r.get("description", ""),
                    nullable=r.get("nullable", True),
                    score=r["score"],
                )
            )
        
        return matches[:k]
    
    async def find_fk_paths(self, table_keys: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Find foreign key relationships between tables"""
        if len(table_keys) < 2:
            return []
        
        query = """
        WITH $keys AS keys
        MATCH (t1:Table)-[r:FK_TO_TABLE*..3]-(t2:Table)
        WHERE (t1.db + '|' + t1.schema + '|' + t1.name) IN keys
          AND (t2.db + '|' + t2.schema + '|' + t2.name) IN keys
          AND t1 <> t2
        WITH t1, t2, r, size(r) AS path_length
        ORDER BY path_length
        RETURN DISTINCT (t1.schema + '.' + t1.name) AS from_table,
               (t2.schema + '.' + t2.name) AS to_table,
               path_length,
               [rel IN r | type(rel)] AS relationship_types
        LIMIT 20
        """

        keys = [f"{t.get('db','')}|{t.get('schema','')}|{t.get('name','')}" for t in (table_keys or [])]
        result = await self.session.run(query, keys=keys)
        records = await result.data()
        
        return records
    
    async def get_table_columns(self, table_keys: List[Dict[str, str]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get all columns for specified tables (keyed by 'schema.table' lower)"""
        query = """
        UNWIND $tables AS t0
        MATCH (t:Table {db: t0.db, schema: t0.schema, name: t0.name})-[:HAS_COLUMN]->(c:Column)
        RETURN t.schema AS schema,
               t.name AS table_name,
               collect({
                   name: c.name,
                   dtype: c.dtype,
                   nullable: c.nullable,
                   description: c.description
               }) AS columns
        """

        result = await self.session.run(query, tables=table_keys)
        records = await result.data()

        out: Dict[str, List[Dict[str, Any]]] = {}
        for r in records:
            schema = str(r.get("schema") or "")
            name = str(r.get("table_name") or "")
            key = f"{schema}.{name}".strip(".").lower()
            out[key] = r["columns"]
        return out
    
    async def get_fk_details(self, table_keys: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Get detailed foreign key constraints between tables"""
        query = """
        MATCH (t1:Table)-[:HAS_COLUMN]->(c1:Column)-[fk:FK_TO]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
        WHERE (t1.db + '|' + t1.schema + '|' + t1.name) IN $keys
          AND (t2.db + '|' + t2.schema + '|' + t2.name) IN $keys
        RETURN (t1.schema + '.' + t1.name) AS from_table,
               c1.name AS from_column,
               (t2.schema + '.' + t2.name) AS to_table,
               c2.name AS to_column,
               fk.constraint AS constraint_name
        """

        keys = [f"{t.get('db','')}|{t.get('schema','')}|{t.get('name','')}" for t in (table_keys or [])]
        result = await self.session.run(query, keys=keys)
        records = await result.data()
        
        return records
    
    async def build_subschema(
        self,
        query_embedding: List[float],
        top_k_tables: int = None,
        top_k_columns: int = None,
        datasource: str | None = None,
        schema_filter: List[str] | None = None,
    ) -> SubSchema:
        """Build a subschema from vector search and graph traversal"""
        top_k_tables = top_k_tables or self.top_k
        top_k_columns = top_k_columns or self.top_k
        
        # Search tables and columns
        table_matches = await self.search_tables(
            query_embedding, k=top_k_tables, schema_filter=schema_filter, datasource=datasource
        )
        column_matches = await self.search_columns(
            query_embedding, k=top_k_columns, schema_filter=schema_filter, datasource=datasource
        )
        
        # Collect unique tables from both searches
        table_keys_set = set()
        for t in table_matches:
            if t.name and t.schema and t.db:
                table_keys_set.add((t.db, t.schema, t.name))
        for c in column_matches:
            if c.table_name and c.table_schema and c.db:
                table_keys_set.add((c.db, c.table_schema, c.table_name))
        table_keys = [
            {"db": db, "schema": schema, "name": name}
            for (db, schema, name) in sorted(table_keys_set)
        ]
        
        # Get all columns for matched tables
        table_columns = await self.get_table_columns(table_keys)
        
        # Add columns to table matches
        for table in table_matches:
            key = f"{table.schema}.{table.name}".strip(".").lower()
            table.columns = table_columns.get(key, [])
        
        # Find FK relationships
        fk_paths = await self.find_fk_paths(table_keys)
        fk_details = await self.get_fk_details(table_keys)
        
        # Generate join hints
        join_hints = self._generate_join_hints(fk_details)
        
        return SubSchema(
            tables=table_matches,
            columns=column_matches,
            fk_relationships=fk_details,
            join_hints=join_hints
        )
    
    def _generate_join_hints(self, fk_details: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable join hints"""
        hints = []
        for fk in fk_details:
            hint = (
                f"JOIN {fk['to_table']} ON {fk['from_table']}.{fk['from_column']} = "
                f"{fk['to_table']}.{fk['to_column']}"
            )
            hints.append(hint)
        return hints


def format_subschema_for_prompt(subschema: SubSchema) -> str:
    """Format subschema as text for LLM prompt"""
    lines = []
    
    # Tables section
    lines.append("=== Available Tables ===")
    for table in subschema.tables:
        lines.append(f"\nTable: {table.schema}.{table.name}")
        if table.description:
            lines.append(f"  Description: {table.description}")
        if table.columns:
            lines.append("  Columns:")
            for col in table.columns:
                null_str = "NULL" if col.get("nullable") else "NOT NULL"
                desc = col.get("description", "")
                col_line = f"    - {col['name']} ({col['dtype']}, {null_str})"
                if desc:
                    col_line += f" - {desc}"
                lines.append(col_line)
    
    # Foreign keys section
    if subschema.fk_relationships:
        lines.append("\n=== Foreign Key Relationships ===")
        for fk in subschema.fk_relationships:
            lines.append(
                f"  {fk['from_table']}.{fk['from_column']} -> "
                f"{fk['to_table']}.{fk['to_column']}"
            )
    
    # Join hints
    if subschema.join_hints:
        lines.append("\n=== Suggested Joins ===")
        for hint in subschema.join_hints:
            lines.append(f"  {hint}")
    
    return "\n".join(lines)

