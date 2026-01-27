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
    score: float
    datasource: str = None  # 데이터 소스 이름 (MindsDB에서 사용)
    columns: List[Dict[str, Any]] = None


@dataclass
class ColumnMatch:
    """Column match result from vector search"""
    name: str
    table_name: str
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
        schema_filter: List[str] = None
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
               node.datasource AS datasource,
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
                score=r["score"],
                datasource=r.get("datasource")
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
        
        return matches[:k]
    
    async def search_columns(
        self,
        query_embedding: List[float],
        k: int = None,
        schema_filter: List[str] = None
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
               node.dtype AS dtype,
               node.description AS description,
               node.nullable AS nullable,
               score
        ORDER BY score DESC, t.name ASC, node.name ASC
        """
        
        result = await self.session.run(query, k=fetch_k, embedding=query_embedding)
        records = await result.data()
        
        matches = [
            ColumnMatch(
                name=r["name"],
                table_name=r["table_name"],
                dtype=r["dtype"],
                description=r.get("description", ""),
                nullable=r.get("nullable", True),
                score=r["score"]
            )
            for r in records
            if not schema_filter or (
                r.get("table_schema") and 
                r.get("table_schema").lower() in [s.lower() for s in schema_filter]
            )
        ]
        
        return matches[:k]
    
    async def find_fk_paths(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """Find foreign key relationships between tables"""
        if len(table_names) < 2:
            return []
        
        query = """
        MATCH (t1:Table)-[r:FK_TO_TABLE*..3]-(t2:Table)
        WHERE t1.name IN $tables AND t2.name IN $tables AND t1 <> t2
        WITH t1, t2, r, size(r) AS path_length
        ORDER BY path_length
        RETURN DISTINCT t1.name AS from_table,
               t2.name AS to_table,
               path_length,
               [rel IN r | type(rel)] AS relationship_types
        LIMIT 20
        """
        
        result = await self.session.run(query, tables=table_names)
        records = await result.data()
        
        return records
    
    async def get_table_columns(self, table_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Get all columns for specified tables"""
        query = """
        MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
        WHERE t.name IN $tables
        RETURN t.name AS table_name,
               collect({
                   name: c.name,
                   dtype: c.dtype,
                   nullable: c.nullable,
                   description: c.description
               }) AS columns
        """
        
        result = await self.session.run(query, tables=table_names)
        records = await result.data()
        
        return {r["table_name"]: r["columns"] for r in records}
    
    async def get_fk_details(self, table_names: List[str]) -> List[Dict[str, Any]]:
        """Get detailed foreign key constraints between tables"""
        query = """
        MATCH (t1:Table)-[:HAS_COLUMN]->(c1:Column)-[fk:FK_TO]->(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
        WHERE t1.name IN $tables AND t2.name IN $tables
        RETURN t1.name AS from_table,
               t1.schema AS from_schema,
               t1.datasource AS from_datasource,
               c1.name AS from_column,
               t2.name AS to_table,
               t2.schema AS to_schema,
               t2.datasource AS to_datasource,
               c2.name AS to_column,
               fk.constraint AS constraint_name
        """
        
        result = await self.session.run(query, tables=table_names)
        records = await result.data()
        
        return records
    
    async def build_subschema(
        self,
        query_embedding: List[float],
        top_k_tables: int = None,
        top_k_columns: int = None
    ) -> SubSchema:
        """Build a subschema from vector search and graph traversal"""
        top_k_tables = top_k_tables or self.top_k
        top_k_columns = top_k_columns or self.top_k
        
        # Search tables and columns
        table_matches = await self.search_tables(query_embedding, k=top_k_tables)
        column_matches = await self.search_columns(query_embedding, k=top_k_columns)
        
        # Collect unique tables from both searches
        table_names = list(set(
            [t.name for t in table_matches] +
            [c.table_name for c in column_matches]
        ))
        
        # Get all columns for matched tables
        table_columns = await self.get_table_columns(table_names)
        
        # Add columns to table matches
        for table in table_matches:
            table.columns = table_columns.get(table.name, [])
        
        # Find FK relationships
        fk_paths = await self.find_fk_paths(table_names)
        fk_details = await self.get_fk_details(table_names)
        
        # Generate join hints
        join_hints = self._generate_join_hints(fk_details)
        
        return SubSchema(
            tables=table_matches,
            columns=column_matches,
            fk_relationships=fk_details,
            join_hints=join_hints
        )
    
    def _generate_join_hints(self, fk_details: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable join hints with datasource prefix"""
        hints = []
        for fk in fk_details:
            # 데이터 소스가 있으면 datasource.schema.table 형식 사용
            from_ds = fk.get('from_datasource')
            from_schema = fk.get('from_schema', '')
            to_ds = fk.get('to_datasource')
            to_schema = fk.get('to_schema', '')
            
            if from_ds:
                from_table_ref = f"{from_ds}.{from_schema}.{fk['from_table']}"
            else:
                from_table_ref = f"{from_schema}.{fk['from_table']}" if from_schema else fk['from_table']
            
            if to_ds:
                to_table_ref = f"{to_ds}.{to_schema}.{fk['to_table']}"
            else:
                to_table_ref = f"{to_schema}.{fk['to_table']}" if to_schema else fk['to_table']
            
            hint = (
                f"JOIN {to_table_ref} ON {from_table_ref}.{fk['from_column']} = "
                f"{to_table_ref}.{fk['to_column']}"
            )
            hints.append(hint)
        return hints


def format_subschema_for_prompt(subschema: SubSchema) -> str:
    """Format subschema as text for LLM prompt"""
    lines = []
    
    # Tables section
    lines.append("=== Available Tables ===")
    for table in subschema.tables:
        # 데이터 소스가 있으면 datasource.schema.table 형식 사용
        if table.datasource:
            table_ref = f"{table.datasource}.{table.schema}.{table.name}"
        else:
            table_ref = f"{table.schema}.{table.name}"
        lines.append(f"\nTable: {table_ref}")
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
            # 데이터 소스가 있으면 포함
            from_ds = fk.get('from_datasource')
            from_schema = fk.get('from_schema', '')
            to_ds = fk.get('to_datasource')
            to_schema = fk.get('to_schema', '')
            
            if from_ds:
                from_ref = f"{from_ds}.{from_schema}.{fk['from_table']}"
            else:
                from_ref = f"{from_schema}.{fk['from_table']}" if from_schema else fk['from_table']
            
            if to_ds:
                to_ref = f"{to_ds}.{to_schema}.{fk['to_table']}"
            else:
                to_ref = f"{to_schema}.{fk['to_table']}" if to_schema else fk['to_table']
            
            lines.append(
                f"  {from_ref}.{fk['from_column']} -> "
                f"{to_ref}.{fk['to_column']}"
            )
    
    # Join hints
    if subschema.join_hints:
        lines.append("\n=== Suggested Joins ===")
        for hint in subschema.join_hints:
            lines.append(f"  {hint}")
    
    return "\n".join(lines)

