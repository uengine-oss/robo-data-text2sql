"""Load extracted schema into Neo4j graph"""
from typing import List, Dict, Any
import asyncio

from app.config import settings
from app.core.embedding import EmbeddingClient


class Neo4jSchemaLoader:
    """Load schema metadata into Neo4j graph database"""
    
    def __init__(self, session, embedding_client: EmbeddingClient):
        self.session = session
        self.embedding_client = embedding_client
    
    def _normalize_nullable(self, value: Any) -> bool:
        """Normalize nullable field into boolean"""
        if value is None:
            return True
        if isinstance(value, bool):
            return value
        
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"no", "false", "n", "0"}:
                return False
            if normalized in {"yes", "true", "y", "1"}:
                return True
        
        return bool(value)
    
    async def setup_constraints_and_indexes(self):
        """Create Neo4j constraints and vector indexes"""
        queries = [
            # Constraints
            """
            CREATE CONSTRAINT table_key IF NOT EXISTS
            FOR (t:Table) REQUIRE (t.db, t.schema, t.name) IS NODE KEY
            """,
            """
            CREATE CONSTRAINT column_fqn IF NOT EXISTS
            FOR (c:Column) REQUIRE c.fqn IS UNIQUE
            """,
            # Vector indexes
            """
            CREATE VECTOR INDEX table_vec_index IF NOT EXISTS
            FOR (t:Table) ON (t.vector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """,
            """
            CREATE VECTOR INDEX column_vec_index IF NOT EXISTS
            FOR (c:Column) ON (c.vector)
            OPTIONS {
                indexConfig: {
                    `vector.dimensions`: $dimensions,
                    `vector.similarity_function`: 'cosine'
                }
            }
            """
        ]
        
        for query in queries:
            try:
                await self.session.run(query, dimensions=settings.embedding_dimension)
            except Exception as e:
                # Constraint/index might already exist
                print(f"Setup warning: {e}")
    
    async def load_tables(self, tables: List[Dict[str, Any]], db_name: str = "postgres"):
        """Load tables into Neo4j with embeddings"""
        for table in tables:
            # Normalize table name to lowercase for consistent MERGE
            table_name_lower = table["name"].lower()
            table_name_original = table["name"]
            schema_lower = table["schema"].lower()
            
            # Generate embedding
            description = table.get("description") or ""
            text = self.embedding_client.format_table_text(
                table_name=table_name_original,
                description=description,
                columns=[]  # Will add later
            )
            embedding = await self.embedding_client.embed_text(text)
            
            # Create table node - use lowercase name for MERGE to match existing nodes
            query = """
            MERGE (t:Table {db: $db, schema: $schema, name: $name})
            SET t.vector = $vector,
                t.description = COALESCE(t.description, $description),
                t.original_name = $original_name,
                t.updated_at = datetime()
            RETURN t
            """
            
            await self.session.run(
                query,
                db=db_name,
                schema=schema_lower,
                name=table_name_lower,
                vector=embedding,
                description=description,
                original_name=table_name_original
            )
        
        print(f"Loaded {len(tables)} tables")
    
    async def load_columns(self, columns: List[Dict[str, Any]], db_name: str = "postgres"):
        """Load columns into Neo4j with embeddings"""
        # Batch process embeddings
        texts = [
            self.embedding_client.format_column_text(
                column_name=col["name"],
                table_name=col["table_name"],
                dtype=col["dtype"],
                description=col.get("description", "")
            )
            for col in columns
        ]
        
        # Generate embeddings in batches
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings = await self.embedding_client.embed_batch(batch)
            all_embeddings.extend(embeddings)
            print(f"Generated embeddings for columns {i} to {i+len(batch)}")
        
        # Load columns
        for col, embedding in zip(columns, all_embeddings):
            # fqn = f"{db_name}.{col['schema']}.{col['table_name']}.{col['name']}"
            # Standardize FQN: schema.table.column in lowercase (no db prefix)
            fqn = f"{col['schema']}.{col['table_name']}.{col['name']}".lower()
            nullable = self._normalize_nullable(col.get("nullable"))
            
            # Normalize table name and schema to lowercase for MATCH
            table_name_lower = col["table_name"].lower()
            schema_lower = col["schema"].lower()
            
            query = """
            MATCH (t:Table {db: $db, schema: $schema, name: $table_name})
            MERGE (c:Column {fqn: $fqn})
            SET c.vector = $vector,
                c.name = $column_name,
                c.dtype = $dtype,
                c.description = COALESCE(c.description, $description),
                c.nullable = $nullable,
                c.updated_at = datetime()
            MERGE (t)-[:HAS_COLUMN]->(c)
            RETURN c
            """
            
            await self.session.run(
                query,
                db=db_name,
                schema=schema_lower,
                table_name=table_name_lower,
                fqn=fqn,
                vector=embedding,
                column_name=col["name"],
                dtype=col.get("dtype") or "",
                description=col.get("description") or "",
                nullable=nullable
            )
        
        print(f"Loaded {len(columns)} columns")
    
    async def load_foreign_keys(self, foreign_keys: List[Dict[str, Any]], db_name: str = "postgres"):
        """Load foreign key relationships"""
        for fk in foreign_keys:
            # from_fqn = f"{db_name}.{fk['from_schema']}.{fk['from_table']}.{fk['from_column']}"
            # to_fqn = f"{db_name}.{fk['to_schema']}.{fk['to_table']}.{fk['to_column']}"
            # Use standardized FQN (no db prefix, lowercase)
            from_fqn = f"{fk['from_schema']}.{fk['from_table']}.{fk['from_column']}".lower()
            to_fqn = f"{fk['to_schema']}.{fk['to_table']}.{fk['to_column']}".lower()
            
            # Column-to-column FK
            query = """
            MATCH (c1:Column {fqn: $from_fqn})
            MATCH (c2:Column {fqn: $to_fqn})
            MERGE (c1)-[fk:FK_TO]->(c2)
            SET fk.constraint = $constraint_name,
                fk.on_update = $on_update,
                fk.on_delete = $on_delete
            """
            
            await self.session.run(
                query,
                from_fqn=from_fqn,
                to_fqn=to_fqn,
                constraint_name=fk["constraint_name"],
                on_update=fk.get("on_update", "NO ACTION"),
                on_delete=fk.get("on_delete", "NO ACTION")
            )
            
            # Table-to-table FK (for easier path finding)
            query2 = """
            MATCH (t1:Table {db: $db, schema: $from_schema, name: $from_table})
            MATCH (t2:Table {db: $db, schema: $to_schema, name: $to_table})
            MERGE (t1)-[:FK_TO_TABLE]->(t2)
            """
            
            await self.session.run(
                query2,
                db=db_name,
                from_schema=fk["from_schema"].lower(),
                from_table=fk["from_table"].lower(),
                to_schema=fk["to_schema"].lower(),
                to_table=fk["to_table"].lower()
            )
        
        print(f"Loaded {len(foreign_keys)} foreign keys")
    
    async def load_primary_keys(self, primary_keys: List[Dict[str, Any]], db_name: str = "postgres"):
        """Mark primary key columns"""
        for pk in primary_keys:
            # fqn = f"{db_name}.{pk['schema']}.{pk['table_name']}.{pk['column_name']}"
            # Use standardized FQN (no db prefix, lowercase)
            fqn = f"{pk['schema']}.{pk['table_name']}.{pk['column_name']}".lower()
            
            query = """
            MATCH (c:Column {fqn: $fqn})
            SET c.is_primary_key = true,
                c.pk_constraint = $constraint_name
            """
            
            await self.session.run(
                query,
                fqn=fqn,
                constraint_name=pk["constraint_name"]
            )
        
        print(f"Marked {len(primary_keys)} primary key columns")
    
    async def clear_schema(self, db_name: str = "postgres"):
        """Clear existing schema data for a database"""
        query = """
        MATCH (n)
        WHERE (n:Table OR n:Column) AND (n.db = $db OR n.db IS NULL)
        DETACH DELETE n
        """
        
        result = await self.session.run(query, db=db_name)
        summary = await result.consume()
        print(f"Cleared existing schema nodes: {summary.counters.nodes_deleted}")

