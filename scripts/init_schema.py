#!/usr/bin/env python3
"""
Initialize Neo4j schema with constraints and indexes.
Run this script after starting Neo4j for the first time.
"""
import asyncio
from neo4j import AsyncGraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))


async def init_schema():
    """Initialize Neo4j schema"""
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
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
        # Regular indexes
        """
        CREATE INDEX table_name_idx IF NOT EXISTS
        FOR (t:Table) ON (t.name)
        """,
        """
        CREATE INDEX column_name_idx IF NOT EXISTS
        FOR (c:Column) ON (c.name)
        """,
        # Vector indexes
        f"""
        CREATE VECTOR INDEX table_vec_index IF NOT EXISTS
        FOR (t:Table) ON (t.vector)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """,
        f"""
        CREATE VECTOR INDEX column_vec_index IF NOT EXISTS
        FOR (c:Column) ON (c.vector)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        ,
        # Query history constraints / indexes (for similar query search)
        """
        CREATE CONSTRAINT query_id IF NOT EXISTS
        FOR (q:Query) REQUIRE q.id IS UNIQUE
        """
        ,
        f"""
        CREATE VECTOR INDEX query_question_vec_index IF NOT EXISTS
        FOR (q:Query) ON (q.vector_question)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
        ,
        f"""
        CREATE VECTOR INDEX query_intent_vec_index IF NOT EXISTS
        FOR (q:Query) ON (q.vector_intent)
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIMENSION},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    ]
    
    async with driver.session() as session:
        for i, query in enumerate(queries, 1):
            try:
                print(f"[{i}/{len(queries)}] Executing: {query.strip()[:60]}...")
                await session.run(query)
                print(f"  ✓ Success")
            except Exception as e:
                print(f"  ⚠ Warning: {e}")
    
    await driver.close()
    print("\n✅ Schema initialization completed!")


if __name__ == "__main__":
    asyncio.run(init_schema())

