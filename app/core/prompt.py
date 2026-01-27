"""LangChain prompts and SQL generation chain"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from app.config import settings


SQL_GENERATION_TEMPLATE = """You are a senior database engineer. Your task is to generate a SINGLE valid SQL SELECT query based on the user's question.

User Question:
{question}

Available Schema:
{schema_text}

Constraints and Rules:
1. Generate ONLY a SELECT statement - no INSERT, UPDATE, DELETE, or DDL
2. Use ONLY the tables and columns listed in the schema above
3. ALWAYS use the EXACT table reference format shown in the schema:
   - If table is shown as "datasource.schema.table", use that exact 3-part format (for MindsDB federated queries)
   - If table is shown as "schema.table", use that 2-part format
4. IMPORTANT: For MindsDB queries, use BACKTICKS (`) to quote identifiers that contain uppercase letters or special characters:
   - Example: posgres.rwis.`AAA` or posgres.`common_db`.`customers`
   - Datasource and schema names can be lowercase without quotes
   - Table and column names with uppercase MUST use backticks
5. Preserve the exact letter case of table/column names provided in the schema; do NOT lowercase identifiers
6. Follow the suggested joins if tables need to be joined
7. Do NOT use CTEs (WITH clauses) unless absolutely necessary
8. Do NOT add SQL comments (-- or /* */)
9. The query will automatically have a LIMIT applied, don't worry about it
10. Use appropriate WHERE clauses to filter data efficiently
11. Prefer simple queries over complex nested subqueries
12. Return properly formatted column aliases for clarity

Additional Context:
{join_hints}

Output Requirements:
- Return ONLY the SQL query
- No explanations, no markdown, no code blocks
- Just the raw SQL statement
- The query should be executable as-is

SQL Query:"""


class SQLChain:
    """SQL generation chain using LangChain"""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.openai_llm_model,
            temperature=0,
            api_key=settings.openai_api_key
        )
        self.prompt = ChatPromptTemplate.from_template(SQL_GENERATION_TEMPLATE)
        self.output_parser = StrOutputParser()
        
        # Build chain
        self.chain = self.prompt | self.llm | self.output_parser
    
    async def generate_sql(
        self,
        question: str,
        schema_text: str,
        join_hints: str = ""
    ) -> str:
        """Generate SQL from natural language question"""
        result = await self.chain.ainvoke({
            "question": question,
            "schema_text": schema_text,
            "join_hints": join_hints or "No specific join hints."
        })
        
        # Clean up the result
        sql = result.strip()
        
        # Remove markdown code blocks if present
        if sql.startswith("```"):
            lines = sql.split("\n")
            sql = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            sql = sql.strip()
        
        # Remove language identifier if present
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
        
        return sql


class PromptImprover:
    """Improve prompts based on feedback"""
    
    def __init__(self):
        self.domain_terms: Dict[str, str] = {}
        self.preferred_joins: List[str] = []
    
    def add_domain_term(self, term: str, table_column: str):
        """Add domain-specific term mapping"""
        self.domain_terms[term.lower()] = table_column
    
    def add_preferred_join(self, join_pattern: str):
        """Add preferred join pattern"""
        if join_pattern not in self.preferred_joins:
            self.preferred_joins.append(join_pattern)
    
    def enhance_question(self, question: str) -> str:
        """Enhance question with domain knowledge"""
        enhanced = question
        for term, mapping in self.domain_terms.items():
            # Simple replacement (can be more sophisticated)
            enhanced = enhanced.replace(term, f"{term} ({mapping})")
        return enhanced

