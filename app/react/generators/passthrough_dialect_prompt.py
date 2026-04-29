from __future__ import annotations

from typing import Optional


PASSTHROUGH_DIALECT_RULES_TOKEN = "{{passthrough_dialect_rules}}"


def passthrough_dialect_rules(*, generation_mode: Optional[str], inner_dbms: Optional[str]) -> str:
    mode = (generation_mode or "").strip().lower()
    dialect = (inner_dbms or "").strip().lower()
    if mode != "passthrough_inner_only":
        return (
            "- If generation_mode is not passthrough_inner_only, generate SQL for the provided dbms.\n"
            "- Do not output any MindsDB wrapper unless the caller explicitly asks for it."
        )

    if dialect in {"postgresql", "postgres", "pg"}:
        return (
            "- NEVER output the outer MindsDB wrapper.\n"
            "  - Do NOT output: SELECT * FROM `datasource` ( ... )\n"
            "  - Output ONLY the SQL that belongs inside the parentheses.\n"
            "- Use PostgreSQL syntax only.\n"
            "- Use double-quoted identifiers such as \"RWIS\".\"RDITAG_TB\" and alias.\"TAGSN\".\n"
            "- NEVER use MySQL-only functions such as DATE_FORMAT, STR_TO_DATE, DATE_SUB, CURDATE, or IFNULL.\n"
            "- The system will wrap the returned inner SQL statically for MindsDB execution."
        )

    if dialect in {"mysql", "mariadb"}:
        return (
            "- NEVER output the outer MindsDB wrapper.\n"
            "  - Do NOT output: SELECT * FROM `datasource` ( ... )\n"
            "  - Output ONLY the SQL that belongs inside the parentheses.\n"
            "- Use MySQL/MariaDB syntax only.\n"
            "- The system will wrap the returned inner SQL statically for MindsDB execution."
        )

    return (
        "- NEVER output the outer MindsDB wrapper.\n"
        "  - Do NOT output: SELECT * FROM `datasource` ( ... )\n"
        "  - Output ONLY the SQL that belongs inside the parentheses.\n"
        "- Use only the inner SQL dialect provided by inner_dbms.\n"
        "- The system will wrap the returned inner SQL statically for MindsDB execution."
    )


def render_passthrough_dialect_prompt(
    prompt_template: str,
    *,
    generation_mode: Optional[str],
    inner_dbms: Optional[str],
) -> str:
    rules = passthrough_dialect_rules(generation_mode=generation_mode, inner_dbms=inner_dbms)
    return (prompt_template or "").replace(PASSTHROUGH_DIALECT_RULES_TOKEN, rules)
