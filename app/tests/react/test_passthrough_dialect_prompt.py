from app.react.generators.passthrough_dialect_prompt import render_passthrough_dialect_prompt


def test_postgresql_passthrough_rules_do_not_include_mysql_alternative() -> None:
    rendered = render_passthrough_dialect_prompt(
        "Passthrough inner-only mode:\n{{passthrough_dialect_rules}}",
        generation_mode="passthrough_inner_only",
        inner_dbms="postgresql",
    )

    assert "Use PostgreSQL syntax only" in rendered
    assert "MySQL/MariaDB syntax only" not in rendered
    assert "If inner_dbms is mysql" not in rendered


def test_mysql_passthrough_rules_do_not_include_postgresql_alternative() -> None:
    rendered = render_passthrough_dialect_prompt(
        "Passthrough inner-only mode:\n{{passthrough_dialect_rules}}",
        generation_mode="passthrough_inner_only",
        inner_dbms="mysql",
    )

    assert "Use MySQL/MariaDB syntax only" in rendered
    assert "Use PostgreSQL syntax only" not in rendered
    assert "If inner_dbms is postgresql" not in rendered
