# python -m pytest app/tests/react/test_explain_analysis_generator_models.py -v

"""Tests for ExplainAnalysis generator models."""

from typing import Any, Dict, List

from app.react.generators.explain_analysis_generator import (
    ExplainAnalysisLLMResponse,
    ExplainAnalysisResult,
    ValidationQuery,
    ValidationQueryResult,
)
from app.react.utils.db_query_builder.type import TableIndexMetadata, TableMetadata, ExecutionPlanResult


def _sample_plan() -> ExecutionPlanResult:
    return ExecutionPlanResult(
        total_cost=10.5,
        execution_time_ms=2.2,
        row_count=100,
        raw_plan={"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders"}},
    )


def _sample_validation_results() -> List[ValidationQueryResult]:
    query = ValidationQuery(query_id="1", reason="Check row counts", sql="SELECT 1;")
    return [
        ValidationQueryResult(
            query=query,
            success=True,
            row_count=1,
            execution_time_ms=1.5,
            columns=["count"],
            rows=[[123]],
        )
    ]


def test_llm_response_parser_handles_queries():
    """LLM XML parser should produce structured response."""
    xml = """
    <validation_plan>
        <risk_analysis>
            <summary>Potential seq scan risk.</summary>
        </risk_analysis>
        <queries>
            <query id="A1">
                <reason>Verify filter selectivity.</reason>
                <sql>SELECT COUNT(*) FROM "public"."orders";</sql>
            </query>
            <query>
                <reason>Fallback</reason>
                <sql>SELECT COUNT(*) FROM "public"."customers";</sql>
            </query>
        </queries>
    </validation_plan>
    """

    parsed = ExplainAnalysisLLMResponse.from_xml(xml)

    assert parsed.risk_summary.startswith("Potential")
    assert len(parsed.validation_queries) == 2
    assert parsed.validation_queries[0].query_id == "A1"
    assert "customers" in parsed.validation_queries[1].sql


def test_result_to_xml_contains_required_sections():
    """ExplainAnalysisResult.to_xml_str should include all sections."""
    metadata = [
        TableMetadata(
            schema_name="public",
            table_name="orders",
            row_count=42,
            indexes=[
                TableIndexMetadata(
                    index_name="orders_pkey",
                    is_unique=True,
                    columns=["id"],
                    definition="CREATE INDEX orders_pkey ...",
                )
            ],
        )
    ]

    result = ExplainAnalysisResult(
        input_sql='SELECT * FROM "public"."orders";',
        execution_plan=_sample_plan(),
        table_metadata=metadata,
        risk_analysis_summary="Seq scan on orders may be slow.",
        validation_results=_sample_validation_results(),
        llm_raw_response="<validation_plan />",
    )

    xml_payload = result.to_xml_str()

    assert "<input_sql>" in xml_payload
    assert "<execution_plan>" in xml_payload
    assert "<table_metadata>" in xml_payload
    assert "<risk_analysis>" in xml_payload
    assert "<validation_queries>" in xml_payload
    assert "orders_pkey" in xml_payload

