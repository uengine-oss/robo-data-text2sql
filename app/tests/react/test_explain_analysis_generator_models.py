# python -m pytest app/tests/react/test_explain_analysis_generator_models.py -v

"""Tests for ExplainAnalysis generator models."""

from typing import Any, Dict, List

from app.react.generators.explain_analysis_generator import (
    ExplainAnalysisLLMResponse,
    ExplainAnalysisResult,
)
from app.react.utils.db_query_builder.type import TableIndexMetadata, TableMetadata, ExecutionPlanResult


def _sample_plan() -> ExecutionPlanResult:
    return ExecutionPlanResult(
        total_cost=10.5,
        execution_time_ms=2.2,
        row_count=100,
        raw_plan={"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders"}},
    )


def test_llm_response_parser_handles_explain_verdict():
    """LLM XML parser should produce PASS/FAIL verdict without validation queries."""
    xml = """
    <explain_verdict>
        <verdict>FAIL</verdict>
        <reason>Potential seq scan risk.</reason>
        <suggested_fixes>
            <fix>Add a date filter.</fix>
            <fix>Add a selective predicate.</fix>
        </suggested_fixes>
    </explain_verdict>
    """

    parsed = ExplainAnalysisLLMResponse.from_xml(xml)
    assert parsed.verdict == "FAIL"
    assert parsed.reason.startswith("Potential")
    assert len(parsed.suggested_fixes) == 2


def test_result_to_xml_contains_required_sections():
    """ExplainAnalysisResult.to_xml_str should include verdict and omit validation queries."""
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
        verdict="FAIL",
        fail_reason="Seq scan on orders may be slow.",
        suggested_fixes=["Add a date filter"],
        llm_raw_response="<explain_verdict />",
    )

    xml_payload = result.to_xml_str()

    assert "<input_sql>" in xml_payload
    assert "<execution_plan>" in xml_payload
    assert "<table_metadata>" in xml_payload
    assert "<explain_verdict>" in xml_payload
    assert "<validation_queries>" not in xml_payload
    assert "orders_pkey" in xml_payload

