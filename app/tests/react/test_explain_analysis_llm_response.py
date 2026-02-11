from app.react.generators.explain_analysis_generator import ExplainAnalysisLLMResponse


def test_explain_analysis_from_xml_repairs_text_fields_with_angle_brackets() -> None:
    # '<' in free text can break XML; the parser should still extract the verdict/reason/fixes.
    xml_text = """<explain_verdict>
  <verdict>FAIL</verdict>
  <reason>Need a &lt;schema&gt; tag and 2 &lt; 3 comparisons can appear</reason>
  <suggested_fixes>
    <fix>Add a date filter where 2 &lt; 3 is present in predicates</fix>
  </suggested_fixes>
</explain_verdict>
"""

    parsed = ExplainAnalysisLLMResponse.from_xml(xml_text)
    assert parsed.verdict == "FAIL"
    assert "<schema>" in parsed.reason
    assert "2 < 3" in parsed.reason
    assert len(parsed.suggested_fixes) == 1
    assert "2 < 3" in parsed.suggested_fixes[0]


