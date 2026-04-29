from xml.etree import ElementTree as ET

from app.react.context_compactor import compact_build_sql_context_for_prompt


def test_compact_build_sql_context_preserves_xml_and_evidence() -> None:
    raw = """
<tool_result>
  <build_sql_context_result>
    <schema_candidates>
      <tables>
        <table>
          <schema>RWIS</schema>
          <name>RDR15MI_TB</name>
          <score>0.9912</score>
        </table>
      </tables>
    </schema_candidates>
    <light_queries>
      <target_k>3</target_k>
      <query index="1">
        <sql><![CDATA[SELECT "LOG_TIME" FROM "RWIS"."RDR15MI_TB"]]></sql>
        <preview>
          <rows>
            <row index="1">
              <cell column="LOG_TIME"><![CDATA[202511292345]]></cell>
            </row>
          </rows>
          <preview_execution_time_ms>15.2</preview_execution_time_ms>
        </preview>
      </query>
    </light_queries>
  </build_sql_context_result>
</tool_result>
""".strip()

    compact = compact_build_sql_context_for_prompt(raw)

    ET.fromstring(compact)
    assert "RDR15MI_TB" in compact
    assert "LOG_TIME" in compact
    assert 'SELECT "LOG_TIME" FROM "RWIS"."RDR15MI_TB"' in compact
    assert "202511292345" in compact
    assert "<score>" not in compact
    assert "<target_k>" not in compact
    assert "<preview_execution_time_ms>" not in compact
    assert len(compact) < len(raw) * 0.85


def test_compact_build_sql_context_returns_raw_on_parse_failure() -> None:
    raw = "<tool_result><build_sql_context_result><score>0.1</score>"

    assert compact_build_sql_context_for_prompt(raw) == raw
