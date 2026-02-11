import json
from typing import List


def parse_keyword_list_string(keyword_list_string: str) -> List[str]:
    """
    JSON 배열 혹은 콤마 구분 문자열을 파싱하여 키워드 리스트를 반환한다.
    run_mocked_tools.py 와 공유하기 위해 별도 모듈로 분리했다.
    """
    try:
        keywords = json.loads(keyword_list_string)
        if not isinstance(keywords, list):
            keywords = [keyword_list_string]
    except json.JSONDecodeError:
        keywords = [kw.strip() for kw in keyword_list_string.split(",")]
    return [kw for kw in keywords if kw]


def to_cdata(value: str) -> str:
    """XML CDATA 블록으로 감싼 문자열을 반환한다."""
    return f"<![CDATA[{value}]]>"

