import re
from typing import Any, Dict, Iterable


_REDACTED = "<REDACTED>"


# NOTE: "명백한 시크릿"만 마스킹한다.
_SENSITIVE_KEY_FRAGMENTS = {
    "api_key",
    "apikey",
    "openai_api_key",
    "openai_compatible_api_key",
    "authorization",
    "access_token",
    "refresh_token",
    "token",
    "password",
    "secret",
    "client_secret",
}


_SECRET_PATTERNS = [
    # OpenAI-style API key
    re.compile(r"\bsk-[A-Za-z0-9]{10,}\b"),
    # Bearer tokens in headers / logs
    re.compile(r"\bBearer\s+[A-Za-z0-9\-\._=]{10,}\b", re.IGNORECASE),
    # Common "Authorization: ..." header formats inside free text
    re.compile(r"\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9\-\._=]{10,}\b", re.IGNORECASE),
]


def _sanitize_string(value: str) -> str:
    if not value:
        return value
    result = value
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(_REDACTED, result)
    return result


def _is_sensitive_key(key: Any) -> bool:
    if key is None:
        return False
    k = str(key).strip().lower()
    if not k:
        return False
    for frag in _SENSITIVE_KEY_FRAGMENTS:
        if frag in k:
            return True
    return False


def sanitize_for_log(obj: Any, *, _depth: int = 0, _max_depth: int = 50) -> Any:
    """
    SmartLogger 로깅용 데이터에서 '명백한 시크릿'만 마스킹한다.

    - dict: key 기반으로 민감 키는 값 전체를 <REDACTED> 처리
    - str: 문자열 내부의 sk-... / Bearer ... 패턴만 치환
    - list/tuple/set: 원소별 재귀 처리
    - 그 외: 그대로 반환 (JSON 직렬화는 SmartLogger가 default=str로 처리)
    """
    if _depth >= _max_depth:
        return obj

    if obj is None:
        return None

    if isinstance(obj, str):
        return _sanitize_string(obj)

    if isinstance(obj, (int, float, bool)):
        return obj

    if isinstance(obj, dict):
        sanitized: Dict[Any, Any] = {}
        for k, v in obj.items():
            if _is_sensitive_key(k):
                sanitized[k] = _REDACTED
            else:
                sanitized[k] = sanitize_for_log(v, _depth=_depth + 1, _max_depth=_max_depth)
        return sanitized

    if isinstance(obj, (list, tuple, set)):
        items = [sanitize_for_log(v, _depth=_depth + 1, _max_depth=_max_depth) for v in obj]
        if isinstance(obj, tuple):
            return tuple(items)
        if isinstance(obj, set):
            return set(items)
        return items

    # Handle generic iterables (rare) but avoid iterating over bytes.
    if isinstance(obj, (bytes, bytearray)):
        return obj

    if isinstance(obj, Iterable):
        try:
            return [
                sanitize_for_log(v, _depth=_depth + 1, _max_depth=_max_depth)
                for v in obj
            ]
        except Exception:
            return obj

    return obj


