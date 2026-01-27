"""
LLM-free keyword extractor for Text2SQL.

Extracts meaningful keywords from natural language questions
for table/column search without requiring LLM calls.
"""

import re
from typing import List, Set


# Korean stop words (common particles, endings, etc.)
KOREAN_STOPWORDS: Set[str] = {
    # Particles
    "은", "는", "이", "가", "을", "를", "의", "에", "에서", "로", "으로",
    "와", "과", "랑", "이랑", "하고", "도", "만", "부터", "까지", "보다",
    # Common verbs/endings
    "하다", "되다", "있다", "없다", "하는", "되는", "있는", "없는",
    "해줘", "해주세요", "알려줘", "알려주세요", "보여줘", "보여주세요",
    "조회", "검색", "찾아", "찾아줘", "뭐야", "뭐지", "무엇",
    # Question words
    "어떤", "어떻게", "얼마", "몇", "언제", "어디", "누가", "왜",
    # Time-related common words (keep specific time values)
    "오늘", "어제", "내일", "지금", "현재", "최근", "작년", "올해",
    # Common SQL-related words
    "데이터", "정보", "값", "결과", "목록", "리스트",
}

# English stop words
ENGLISH_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "and",
    "but", "if", "or", "because", "until", "while", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "i", "me",
    "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "her", "hers", "herself", "it", "its", "itself", "they",
    "them", "their", "theirs", "themselves", "show", "find", "get",
    "give", "list", "display", "query", "search", "tell", "please",
}

# Combined stopwords
ALL_STOPWORDS = KOREAN_STOPWORDS | ENGLISH_STOPWORDS

# Pattern for Korean word boundaries (simplified)
KOREAN_PATTERN = re.compile(r"[가-힣]+")
# Pattern for English/alphanumeric words
ENGLISH_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]*")
# Pattern for numbers with units (e.g., "3일", "2024년")
NUMBER_UNIT_PATTERN = re.compile(r"\d+[가-힣a-zA-Z]+")
# Pattern for pure numbers
PURE_NUMBER_PATTERN = re.compile(r"\d+")


def extract_keywords(
    question: str,
    *,
    min_length: int = 2,
    max_keywords: int = 10,
    include_numbers: bool = True,
) -> List[str]:
    """
    Extract meaningful keywords from a natural language question.
    
    Args:
        question: The user's natural language question
        min_length: Minimum character length for a keyword
        max_keywords: Maximum number of keywords to return
        include_numbers: Whether to include numeric values
        
    Returns:
        List of extracted keywords, ordered by potential relevance
    """
    if not question:
        return []
    
    text = question.strip()
    keywords: List[str] = []
    seen: Set[str] = set()
    
    def add_keyword(word: str) -> None:
        word_lower = word.lower()
        if (
            word_lower not in seen
            and word_lower not in ALL_STOPWORDS
            and len(word) >= min_length
        ):
            keywords.append(word)
            seen.add(word_lower)
    
    # Extract Korean words
    for match in KOREAN_PATTERN.finditer(text):
        word = match.group()
        # Try to extract meaningful parts from compound words
        # Korean compound words often have 2-3 char meaningful units
        if len(word) > 4:
            # Keep full word and also try splitting
            add_keyword(word)
            # Simple split at 2-char boundaries for potential sub-words
            for i in range(0, len(word) - 1, 2):
                sub = word[i:i+2]
                if len(sub) >= 2:
                    add_keyword(sub)
        else:
            add_keyword(word)
    
    # Extract English words (potential table/column names)
    for match in ENGLISH_PATTERN.finditer(text):
        word = match.group()
        add_keyword(word)
    
    # Extract numbers with units (e.g., "2024년", "3일")
    if include_numbers:
        for match in NUMBER_UNIT_PATTERN.finditer(text):
            word = match.group()
            add_keyword(word)
        
        # Extract pure numbers (potential IDs, years, etc.)
        for match in PURE_NUMBER_PATTERN.finditer(text):
            num = match.group()
            if len(num) >= 4:  # Likely a year or ID
                add_keyword(num)
    
    # Prioritize longer keywords (often more specific)
    keywords.sort(key=lambda x: (-len(x), x))
    
    return keywords[:max_keywords]


def extract_search_keywords(
    question: str,
    *,
    max_keywords: int = 5,
) -> List[str]:
    """
    Extract keywords specifically optimized for Neo4j table/column search.
    
    This is a specialized version that focuses on:
    - Domain-specific terms (likely table/column names)
    - Proper nouns (locations, organization names)
    - Technical terms
    
    Args:
        question: The user's natural language question
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of search keywords
    """
    keywords = extract_keywords(
        question,
        min_length=2,
        max_keywords=max_keywords * 2,  # Get more, then filter
        include_numbers=False,  # Numbers rarely match table names
    )
    
    # Filter and prioritize
    prioritized: List[str] = []
    secondary: List[str] = []
    
    for kw in keywords:
        # Check if it looks like a potential table/column name
        # (longer Korean words, English identifiers, etc.)
        if (
            len(kw) >= 3
            or ENGLISH_PATTERN.fullmatch(kw)
        ):
            prioritized.append(kw)
        else:
            secondary.append(kw)
    
    result = prioritized + secondary
    return result[:max_keywords]


if __name__ == "__main__":
    # Test examples
    test_questions = [
        "창원 정수장의 평균 유량은?",
        "2024년 1월 서울시 매출 데이터 보여줘",
        "사용자별 주문 건수를 알려줘",
        "What is the average temperature in Seoul?",
        "Show me the sales report for Q1 2024",
        "고객 테이블에서 이름과 전화번호 조회",
    ]
    
    for q in test_questions:
        kws = extract_search_keywords(q)
        print(f"Q: {q}")
        print(f"Keywords: {kws}")
        print()
