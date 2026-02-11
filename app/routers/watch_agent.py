"""
Watch Agent - Text-to-SQL API
자연어 감시 조건에서 SQL 쿼리 생성

감시 에이전트의 관리 기능은 agent-scheduler 서비스에서 담당합니다.
이 라우터는 Text-to-SQL 기능만 제공합니다:
- 자연어에서 데이터 가용성 분석
- 감시용 SQL 쿼리 생성
"""
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.deps import get_neo4j_session
from app.core.llm_factory import create_embedding_client
from app.core.graph_search import GraphSearcher, format_subschema_for_prompt
from app.core.prompt import SQLChain
from app.smart_logger import SmartLogger


router = APIRouter(prefix="/watch-agent", tags=["Watch Agent - Text-to-SQL"])


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: str
    content: str


class ChatRequest(BaseModel):
    """에이전트 빌더 채팅 요청"""
    message: str = Field(..., description="사용자 메시지")
    history: List[ChatMessage] = Field(default=[], description="이전 대화 기록")
    current_config: Optional[Dict[str, Any]] = Field(default=None, description="현재 설정 상태")
    step: str = Field(default="initial", description="설정 단계")


class ChatResponse(BaseModel):
    """에이전트 빌더 채팅 응답"""
    response: str
    extracted_config: Optional[Dict[str, Any]] = None
    generated_sql: Optional[str] = None
    ready_to_confirm: bool = False
    agent_created: bool = False
    data_available: bool = True
    relevant_tables: Optional[List[str]] = None


class SQLGenerationRequest(BaseModel):
    """SQL 생성 요청"""
    question: str = Field(..., description="자연어 질문")
    datasource: str = Field(..., description="MindsDB datasource (required; Phase 1 MindsDB-only)")


class SQLGenerationResponse(BaseModel):
    """SQL 생성 응답"""
    sql: Optional[str] = None
    tables: List[str] = []
    available: bool = True
    message: str = ""


# =============================================================================
# Helper Functions
# =============================================================================

async def analyze_query_for_data_availability(
    question: str,
    datasource: str,
    neo4j_session,
) -> Dict[str, Any]:
    """자연어 쿼리에서 데이터 가용성 분석"""
    try:
        embedding_client = create_embedding_client()
        query_embedding = await embedding_client.embed_text(question)
        
        searcher = GraphSearcher(neo4j_session)
        ds = (datasource or "").strip()
        subschema = await searcher.build_subschema(query_embedding, datasource=(ds or None))
        
        if not subschema.tables:
            return {
                "available": False,
                "tables": [],
                "message": "관련 테이블을 찾을 수 없습니다."
            }
        
        table_names = [t.name for t in subschema.tables]
        return {
            "available": True,
            "tables": table_names,
            "subschema": subschema,
            "message": f"{len(table_names)}개의 관련 테이블을 찾았습니다."
        }
    except Exception as e:
        return {
            "available": False,
            "tables": [],
            "message": f"데이터 검색 중 오류: {str(e)}"
        }


async def generate_monitoring_sql(
    question: str,
    subschema,
) -> str:
    """감시용 SQL 쿼리 생성"""
    schema_text = format_subschema_for_prompt(subschema)
    join_hints = "\n".join(subschema.join_hints) if subschema.join_hints else ""
    
    sql_chain = SQLChain()
    
    # 감시 특화 프롬프트 추가
    monitoring_hint = """
    이 쿼리는 주기적 모니터링에 사용됩니다.
    - 최근 데이터만 조회하도록 시간 조건을 포함하세요
    - 트렌드 분석을 위해 LAG/LEAD 윈도우 함수를 고려하세요
    - 집계 함수(AVG, MAX, MIN)를 활용하세요
    """
    
    enhanced_question = f"{question}\n\n{monitoring_hint}"
    
    sql = await sql_chain.generate_sql(
        question=enhanced_question,
        schema_text=schema_text,
        join_hints=join_hints
    )
    
    return sql


def extract_config_from_message(message: str) -> Dict[str, Any]:
    """메시지에서 설정 정보 추출"""
    config = {}
    
    # 시간 간격 추출
    interval_match = re.search(r'(\d+)\s*(분|시간|초)', message)
    if interval_match:
        interval = int(interval_match.group(1))
        unit = interval_match.group(2)
        if unit == '시간':
            interval *= 60
        elif unit == '초':
            interval = max(1, interval // 60)
        config['check_interval_minutes'] = interval
    
    # 프로세스 이름 추출
    process_match = re.search(r'(?:프로세스|실행)[:\s]*([가-힣\w_]+)', message, re.IGNORECASE)
    if process_match:
        config['process_name'] = process_match.group(1)
        config['process_id'] = process_match.group(1).lower().replace(' ', '_')
    
    # 조건 추출
    if '지속' in message or '상승' in message:
        config['condition_expression'] = 'trend == "rising"'
        config['condition_type'] = 'rising'
        config['condition_description'] = '지속 상승'
    elif '이상' in message or '초과' in message:
        value_match = re.search(r'(\d+(?:\.\d+)?)\s*(이상|초과)', message)
        if value_match:
            config['condition_expression'] = f'value >= {value_match.group(1)}'
            config['condition_type'] = 'threshold'
            config['condition_description'] = f'{value_match.group(1)} 이상'
    else:
        config['condition_type'] = 'exists'
        config['condition_expression'] = 'rows > 0'
        config['condition_description'] = '데이터 존재'
    
    # 이름 추출
    name_patterns = [
        r'(원수|탁도|수위|유량|온도).{0,10}(감시|모니터링)',
        r'(감시|모니터링).{0,10}(에이전트)'
    ]
    for pattern in name_patterns:
        name_match = re.search(pattern, message, re.IGNORECASE)
        if name_match:
            config['name'] = name_match.group(0)
            break
    
    if 'name' not in config:
        config['name'] = message[:30] + ('...' if len(message) > 30 else '')
    
    return config


# =============================================================================
# API Endpoints
# =============================================================================

@router.post("/chat", response_model=ChatResponse)
async def chat_with_builder(
    request: ChatRequest,
    neo4j_session=Depends(get_neo4j_session),
):
    """
    에이전트 빌더와 대화
    
    자연어로 감시 조건을 설명하면:
    1. Neo4j에서 관련 데이터 가용성 확인
    2. SQL 쿼리 생성
    3. 설정 정보 추출
    
    실제 에이전트 생성/관리는 agent-scheduler 서비스에서 담당합니다.
    """
    SmartLogger.log(
        "INFO",
        "watch_agent.chat",
        category="watch_agent.chat",
        params={"message": request.message[:100], "step": request.step}
    )
    
    try:
        # 설정 정보 추출
        extracted_config = extract_config_from_message(request.message)
        if request.current_config:
            extracted_config = {**request.current_config, **extracted_config}
        
        # 자연어 쿼리 저장
        extracted_config['natural_language_query'] = request.message
        
        # 데이터 가용성 분석
        data_analysis = await analyze_query_for_data_availability(
            request.message,
            request.current_config.get("datasource") if isinstance(request.current_config, dict) else "",
            neo4j_session,
        )
        
        if not data_analysis["available"]:
            return ChatResponse(
                response=f"""죄송합니다. 요청하신 조건에 맞는 데이터를 찾을 수 없습니다.

{data_analysis["message"]}

다음을 확인해주세요:
- 데이터베이스 스키마가 등록되어 있는지
- 관련 테이블 이름이 올바른지

다른 방식으로 설명해 주시겠어요?""",
                data_available=False,
                relevant_tables=[]
            )
        
        # SQL 쿼리 생성
        generated_sql = await generate_monitoring_sql(
            request.message,
            data_analysis["subschema"],
        )
        
        extracted_config['generated_sql'] = generated_sql
        extracted_config['monitored_tables'] = data_analysis["tables"]
        
        # 응답 생성
        interval = extracted_config.get('check_interval_minutes', 10)
        process_name = extracted_config.get('process_name', '(지정 필요)')
        condition = extracted_config.get('condition_description', '데이터 존재')
        
        response_text = f"""이해했습니다! 감시 에이전트를 다음과 같이 구성하겠습니다.

🔍 **감시 대상**
{request.message}

📊 **관련 테이블**: {', '.join(data_analysis["tables"][:5])}

⏱️ **감시 주기**: {interval}분마다 확인

📋 **조치 조건**: {condition}

⚡ **조치**: ProcessGPT 프로세스 실행
└ 프로세스: {process_name}

아래 SQL 쿼리로 데이터를 모니터링합니다. 수정이 필요하면 말씀해주세요."""

        return ChatResponse(
            response=response_text,
            extracted_config=extracted_config,
            generated_sql=generated_sql,
            ready_to_confirm=True,
            data_available=True,
            relevant_tables=data_analysis["tables"]
        )
        
    except Exception as e:
        SmartLogger.log(
            "ERROR",
            "watch_agent.chat.error",
            category="watch_agent.chat.error",
            params={"error": str(e)}
        )
        return ChatResponse(
            response=f"처리 중 오류가 발생했습니다: {str(e)}\n다시 시도해주세요.",
            data_available=False
        )


@router.post("/generate-sql", response_model=SQLGenerationResponse)
async def generate_sql(
    request: SQLGenerationRequest,
    neo4j_session=Depends(get_neo4j_session),
):
    """
    자연어에서 감시용 SQL 쿼리 생성
    
    프론트엔드에서 SQL 생성만 필요할 때 사용합니다.
    """
    try:
        # 데이터 가용성 분석
        data_analysis = await analyze_query_for_data_availability(
            request.question,
            request.datasource,
            neo4j_session,
        )
        
        if not data_analysis["available"]:
            return SQLGenerationResponse(
                sql=None,
                tables=[],
                available=False,
                message=data_analysis["message"]
            )
        
        # SQL 생성
        sql = await generate_monitoring_sql(
            request.question,
            data_analysis["subschema"],
        )
        
        return SQLGenerationResponse(
            sql=sql,
            tables=data_analysis["tables"],
            available=True,
            message=f"{len(data_analysis['tables'])}개 테이블에서 SQL을 생성했습니다."
        )
        
    except Exception as e:
        return SQLGenerationResponse(
            sql=None,
            tables=[],
            available=False,
            message=f"SQL 생성 오류: {str(e)}"
        )


@router.post("/analyze-availability")
async def analyze_availability(
    request: SQLGenerationRequest,
    neo4j_session=Depends(get_neo4j_session),
):
    """
    자연어 쿼리의 데이터 가용성 분석
    
    SQL 생성 전에 데이터가 존재하는지 먼저 확인할 때 사용합니다.
    """
    data_analysis = await analyze_query_for_data_availability(
        request.question,
        request.datasource,
        neo4j_session,
    )
    
    return {
        "available": data_analysis["available"],
        "tables": data_analysis.get("tables", []),
        "message": data_analysis["message"]
    }
