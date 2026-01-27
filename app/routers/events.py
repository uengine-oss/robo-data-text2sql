"""
이벤트 감지 및 조치 API 라우터

기능:
1. 이벤트 규칙 CRUD (생성, 조회, 수정, 삭제)
2. 이벤트 규칙 활성/비활성
3. 수동 실행 (SQL 실행 및 조건 체크)
4. 스케줄러 관리
5. 대화형 이벤트 설정 (Chat API)
6. Esper CEP 연동
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import get_db_connection, get_neo4j_session
from app.core.sql_exec import SQLExecutor, SQLExecutionError
from app.core.sql_guard import SQLGuard, SQLValidationError
from app.smart_logger import SmartLogger

# CEP 클라이언트 (선택적 - 없으면 로컬 스케줄러 사용)
try:
    from app.core.cep_client import get_cep_client, sync_rule_to_cep, delete_rule_from_cep
    CEP_AVAILABLE = True
except ImportError:
    CEP_AVAILABLE = False

# SimpleCEP 엔진 (Python 기반 경량 CEP)
try:
    from app.core.simple_cep import (
        get_simple_cep_engine, 
        SimpleCEPEngine,
        EventRule as CEPEventRule,
        Event as CEPEvent,
        TriggerResult,
        ConditionOperator,
        create_rule_from_natural_language
    )
    from app.core.event_poller import get_event_poller, EventPoller
    SIMPLE_CEP_AVAILABLE = True
except ImportError:
    SIMPLE_CEP_AVAILABLE = False


router = APIRouter(prefix="/events", tags=["Events"])


# ============================================================================
# 모델 정의
# ============================================================================

class AlertConfig(BaseModel):
    """알림 설정"""
    channels: List[str] = Field(default=["platform"], description="알림 채널 목록")
    message: str = Field(default="", description="알림 메시지 템플릿")


class ProcessConfig(BaseModel):
    """프로세스 실행 설정"""
    process_name: str = Field(default="", description="실행할 프로세스 이름")
    process_params: Dict[str, Any] = Field(default_factory=dict, description="프로세스 파라미터")


class EventRuleCreate(BaseModel):
    """이벤트 규칙 생성 요청"""
    name: str = Field(..., description="이벤트 규칙 이름")
    description: str = Field(default="", description="설명")
    natural_language_condition: str = Field(..., description="자연어 감지 조건")
    sql: str = Field(..., description="감지 SQL 쿼리")
    check_interval_minutes: int = Field(default=10, ge=1, le=1440, description="감지 간격(분)")
    condition_threshold: str = Field(default="rows > 0", description="트리거 조건")
    action_type: Literal["alert", "process"] = Field(default="alert", description="조치 유형")
    alert_config: Optional[AlertConfig] = Field(default=None, description="알림 설정")
    process_config: Optional[ProcessConfig] = Field(default=None, description="프로세스 설정")


class EventRuleUpdate(BaseModel):
    """이벤트 규칙 수정 요청"""
    name: Optional[str] = None
    description: Optional[str] = None
    natural_language_condition: Optional[str] = None
    sql: Optional[str] = None
    check_interval_minutes: Optional[int] = Field(default=None, ge=1, le=1440)
    condition_threshold: Optional[str] = None
    action_type: Optional[Literal["alert", "process"]] = None
    alert_config: Optional[AlertConfig] = None
    process_config: Optional[ProcessConfig] = None
    is_active: Optional[bool] = None


class EventRule(BaseModel):
    """이벤트 규칙"""
    id: str
    name: str
    description: str
    natural_language_condition: str
    sql: str
    check_interval_minutes: int
    condition_threshold: str
    action_type: Literal["alert", "process"]
    alert_config: Optional[AlertConfig] = None
    process_config: Optional[ProcessConfig] = None
    is_active: bool = True
    last_checked_at: Optional[str] = None
    last_triggered_at: Optional[str] = None
    trigger_count: int = 0
    created_at: str
    updated_at: str


class EventExecutionResult(BaseModel):
    """이벤트 실행 결과"""
    event_id: str
    executed_at: str
    sql_result: Dict[str, Any]
    condition_met: bool
    action_taken: Optional[str] = None
    error: Optional[str] = None


class EventNotification(BaseModel):
    """이벤트 알림"""
    id: str
    event_id: str
    event_name: str
    message: str
    triggered_at: str
    acknowledged: bool = False
    data: Optional[Dict[str, Any]] = None


# ============================================================================
# 인메모리 저장소 (프로덕션에서는 DB 사용)
# ============================================================================

_event_rules: Dict[str, EventRule] = {}
_notifications: List[EventNotification] = []
_scheduler_tasks: Dict[str, asyncio.Task] = {}
_scheduler_running = False


def _now_iso() -> str:
    return datetime.now().isoformat()


# ============================================================================
# 이벤트 규칙 CRUD
# ============================================================================

@router.get("/rules", response_model=List[EventRule])
async def list_event_rules():
    """등록된 이벤트 규칙 목록 조회"""
    return list(_event_rules.values())


@router.get("/rules/{event_id}", response_model=EventRule)
async def get_event_rule(event_id: str):
    """이벤트 규칙 상세 조회"""
    if event_id not in _event_rules:
        raise HTTPException(status_code=404, detail="Event rule not found")
    return _event_rules[event_id]


@router.post("/rules", response_model=EventRule)
async def create_event_rule(request: EventRuleCreate):
    """새 이벤트 규칙 생성"""
    event_id = str(uuid.uuid4())
    now = _now_iso()
    
    # SQL 유효성 검증
    guard = SQLGuard()
    try:
        guard.validate(request.sql)
    except SQLValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid SQL: {e}")
    
    rule = EventRule(
        id=event_id,
        name=request.name,
        description=request.description,
        natural_language_condition=request.natural_language_condition,
        sql=request.sql,
        check_interval_minutes=request.check_interval_minutes,
        condition_threshold=request.condition_threshold,
        action_type=request.action_type,
        alert_config=request.alert_config,
        process_config=request.process_config,
        is_active=True,
        created_at=now,
        updated_at=now
    )
    
    _event_rules[event_id] = rule
    
    SmartLogger.log(
        "INFO",
        f"Event rule created: {rule.name}",
        category="events.create",
        params={"event_id": event_id, "name": rule.name}
    )
    
    # 스케줄러가 실행 중이면 새 규칙 스케줄링
    if _scheduler_running:
        _schedule_event_rule(rule)
    
    return rule


@router.put("/rules/{event_id}", response_model=EventRule)
async def update_event_rule(event_id: str, request: EventRuleUpdate):
    """이벤트 규칙 수정"""
    if event_id not in _event_rules:
        raise HTTPException(status_code=404, detail="Event rule not found")
    
    rule = _event_rules[event_id]
    update_data = request.model_dump(exclude_unset=True)
    
    # SQL 유효성 검증
    if "sql" in update_data:
        guard = SQLGuard()
        try:
            guard.validate(update_data["sql"])
        except SQLValidationError as e:
            raise HTTPException(status_code=400, detail=f"Invalid SQL: {e}")
    
    for key, value in update_data.items():
        setattr(rule, key, value)
    
    rule.updated_at = _now_iso()
    _event_rules[event_id] = rule
    
    SmartLogger.log(
        "INFO",
        f"Event rule updated: {rule.name}",
        category="events.update",
        params={"event_id": event_id}
    )
    
    return rule


@router.delete("/rules/{event_id}")
async def delete_event_rule(event_id: str):
    """이벤트 규칙 삭제"""
    if event_id not in _event_rules:
        raise HTTPException(status_code=404, detail="Event rule not found")
    
    # 스케줄러 태스크 취소
    if event_id in _scheduler_tasks:
        _scheduler_tasks[event_id].cancel()
        del _scheduler_tasks[event_id]
    
    del _event_rules[event_id]
    
    SmartLogger.log(
        "INFO",
        f"Event rule deleted: {event_id}",
        category="events.delete",
        params={"event_id": event_id}
    )
    
    return {"message": "Event rule deleted", "id": event_id}


@router.post("/rules/{event_id}/toggle")
async def toggle_event_rule(event_id: str):
    """이벤트 규칙 활성/비활성 토글"""
    if event_id not in _event_rules:
        raise HTTPException(status_code=404, detail="Event rule not found")
    
    rule = _event_rules[event_id]
    rule.is_active = not rule.is_active
    rule.updated_at = _now_iso()
    
    # 비활성화되면 스케줄러 태스크 취소
    if not rule.is_active and event_id in _scheduler_tasks:
        _scheduler_tasks[event_id].cancel()
        del _scheduler_tasks[event_id]
    # 활성화되면 스케줄링
    elif rule.is_active and _scheduler_running:
        _schedule_event_rule(rule)
    
    return {"message": f"Event rule {'activated' if rule.is_active else 'deactivated'}", "is_active": rule.is_active}


# ============================================================================
# 이벤트 실행
# ============================================================================

@router.post("/rules/{event_id}/run", response_model=EventExecutionResult)
async def run_event_check(
    event_id: str,
    db_conn=Depends(get_db_connection)
):
    """이벤트 규칙 수동 실행"""
    if event_id not in _event_rules:
        raise HTTPException(status_code=404, detail="Event rule not found")
    
    rule = _event_rules[event_id]
    result = await _execute_event_check(rule, db_conn)
    
    return result


async def _execute_event_check(rule: EventRule, db_conn) -> EventExecutionResult:
    """이벤트 규칙 실행 및 조건 체크"""
    now = _now_iso()
    
    try:
        # SQL 실행
        executor = SQLExecutor()
        guard = SQLGuard()
        validated_sql, _ = guard.validate(rule.sql)
        
        raw_result = await executor.execute_query(db_conn, validated_sql, timeout=60.0)
        formatted = executor.format_results_for_json(raw_result)
        
        # 조건 평가
        rows = formatted.get("row_count", 0)
        condition_met = _evaluate_condition(rule.condition_threshold, rows, formatted)
        
        # 규칙 상태 업데이트
        rule.last_checked_at = now
        
        action_taken = None
        
        if condition_met:
            rule.last_triggered_at = now
            rule.trigger_count += 1
            
            # 조치 실행
            action_taken = await _execute_action(rule, formatted)
        
        SmartLogger.log(
            "INFO",
            f"Event check executed: {rule.name}",
            category="events.execute",
            params={
                "event_id": rule.id,
                "condition_met": condition_met,
                "rows": rows
            }
        )
        
        return EventExecutionResult(
            event_id=rule.id,
            executed_at=now,
            sql_result=formatted,
            condition_met=condition_met,
            action_taken=action_taken
        )
        
    except SQLExecutionError as e:
        SmartLogger.log(
            "ERROR",
            f"Event check failed: {rule.name}",
            category="events.execute.error",
            params={"event_id": rule.id, "error": str(e)}
        )
        return EventExecutionResult(
            event_id=rule.id,
            executed_at=now,
            sql_result={},
            condition_met=False,
            error=str(e)
        )
    except Exception as e:
        SmartLogger.log(
            "ERROR",
            f"Event check failed: {rule.name}",
            category="events.execute.error",
            params={"event_id": rule.id, "error": str(e)}
        )
        return EventExecutionResult(
            event_id=rule.id,
            executed_at=now,
            sql_result={},
            condition_met=False,
            error=str(e)
        )


def _evaluate_condition(threshold: str, rows: int, result: Dict) -> bool:
    """조건 평가"""
    try:
        # 간단한 조건 파서
        # 지원 형식: "rows > 0", "rows >= 5", "rows == 0", "rows != 0"
        threshold = threshold.strip().lower()
        
        if "rows" in threshold:
            # rows 변수를 현재 rows 값으로 대체
            condition = threshold.replace("rows", str(rows))
            return eval(condition)
        
        # 기본값
        return rows > 0
    except Exception:
        # 파싱 실패 시 기본 동작
        return rows > 0


async def _execute_action(rule: EventRule, result: Dict) -> str:
    """조치 실행"""
    if rule.action_type == "alert":
        return await _send_alert(rule, result)
    elif rule.action_type == "process":
        return await _execute_process(rule, result)
    return "No action configured"


async def _send_alert(rule: EventRule, result: Dict) -> str:
    """알림 발송"""
    notification_id = str(uuid.uuid4())
    now = _now_iso()
    
    # 메시지 생성
    message = rule.alert_config.message if rule.alert_config and rule.alert_config.message else \
        f"이벤트 '{rule.name}'이(가) 트리거되었습니다."
    
    notification = EventNotification(
        id=notification_id,
        event_id=rule.id,
        event_name=rule.name,
        message=message,
        triggered_at=now,
        data=result
    )
    
    _notifications.insert(0, notification)
    
    # 알림 개수 제한 (최근 100개만 유지)
    if len(_notifications) > 100:
        _notifications.pop()
    
    channels = rule.alert_config.channels if rule.alert_config else ["platform"]
    
    SmartLogger.log(
        "INFO",
        f"Alert sent for event: {rule.name}",
        category="events.alert",
        params={"event_id": rule.id, "channels": channels}
    )
    
    # TODO: 실제 알림 채널 연동 (Slack, Email, Webhook 등)
    
    return f"Alert sent to: {', '.join(channels)}"


async def _execute_process(rule: EventRule, result: Dict) -> str:
    """프로세스 실행 (ProcessGPT MCP 연동)"""
    if not rule.process_config or not rule.process_config.process_name:
        return "No process configured"
    
    process_name = rule.process_config.process_name
    params = rule.process_config.process_params or {}
    
    try:
        from app.core.mcp_client import execute_process_via_mcp
        
        # MCP를 통해 프로세스 실행
        execution_result = await execute_process_via_mcp(
            process_name=process_name,
            params=params,
            event_data={
                "event_id": rule.id,
                "event_name": rule.name,
                "condition": rule.natural_language_condition,
                "sql_result": result
            }
        )
        
        if execution_result.get("success"):
            SmartLogger.log(
                "INFO",
                f"Process executed successfully: {process_name}",
                category="events.process.success",
                params={
                    "event_id": rule.id,
                    "process_name": process_name,
                    "result": execution_result
                }
            )
            return f"Process '{process_name}' executed successfully"
        else:
            error = execution_result.get("error", "Unknown error")
            SmartLogger.log(
                "WARNING",
                f"Process execution failed: {process_name}",
                category="events.process.failed",
                params={
                    "event_id": rule.id,
                    "process_name": process_name,
                    "error": error
                }
            )
            return f"Process '{process_name}' execution failed: {error}"
            
    except ImportError:
        # MCP 클라이언트 모듈이 없는 경우 폴백
        SmartLogger.log(
            "WARNING",
            f"MCP client not available, logging process request only",
            category="events.process.fallback",
            params={
                "event_id": rule.id,
                "process_name": process_name
            }
        )
        return f"Process '{process_name}' execution requested (MCP not available)"
    except Exception as e:
        SmartLogger.log(
            "ERROR",
            f"Process execution error: {e}",
            category="events.process.error",
            params={
                "event_id": rule.id,
                "process_name": process_name,
                "error": str(e)
            }
        )
        return f"Process '{process_name}' execution error: {str(e)}"


# ============================================================================
# 알림 관리
# ============================================================================

@router.get("/notifications", response_model=List[EventNotification])
async def list_notifications(
    limit: int = 50,
    unacknowledged_only: bool = False
):
    """알림 목록 조회"""
    notifications = _notifications
    
    if unacknowledged_only:
        notifications = [n for n in notifications if not n.acknowledged]
    
    return notifications[:limit]


@router.post("/notifications/{notification_id}/acknowledge")
async def acknowledge_notification(notification_id: str):
    """알림 확인 처리"""
    for notification in _notifications:
        if notification.id == notification_id:
            notification.acknowledged = True
            return {"message": "Notification acknowledged"}
    
    raise HTTPException(status_code=404, detail="Notification not found")


@router.delete("/notifications/{notification_id}")
async def delete_notification(notification_id: str):
    """알림 삭제"""
    global _notifications
    _notifications = [n for n in _notifications if n.id != notification_id]
    return {"message": "Notification deleted"}


# ============================================================================
# 스케줄러 관리
# ============================================================================

def _schedule_event_rule(rule: EventRule):
    """이벤트 규칙 스케줄링"""
    if not rule.is_active:
        return
    
    async def scheduler_loop():
        while True:
            try:
                # 간격만큼 대기
                await asyncio.sleep(rule.check_interval_minutes * 60)
                
                # 규칙이 여전히 활성 상태인지 확인
                if rule.id not in _event_rules or not _event_rules[rule.id].is_active:
                    break
                
                # DB 연결 획득 및 실행
                # 주의: 실제 구현에서는 의존성 주입 방식 수정 필요
                from app.deps import db_pool
                async with db_pool.acquire() as conn:
                    await _execute_event_check(_event_rules[rule.id], conn)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                SmartLogger.log(
                    "ERROR",
                    f"Scheduler error for event: {rule.name}",
                    category="events.scheduler.error",
                    params={"event_id": rule.id, "error": str(e)}
                )
                # 오류 발생 시 1분 대기 후 재시도
                await asyncio.sleep(60)
    
    task = asyncio.create_task(scheduler_loop())
    _scheduler_tasks[rule.id] = task


@router.post("/scheduler/start")
async def start_scheduler():
    """스케줄러 시작"""
    global _scheduler_running
    
    if _scheduler_running:
        return {"message": "Scheduler already running", "active_rules": len(_scheduler_tasks)}
    
    _scheduler_running = True
    
    # 모든 활성 규칙 스케줄링
    for rule in _event_rules.values():
        if rule.is_active:
            _schedule_event_rule(rule)
    
    SmartLogger.log(
        "INFO",
        "Event scheduler started",
        category="events.scheduler.start",
        params={"active_rules": len(_scheduler_tasks)}
    )
    
    return {"message": "Scheduler started", "active_rules": len(_scheduler_tasks)}


@router.post("/scheduler/stop")
async def stop_scheduler():
    """스케줄러 중지"""
    global _scheduler_running
    
    _scheduler_running = False
    
    # 모든 태스크 취소
    for task in _scheduler_tasks.values():
        task.cancel()
    
    _scheduler_tasks.clear()
    
    SmartLogger.log(
        "INFO",
        "Event scheduler stopped",
        category="events.scheduler.stop"
    )
    
    return {"message": "Scheduler stopped"}


@router.get("/scheduler/status")
async def get_scheduler_status():
    """스케줄러 상태 조회"""
    return {
        "running": _scheduler_running,
        "active_tasks": len(_scheduler_tasks),
        "scheduled_events": list(_scheduler_tasks.keys())
    }


# ============================================================================
# 대화형 이벤트 설정 (Chat API)
# ============================================================================

class ChatMessage(BaseModel):
    """대화 메시지"""
    role: Literal["user", "assistant", "system"]
    content: str


class EventChatRequest(BaseModel):
    """대화형 이벤트 설정 요청"""
    message: str = Field(..., description="사용자 메시지")
    history: List[ChatMessage] = Field(default=[], description="대화 이력")
    current_config: Dict[str, Any] = Field(default={}, description="현재 설정 상태")
    step: str = Field(default="initial", description="현재 단계")


class ExtractedEventConfig(BaseModel):
    """추출된 이벤트 설정"""
    name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[str] = None
    interval: Optional[int] = None
    threshold: Optional[str] = None
    action_type: Optional[Literal["alert", "process"]] = None
    process_name: Optional[str] = None


class EventChatResponse(BaseModel):
    """대화형 이벤트 설정 응답"""
    response: str = Field(..., description="AI 응답")
    extracted_config: Optional[ExtractedEventConfig] = None
    ready_to_confirm: bool = Field(default=False, description="확정 가능 상태")
    event_created: bool = Field(default=False, description="이벤트 생성 완료 여부")
    next_step: Optional[str] = None


@router.post("/chat", response_model=EventChatResponse)
async def event_chat(request: EventChatRequest):
    """
    대화형 이벤트 설정 API
    
    사용자의 자연어 설명을 분석하여 이벤트 규칙을 설정합니다.
    """
    user_message = request.message.strip()
    current_config = request.current_config
    step = request.step
    
    # 자연어에서 이벤트 정보 추출
    extracted = _extract_event_info(user_message)
    
    # 기존 설정과 병합
    for key, value in extracted.items():
        if value is not None:
            current_config[key] = value
    
    # 단계별 응답 생성
    if step == "initial" or step == "done":
        # 초기 분석
        response, ready = _generate_initial_response(user_message, current_config)
        next_step = "confirm" if ready else "analyzing"
        
    elif step == "analyzing" or step == "confirm":
        # 수정 요청 처리
        response, ready = _handle_modification(user_message, current_config)
        next_step = "confirm" if ready else "analyzing"
        
    else:
        response = "죄송합니다, 이해하지 못했어요. 다시 설명해 주시겠어요?"
        ready = False
        next_step = step
    
    return EventChatResponse(
        response=response,
        extracted_config=ExtractedEventConfig(**{
            k: v for k, v in extracted.items() if v is not None
        }) if extracted else None,
        ready_to_confirm=ready,
        event_created=False,
        next_step=next_step
    )


def _extract_event_info(text: str) -> Dict[str, Any]:
    """자연어에서 이벤트 정보 추출"""
    config: Dict[str, Any] = {}
    lower_text = text.lower()
    
    # 조건 추출 (숫자 + 단위 패턴)
    condition_patterns = [
        r'(수위|온도|유량|탁도|압력|수량).{0,20}(\d+(?:\.\d+)?)\s*(m|미터|도|°C|%|퍼센트|이상|이하|초과|미만)',
        r'(\d+(?:\.\d+)?)\s*(m|미터|도|°C|%)\s*(이상|이하|초과|미만)',
    ]
    
    for pattern in condition_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            config['condition'] = match.group(0)
            break
    
    if 'condition' not in config and len(text) < 150:
        config['condition'] = text
    
    # 이름 추출
    name_patterns = [
        r'(수위|온도|유량|탁도|압력).{0,5}(이상|급증|급감|경고|감지)',
        r'(이상|급증|급감).{0,5}(감지|경고|알림)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            config['name'] = match.group(0)
            break
    
    if 'name' not in config:
        config['name'] = text[:30] + ('...' if len(text) > 30 else '')
    
    # 시간 간격 추출
    interval_match = re.search(r'(\d+)\s*(분|시간|초)', text)
    if interval_match:
        interval = int(interval_match.group(1))
        unit = interval_match.group(2)
        if unit == '시간':
            interval *= 60
        elif unit == '초':
            interval = max(1, interval // 60)
        config['interval'] = interval
    else:
        config['interval'] = 10  # 기본값
    
    # 조치 타입 추출
    if any(word in lower_text for word in ['프로세스', '자동', '실행', '조치']):
        config['action_type'] = 'process'
        
        # 프로세스 이름 추출
        process_match = re.search(r'(?:프로세스|실행)[:\s]*([가-힣\w_]+)', text, re.IGNORECASE)
        if process_match:
            config['process_name'] = process_match.group(1)
    else:
        config['action_type'] = 'alert'
    
    # 지속 시간 조건
    if '지속' in text or '계속' in text:
        duration_match = re.search(r'(\d+)\s*(분|시간).{0,5}(지속|계속)', text)
        if duration_match:
            dur_value = duration_match.group(1)
            dur_unit = 'h' if duration_match.group(2) == '시간' else 'm'
            config['threshold'] = f"duration >= {dur_value}{dur_unit}"
    
    if 'threshold' not in config:
        config['threshold'] = 'rows > 0'
    
    return config


def _generate_initial_response(user_message: str, config: Dict[str, Any]) -> tuple[str, bool]:
    """초기 분석 응답 생성"""
    condition = config.get('condition', user_message)
    interval = config.get('interval', 10)
    action_type = config.get('action_type', 'alert')
    process_name = config.get('process_name', '')
    
    response = f"""이해했습니다! 다음과 같이 이벤트를 설정하겠습니다.

📊 **감지 조건**
"{condition}"

⏱️ **감지 간격**: {interval}분마다 확인

{"⚡ **조치**: 자동 프로세스 실행" if action_type == "process" else "🔔 **조치**: 알림 발송"}
{f"🔧 **실행 프로세스**: {process_name}" if process_name else ""}

이대로 설정을 완료할까요? 수정이 필요하면 말씀해 주세요."""

    # 기본 정보가 있으면 확정 가능
    ready = bool(condition)
    
    return response, ready


def _handle_modification(user_message: str, config: Dict[str, Any]) -> tuple[str, bool]:
    """수정 요청 처리"""
    lower_msg = user_message.lower()
    
    # 간격 변경
    interval_match = re.search(r'(\d+)\s*(분|시간)', user_message)
    if interval_match:
        interval = int(interval_match.group(1))
        if interval_match.group(2) == '시간':
            interval *= 60
        config['interval'] = interval
        
        return f"감지 간격을 **{interval}분**으로 변경했습니다. 다른 수정 사항이 있으신가요?", True
    
    # 조치 방법 변경
    if '알림' in lower_msg or '알려' in lower_msg:
        config['action_type'] = 'alert'
        return "조치 방법을 **알림 발송**으로 변경했습니다.", True
    
    if '프로세스' in lower_msg or '자동' in lower_msg or '실행' in lower_msg:
        config['action_type'] = 'process'
        process_match = re.search(r'프로세스[:\s]*([가-힣\w_]+)', user_message, re.IGNORECASE)
        if process_match:
            config['process_name'] = process_match.group(1)
            return f"조치 방법을 **프로세스 실행**으로 변경했습니다. ({process_match.group(1)})", True
        return "조치 방법을 **프로세스 실행**으로 변경했습니다.", True
    
    # 조건 변경
    if '조건' in lower_msg or '수위' in lower_msg or '유량' in lower_msg:
        new_config = _extract_event_info(user_message)
        if new_config.get('condition'):
            config['condition'] = new_config['condition']
            return f"조건을 **\"{new_config['condition']}\"**로 변경했습니다.", True
    
    # 이해 못함
    return "수정 사항을 이해했습니다. 다른 변경이 필요하시면 말씀해 주세요.", True


# ============================================================================
# CEP 콜백 엔드포인트
# ============================================================================

class CEPAlertCallback(BaseModel):
    """CEP 알림 콜백 데이터"""
    rule_id: str
    rule_name: str
    description: Optional[str] = None
    condition: Optional[str] = None
    event_count: int = 0
    triggered_at: str
    alert_config: Optional[str] = None


class CEPProcessCallback(BaseModel):
    """CEP 프로세스 콜백 데이터"""
    rule_id: str
    rule_name: str
    process_config: Optional[str] = None
    event_count: int = 0
    triggered_at: str
    trigger_event: Optional[Dict[str, Any]] = None


@router.post("/cep-alert")
async def handle_cep_alert(callback: CEPAlertCallback):
    """
    CEP 서비스에서 알림 콜백 수신
    
    Esper CEP 엔진에서 조건 충족 시 호출됩니다.
    """
    SmartLogger.log(
        "INFO",
        f"CEP alert received: {callback.rule_name}",
        category="cep.alert.received",
        params={"rule_id": callback.rule_id, "event_count": callback.event_count}
    )
    
    # 플랫폼 알림 생성
    notification_id = str(uuid.uuid4())
    
    message = f"[CEP] 이벤트 '{callback.rule_name}'이(가) 트리거되었습니다."
    if callback.condition:
        message += f" 조건: {callback.condition}"
    
    notification = EventNotification(
        id=notification_id,
        event_id=callback.rule_id,
        event_name=callback.rule_name,
        message=message,
        triggered_at=callback.triggered_at,
        data={"source": "cep", "event_count": callback.event_count}
    )
    
    _notifications.insert(0, notification)
    
    if len(_notifications) > 100:
        _notifications.pop()
    
    return {"status": "processed", "notification_id": notification_id}


@router.post("/cep-process")
async def handle_cep_process(callback: CEPProcessCallback):
    """
    CEP 서비스에서 프로세스 실행 콜백 수신
    
    Esper CEP 엔진에서 프로세스 실행 조건 충족 시 호출됩니다.
    """
    SmartLogger.log(
        "INFO",
        f"CEP process callback received: {callback.rule_name}",
        category="cep.process.received",
        params={"rule_id": callback.rule_id}
    )
    
    # 프로세스 설정 파싱
    process_config = {}
    if callback.process_config:
        try:
            process_config = json.loads(callback.process_config)
        except json.JSONDecodeError:
            pass
    
    process_name = process_config.get("process_name", "")
    process_params = process_config.get("process_params", {})
    
    # MCP를 통해 프로세스 실행
    try:
        from app.core.mcp_client import execute_process_via_mcp
        
        result = await execute_process_via_mcp(
            process_name=process_name,
            params=process_params,
            event_data={
                "rule_id": callback.rule_id,
                "rule_name": callback.rule_name,
                "trigger_event": callback.trigger_event,
                "triggered_at": callback.triggered_at
            }
        )
        
        return {"status": "executed", "result": result}
        
    except ImportError:
        SmartLogger.log(
            "WARNING",
            "MCP client not available for process execution",
            category="cep.process.mcp_unavailable"
        )
        return {"status": "logged", "message": "MCP client not available"}
    except Exception as e:
        SmartLogger.log(
            "ERROR",
            f"Process execution failed: {e}",
            category="cep.process.error"
        )
        return {"status": "error", "error": str(e)}


@router.get("/cep/status")
async def get_cep_status():
    """CEP 서비스 상태 조회"""
    result = {
        "local_scheduler_running": _scheduler_running,
        "local_active_tasks": len(_scheduler_tasks)
    }
    
    # SimpleCEP 상태
    if SIMPLE_CEP_AVAILABLE:
        try:
            cep_engine = get_simple_cep_engine()
            result["simple_cep_status"] = cep_engine.get_status()
        except Exception as e:
            result["simple_cep_status"] = {"status": "error", "error": str(e)}
    
    # Esper CEP 클라이언트 상태
    if CEP_AVAILABLE:
        try:
            client = get_cep_client()
            status = await client.get_status()
            result["esper_cep_available"] = True
            result["esper_cep_status"] = status
        except Exception as e:
            result["esper_cep_available"] = True
            result["esper_cep_status"] = {"status": "error", "error": str(e)}
    else:
        result["esper_cep_available"] = False
    
    return result


# ============================================================================
# SimpleCEP 연동 및 시뮬레이션 테스트
# ============================================================================

class SimulationRequest(BaseModel):
    """시뮬레이션 요청"""
    rule_name: str = Field(..., description="규칙 이름")
    natural_language_condition: str = Field(..., description="자연어 조건")
    field_name: str = Field(default="water_level", description="감시 필드")
    threshold: float = Field(..., description="임계값")
    duration_minutes: int = Field(default=10, description="지속 시간 (분)")
    # 시뮬레이션 데이터
    simulated_value: float = Field(..., description="시뮬레이션 값")
    simulated_duration_minutes: int = Field(default=12, description="시뮬레이션 지속 시간")
    station_id: str = Field(default="TEST-STATION", description="관측소 ID")


class SimulationResult(BaseModel):
    """시뮬레이션 결과"""
    rule_id: str
    rule_name: str
    events_generated: int
    alarms_triggered: int
    alarms: List[Dict[str, Any]] = []
    condition_details: Dict[str, Any]


@router.post("/simulate", response_model=SimulationResult)
async def run_simulation(request: SimulationRequest):
    """
    CEP 시뮬레이션 실행
    
    10분 지속 조건 등을 테스트하기 위해 가짜 타임스탬프로
    이벤트를 생성하고 CEP 엔진에서 알람이 트리거되는지 확인합니다.
    """
    if not SIMPLE_CEP_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="SimpleCEP engine not available"
        )
    
    from datetime import datetime, timedelta
    
    # 새 CEP 엔진 인스턴스 생성 (테스트 격리)
    cep_engine = SimpleCEPEngine()
    
    # 규칙 생성
    rule_id = str(uuid.uuid4())
    
    # 연산자 결정
    operator = ConditionOperator.GTE
    if "초과" in request.natural_language_condition:
        operator = ConditionOperator.GT
    elif "미만" in request.natural_language_condition:
        operator = ConditionOperator.LT
    
    cep_rule = CEPEventRule(
        id=rule_id,
        name=request.rule_name,
        description=request.natural_language_condition,
        field_name=request.field_name,
        operator=operator,
        threshold=request.threshold,
        window_minutes=max(30, request.duration_minutes * 2),
        duration_minutes=request.duration_minutes,
        action_type="alert"
    )
    cep_engine.register_rule(cep_rule)
    
    # 알람 수집
    triggered_alarms = []
    cep_engine.add_trigger_callback(lambda r: triggered_alarms.append({
        "rule_id": r.rule_id,
        "rule_name": r.rule_name,
        "triggered_at": r.triggered_at.isoformat(),
        "duration": str(r.condition_met_duration),
        "matching_events": len(r.matching_events)
    }))
    
    # 시뮬레이션 이벤트 생성
    base_time = datetime.now()
    events = []
    
    num_events = request.simulated_duration_minutes + 1
    for i in range(num_events):
        timestamp = base_time + timedelta(minutes=i)
        events.append(CEPEvent(
            timestamp=timestamp,
            source_id=request.station_id,
            event_type=request.field_name,
            data={
                "station_id": request.station_id,
                request.field_name: request.simulated_value,
                "measured_at": timestamp.isoformat()
            }
        ))
    
    # 이벤트 전송
    for event in events:
        cep_engine.send_event(event)
    
    # 알람이 트리거되면 SSE로 브로드캐스트 + 알림 저장
    for alarm in triggered_alarms:
        alarm_id = str(uuid.uuid4())
        alarm_data = {
            "id": alarm_id,
            "rule_name": alarm["rule_name"],
            "message": f"조건 충족: {request.natural_language_condition}",
            "severity": "warning",
            "triggered_at": alarm["triggered_at"],
            "duration": alarm["duration"],
            "matching_events": alarm["matching_events"],
            "acknowledged": False
        }
        
        # 알림 저장
        notification = EventNotification(
            id=alarm_id,
            event_id=rule_id,
            event_name=alarm["rule_name"],
            message=f"조건 충족: {request.natural_language_condition}",
            triggered_at=alarm["triggered_at"],
            data={
                "severity": "warning",
                "duration": alarm["duration"],
                "matching_events": alarm["matching_events"]
            }
        )
        _notifications.insert(0, notification)
        if len(_notifications) > 100:
            _notifications.pop()
        
        # SSE 브로드캐스트
        asyncio.create_task(broadcast_alarm(alarm_data))
    
    # 결과 반환
    return SimulationResult(
        rule_id=rule_id,
        rule_name=request.rule_name,
        events_generated=len(events),
        alarms_triggered=len(triggered_alarms),
        alarms=triggered_alarms,
        condition_details={
            "field": request.field_name,
            "operator": operator.value,
            "threshold": request.threshold,
            "required_duration_minutes": request.duration_minutes,
            "simulated_value": request.simulated_value,
            "simulated_duration_minutes": request.simulated_duration_minutes
        }
    )


@router.post("/simple-cep/register")
async def register_simple_cep_rule(
    rule_id: str,
    field_name: str = "water_level",
    threshold: float = 3.0,
    duration_minutes: int = 10,
    action_type: str = "alert"
):
    """SimpleCEP에 규칙 등록"""
    if not SIMPLE_CEP_AVAILABLE:
        raise HTTPException(status_code=503, detail="SimpleCEP not available")
    
    # 기존 이벤트 규칙 조회
    if rule_id not in _event_rules:
        raise HTTPException(status_code=404, detail="Event rule not found")
    
    rule = _event_rules[rule_id]
    
    # SimpleCEP 규칙 생성
    cep_engine = get_simple_cep_engine()
    
    cep_rule = CEPEventRule(
        id=rule_id,
        name=rule.name,
        description=rule.description,
        field_name=field_name,
        operator=ConditionOperator.GTE,
        threshold=threshold,
        window_minutes=max(30, duration_minutes * 2),
        duration_minutes=duration_minutes,
        action_type=action_type
    )
    
    cep_engine.register_rule(cep_rule)
    
    return {
        "message": "Rule registered in SimpleCEP",
        "rule_id": rule_id,
        "cep_status": cep_engine.get_status()
    }


@router.post("/simple-cep/send-event")
async def send_event_to_simple_cep(
    station_id: str,
    field_name: str,
    value: float,
    timestamp: Optional[str] = None
):
    """SimpleCEP에 이벤트 전송 (테스트용)"""
    if not SIMPLE_CEP_AVAILABLE:
        raise HTTPException(status_code=503, detail="SimpleCEP not available")
    
    from datetime import datetime
    
    cep_engine = get_simple_cep_engine()
    
    ts = datetime.fromisoformat(timestamp) if timestamp else datetime.now()
    
    event = CEPEvent(
        timestamp=ts,
        source_id=station_id,
        event_type=field_name,
        data={
            "station_id": station_id,
            field_name: value,
            "measured_at": ts.isoformat()
        }
    )
    
    results = cep_engine.send_event(event)
    
    triggered = [
        {
            "rule_id": r.rule_id,
            "rule_name": r.rule_name,
            "triggered_at": r.triggered_at.isoformat(),
            "duration": str(r.condition_met_duration)
        }
        for r in results
    ]
    
    return {
        "event_sent": True,
        "timestamp": ts.isoformat(),
        "triggers": triggered,
        "cep_status": cep_engine.get_status()
    }


@router.get("/simple-cep/status")
async def get_simple_cep_status():
    """SimpleCEP 상태 조회"""
    if not SIMPLE_CEP_AVAILABLE:
        return {"available": False}
    
    cep_engine = get_simple_cep_engine()
    return {
        "available": True,
        **cep_engine.get_status()
    }


# ============================================================================
# SSE (Server-Sent Events) 실시간 알람 스트림
# ============================================================================

import asyncio
from sse_starlette.sse import EventSourceResponse
from datetime import datetime
import json

# 알람 이벤트 큐 (연결된 클라이언트들에게 브로드캐스트)
_alarm_subscribers: list = []


async def broadcast_alarm(alarm_data: dict):
    """모든 구독자에게 알람 브로드캐스트"""
    message = json.dumps(alarm_data, default=str)
    for queue in _alarm_subscribers:
        try:
            await queue.put(message)
        except:
            pass


@router.get("/stream/alarms")
async def stream_alarms(request: Request):
    """
    SSE 알람 스트림
    
    클라이언트가 이 엔드포인트에 연결하면 실시간으로 알람을 받습니다.
    """
    queue = asyncio.Queue()
    _alarm_subscribers.append(queue)
    
    async def event_generator():
        try:
            # 연결 확인 메시지
            yield {
                "event": "connected",
                "data": json.dumps({
                    "message": "알람 스트림에 연결되었습니다.",
                    "timestamp": datetime.now().isoformat()
                })
            }
            
            while True:
                # 클라이언트 연결 종료 확인
                if await request.is_disconnected():
                    break
                
                try:
                    # 큐에서 알람 대기 (1초 타임아웃)
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield {
                        "event": "alarm",
                        "data": message
                    }
                except asyncio.TimeoutError:
                    # 연결 유지를 위한 heartbeat
                    yield {
                        "event": "heartbeat",
                        "data": json.dumps({"timestamp": datetime.now().isoformat()})
                    }
        finally:
            _alarm_subscribers.remove(queue)
    
    return EventSourceResponse(event_generator())


@router.post("/trigger-alarm")
async def trigger_alarm(
    rule_name: str = "테스트 알람",
    message: str = "이것은 테스트 알람입니다.",
    severity: str = "info"
):
    """
    테스트용 알람 트리거
    
    SSE 스트림으로 알람을 브로드캐스트합니다.
    """
    alarm_data = {
        "id": str(uuid.uuid4()),
        "rule_name": rule_name,
        "message": message,
        "severity": severity,
        "triggered_at": datetime.now().isoformat(),
        "acknowledged": False
    }
    
    # 내부 알림 저장소에 추가
    notification = EventNotification(
        id=alarm_data["id"],
        event_id="manual-trigger",
        event_name=rule_name,
        message=message,
        triggered_at=alarm_data["triggered_at"],
        data={"severity": severity}
    )
    _notifications.insert(0, notification)
    
    # 알림 개수 제한
    if len(_notifications) > 100:
        _notifications.pop()
    
    # SSE로 브로드캐스트
    await broadcast_alarm(alarm_data)
    
    return {"message": "알람이 트리거되었습니다.", "alarm": alarm_data}
