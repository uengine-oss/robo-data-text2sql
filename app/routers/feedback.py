"""Feedback endpoint for learning from user corrections"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import json
from datetime import datetime

from app.deps import get_neo4j_session


router = APIRouter(prefix="/feedback", tags=["Feedback"])


class FeedbackRequest(BaseModel):
    """User feedback on generated SQL"""
    # ask.py flow emits prompt_snapshot_id, but ReAct streaming flow doesn't.
    prompt_snapshot_id: Optional[str] = Field(None, description="Original prompt snapshot ID (optional)")
    # Optional context (useful for audit/debug; not required).
    question: Optional[str] = Field(None, description="Original user question (optional)")
    react_run_id: Optional[str] = Field(None, description="ReAct run id (optional)")
    conversation_state: Optional[str] = Field(None, description="Conversation state token (optional)")

    original_sql: str = Field(..., description="Original generated SQL")
    corrected_sql: Optional[str] = Field(None, description="User's corrected SQL")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    notes: Optional[str] = Field(None, description="Additional feedback notes")
    approved: bool = Field(default=False, description="Whether the result was approved")
    source: Optional[str] = Field(default="ui", description="Feedback source tag (ui/api/etc)")


class FeedbackResponse(BaseModel):
    """Response after storing feedback"""
    feedback_id: str
    message: str


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: FeedbackRequest,
    neo4j_session=Depends(get_neo4j_session)
):
    """
    Submit feedback on a generated SQL query.
    This can be used to improve future query generation.
    """
    # Normalize / generate prompt_snapshot_id (best-effort)
    snapshot_id = (feedback.prompt_snapshot_id or "").strip()
    if not snapshot_id:
        base = (feedback.question or "") + "|" + (feedback.original_sql or "")
        snapshot_id = f"ps_{int(datetime.now().timestamp())}_{hash(base) % 10000}"

    # Generate feedback ID
    feedback_id = f"fb_{int(datetime.now().timestamp())}_{hash(snapshot_id) % 10000}"
    
    # Store feedback in Neo4j (or could use a separate feedback database)
    query = """
    CREATE (f:Feedback {
        id: $feedback_id,
        prompt_snapshot_id: $prompt_snapshot_id,
        question: $question,
        react_run_id: $react_run_id,
        conversation_state: $conversation_state,
        original_sql: $original_sql,
        corrected_sql: $corrected_sql,
        rating: $rating,
        notes: $notes,
        approved: $approved,
        source: $source,
        created_at: datetime()
    })
    RETURN f.id AS id
    """
    
    try:
        result = await neo4j_session.run(
            query,
            feedback_id=feedback_id,
            prompt_snapshot_id=snapshot_id,
            question=feedback.question,
            react_run_id=feedback.react_run_id,
            conversation_state=feedback.conversation_state,
            original_sql=feedback.original_sql,
            corrected_sql=feedback.corrected_sql,
            rating=feedback.rating,
            notes=feedback.notes,
            approved=feedback.approved,
            source=(feedback.source or "ui"),
        )
        
        record = await result.single()
        
        return FeedbackResponse(
            feedback_id=record["id"],
            message="Feedback submitted successfully. Thank you!"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to store feedback: {str(e)}")


@router.get("/stats")
async def get_feedback_stats(neo4j_session=Depends(get_neo4j_session)):
    """
    Get feedback statistics.
    """
    query = """
    MATCH (f:Feedback)
    RETURN count(f) AS total_feedback,
           avg(f.rating) AS avg_rating,
           sum(CASE WHEN f.approved THEN 1 ELSE 0 END) AS approved_count,
           sum(CASE WHEN f.corrected_sql IS NOT NULL THEN 1 ELSE 0 END) AS correction_count
    """
    
    result = await neo4j_session.run(query)
    record = await result.single()
    
    if not record:
        return {
            "total_feedback": 0,
            "avg_rating": 0.0,
            "approved_count": 0,
            "correction_count": 0
        }
    
    return {
        "total_feedback": record["total_feedback"],
        "avg_rating": round(record["avg_rating"], 2) if record["avg_rating"] else 0.0,
        "approved_count": record["approved_count"],
        "correction_count": record["correction_count"]
    }

