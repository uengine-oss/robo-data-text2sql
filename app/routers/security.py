"""
Security Management API Router
Neo4j 기반 사용자/역할 관리 및 감사 로그
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import json
import os
import uuid
import bcrypt

from app.deps import neo4j_conn

router = APIRouter(prefix="/security", tags=["security"])

# 감사 로그 파일 경로
AUDIT_LOG_DIR = os.path.join(os.path.dirname(__file__), "../../../data-secure-guard/logs")


# =====================
# Pydantic Models
# =====================

class UserCreate(BaseModel):
    email: str
    username: str
    password: str
    roles: List[str] = []

class UserUpdate(BaseModel):
    username: Optional[str] = None
    roles: Optional[List[str]] = None
    status: Optional[str] = None

class UserResponse(BaseModel):
    uid: str
    email: str
    username: str
    status: str
    roles: List[str]
    created_at: str
    last_login_at: Optional[str] = None

class RoleCreate(BaseModel):
    name: str
    description: str
    priority: int = 50
    permissions: List[str] = []

class RoleResponse(BaseModel):
    name: str
    description: str
    priority: int
    is_system: bool
    permissions: List[str]

class AuditLogEntry(BaseModel):
    log_id: str
    user_email: str
    session_id: str
    original_sql: str
    rewritten_sql: Optional[str] = None
    status: str
    timestamp: str
    execution_time_ms: Optional[int] = None
    details: Optional[str] = None

class AuditLogCreate(BaseModel):
    user_email: str
    session_id: str
    original_sql: str
    rewritten_sql: Optional[str] = None
    status: str  # 'allowed', 'denied', 'rewritten'
    execution_time_ms: Optional[int] = None
    details: Optional[str] = None


# =====================
# User Management
# =====================

@router.get("/users", response_model=List[UserResponse])
async def get_users():
    """모든 사용자 조회"""
    session = await neo4j_conn.get_session()
    try:
        query = """
        MATCH (u:User)
        OPTIONAL MATCH (u)-[:HAS_ROLE]->(r:Role)
        WITH u, collect(r.name) as roles
        RETURN u.uid as uid, u.email as email, u.username as username,
               u.status as status, roles,
               toString(u.created_at) as created_at,
               toString(u.last_login_at) as last_login_at
        ORDER BY u.created_at DESC
        """
        result = await session.run(query)
        users = []
        async for record in result:
            users.append(UserResponse(
                uid=record["uid"] or "",
                email=record["email"] or "",
                username=record["username"] or "",
                status=record["status"] or "active",
                roles=record["roles"] or [],
                created_at=record["created_at"] or "",
                last_login_at=record["last_login_at"]
            ))
        return users
    finally:
        await session.close()


@router.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    """사용자 생성"""
    session = await neo4j_conn.get_session()
    try:
        # 비밀번호 해싱
        hashed = bcrypt.hashpw(user.password.encode(), bcrypt.gensalt()).decode()
        uid = str(uuid.uuid4())
        
        # 먼저 사용자 생성
        create_query = """
        CREATE (u:User {
            uid: $uid,
            email: $email,
            username: $username,
            password_hash: $password_hash,
            status: 'active',
            created_at: datetime()
        })
        RETURN u.uid as uid
        """
        await session.run(create_query, {
            "uid": uid,
            "email": user.email,
            "username": user.username,
            "password_hash": hashed
        })
        
        # 역할 연결 (역할이 있는 경우에만)
        if user.roles:
            role_query = """
            MATCH (u:User {uid: $uid})
            UNWIND $roles as roleName
            MATCH (r:Role {name: roleName})
            CREATE (u)-[:HAS_ROLE]->(r)
            """
            await session.run(role_query, {"uid": uid, "roles": user.roles})
        
        # 생성된 사용자 조회
        result = await session.run("""
            MATCH (u:User {uid: $uid})
            OPTIONAL MATCH (u)-[:HAS_ROLE]->(r:Role)
            RETURN u.uid as uid, u.email as email, u.username as username,
                   u.status as status, collect(r.name) as roles,
                   toString(u.created_at) as created_at
        """, {"uid": uid})
        record = await result.single()
        
        if not record:
            raise HTTPException(status_code=500, detail="Failed to create user")
        
        return UserResponse(
            uid=record["uid"],
            email=record["email"],
            username=record["username"],
            status=record["status"],
            roles=record["roles"] or [],
            created_at=record["created_at"]
        )
    finally:
        await session.close()


@router.patch("/users/{uid}", response_model=UserResponse)
async def update_user(uid: str, updates: UserUpdate):
    """사용자 정보 수정"""
    session = await neo4j_conn.get_session()
    try:
        set_clauses = []
        params = {"uid": uid}
        
        if updates.username:
            set_clauses.append("u.username = $username")
            params["username"] = updates.username
        if updates.status:
            set_clauses.append("u.status = $status")
            params["status"] = updates.status
        
        # 역할 업데이트
        if updates.roles is not None:
            # 기존 역할 삭제
            await session.run(
                "MATCH (u:User {uid: $uid})-[r:HAS_ROLE]->() DELETE r",
                {"uid": uid}
            )
            # 새 역할 추가
            if updates.roles:
                await session.run("""
                    MATCH (u:User {uid: $uid})
                    UNWIND $roles as roleName
                    MATCH (r:Role {name: roleName})
                    CREATE (u)-[:HAS_ROLE]->(r)
                """, {"uid": uid, "roles": updates.roles})
        
        if set_clauses:
            query = f"MATCH (u:User {{uid: $uid}}) SET {', '.join(set_clauses)}"
            await session.run(query, params)
        
        # 업데이트된 사용자 반환
        result = await session.run("""
            MATCH (u:User {uid: $uid})
            OPTIONAL MATCH (u)-[:HAS_ROLE]->(r:Role)
            RETURN u.uid as uid, u.email as email, u.username as username,
                   u.status as status, collect(r.name) as roles,
                   toString(u.created_at) as created_at,
                   toString(u.last_login_at) as last_login_at
        """, {"uid": uid})
        record = await result.single()
        
        if not record:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            uid=record["uid"],
            email=record["email"],
            username=record["username"],
            status=record["status"],
            roles=record["roles"] or [],
            created_at=record["created_at"],
            last_login_at=record["last_login_at"]
        )
    finally:
        await session.close()


@router.delete("/users/{uid}")
async def delete_user(uid: str):
    """사용자 삭제"""
    session = await neo4j_conn.get_session()
    try:
        result = await session.run(
            "MATCH (u:User {uid: $uid}) DETACH DELETE u RETURN count(u) as deleted",
            {"uid": uid}
        )
        record = await result.single()
        if record["deleted"] == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"message": "User deleted successfully"}
    finally:
        await session.close()


# =====================
# Role Management
# =====================

@router.get("/roles", response_model=List[RoleResponse])
async def get_roles():
    """모든 역할 조회"""
    session = await neo4j_conn.get_session()
    try:
        query = """
        MATCH (r:Role)
        OPTIONAL MATCH (r)-[:HAS_PERMISSION]->(p:Permission)
        WITH r, collect(p.action + ':' + p.resource_type) as permissions
        RETURN r.name as name, r.description as description,
               r.priority as priority, 
               coalesce(r.is_system, false) as is_system,
               permissions
        ORDER BY r.priority DESC
        """
        result = await session.run(query)
        roles = []
        async for record in result:
            roles.append(RoleResponse(
                name=record["name"],
                description=record["description"] or "",
                priority=record["priority"] or 50,
                is_system=record["is_system"],
                permissions=record["permissions"] or []
            ))
        return roles
    finally:
        await session.close()


@router.post("/roles", response_model=RoleResponse)
async def create_role(role: RoleCreate):
    """역할 생성"""
    session = await neo4j_conn.get_session()
    try:
        query = """
        CREATE (r:Role {
            name: $name,
            description: $description,
            priority: $priority,
            is_system: false
        })
        RETURN r.name as name, r.description as description,
               r.priority as priority, r.is_system as is_system
        """
        result = await session.run(query, {
            "name": role.name,
            "description": role.description,
            "priority": role.priority
        })
        record = await result.single()
        
        # 권한 연결
        if role.permissions:
            for perm in role.permissions:
                parts = perm.split(":")
                if len(parts) == 2:
                    await session.run("""
                        MATCH (r:Role {name: $role_name})
                        MATCH (p:Permission {action: $action, resource_type: $resource_type})
                        MERGE (r)-[:HAS_PERMISSION]->(p)
                    """, {
                        "role_name": role.name,
                        "action": parts[0],
                        "resource_type": parts[1]
                    })
        
        return RoleResponse(
            name=record["name"],
            description=record["description"],
            priority=record["priority"],
            is_system=record["is_system"],
            permissions=role.permissions
        )
    finally:
        await session.close()


@router.delete("/roles/{name}")
async def delete_role(name: str):
    """역할 삭제 (시스템 역할 제외)"""
    session = await neo4j_conn.get_session()
    try:
        # 시스템 역할 체크
        check = await session.run(
            "MATCH (r:Role {name: $name}) RETURN r.is_system as is_system",
            {"name": name}
        )
        record = await check.single()
        if not record:
            raise HTTPException(status_code=404, detail="Role not found")
        if record["is_system"]:
            raise HTTPException(status_code=400, detail="Cannot delete system role")
        
        await session.run(
            "MATCH (r:Role {name: $name}) DETACH DELETE r",
            {"name": name}
        )
        return {"message": "Role deleted successfully"}
    finally:
        await session.close()


# =====================
# Permissions
# =====================

@router.get("/permissions")
async def get_permissions():
    """모든 권한 조회"""
    session = await neo4j_conn.get_session()
    try:
        result = await session.run("""
            MATCH (p:Permission)
            RETURN p.action as action, p.resource_type as resource_type,
                   p.description as description
            ORDER BY p.resource_type, p.action
        """)
        permissions = []
        async for record in result:
            permissions.append({
                "action": record["action"],
                "resource_type": record["resource_type"],
                "description": record["description"] or ""
            })
        return permissions
    finally:
        await session.close()


# =====================
# Audit Logs (File System)
# =====================

def _ensure_audit_log_dir():
    """감사 로그 디렉토리 확인/생성"""
    os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

def _get_audit_log_file():
    """오늘 날짜의 감사 로그 파일 경로"""
    _ensure_audit_log_dir()
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(AUDIT_LOG_DIR, f"audit_{today}.jsonl")


@router.get("/audit-logs", response_model=List[AuditLogEntry])
async def get_audit_logs(
    user_email: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100
):
    """감사 로그 조회 (파일 시스템 기반)"""
    _ensure_audit_log_dir()
    logs = []
    
    # 최근 7일간의 로그 파일 읽기
    for i in range(7):
        date = datetime.now()
        from datetime import timedelta
        date = date - timedelta(days=i)
        log_file = os.path.join(AUDIT_LOG_DIR, f"audit_{date.strftime('%Y-%m-%d')}.jsonl")
        
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        # 필터링
                        if user_email and entry.get("user_email") != user_email:
                            continue
                        if status and entry.get("status") != status:
                            continue
                        logs.append(AuditLogEntry(**entry))
                    except json.JSONDecodeError:
                        continue
    
    # 최신순 정렬 및 제한
    logs.sort(key=lambda x: x.timestamp, reverse=True)
    return logs[:limit]


@router.post("/audit-logs", response_model=AuditLogEntry)
async def create_audit_log(entry: AuditLogCreate):
    """감사 로그 기록"""
    log_file = _get_audit_log_file()
    
    log_entry = AuditLogEntry(
        log_id=str(uuid.uuid4()),
        user_email=entry.user_email,
        session_id=entry.session_id,
        original_sql=entry.original_sql,
        rewritten_sql=entry.rewritten_sql,
        status=entry.status,
        timestamp=datetime.now().isoformat(),
        execution_time_ms=entry.execution_time_ms,
        details=entry.details
    )
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry.dict(), ensure_ascii=False) + '\n')
    
    return log_entry


# =====================
# Tables (from Neo4j)
# =====================

@router.get("/tables")
async def get_tables():
    """Neo4j에서 테이블 목록 조회"""
    session = await neo4j_conn.get_session()
    try:
        result = await session.run("""
            MATCH (t:Table)
            OPTIONAL MATCH (t)-[:HAS_COLUMN]->(c:Column)
            WITH t, count(c) as column_count
            RETURN t.name as name, 
                   coalesce(t.schema, 'public') as schema,
                   coalesce(t.datasource, 'default') as db,
                   t.description as description,
                   column_count
            ORDER BY t.datasource, t.schema, t.name
        """)
        tables = []
        async for record in result:
            tables.append({
                "name": record["name"],
                "schema": record["schema"],
                "db": record["db"],
                "description": record["description"],
                "column_count": record["column_count"]
            })
        return tables
    finally:
        await session.close()
