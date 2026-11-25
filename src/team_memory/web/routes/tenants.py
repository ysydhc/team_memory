"""Multi-tenant workspace, member RBAC, and per-tenant API key routes.

Endpoints (mounted under /api/v1, with /api/* compat via middleware):
  GET    /tenants                        — list tenants current user belongs to
  POST   /tenants                        — create a new tenant (creator becomes admin)
  GET    /tenants/current                — get active tenant details & user role
  PUT    /tenants/current                — update active tenant name/settings (admin)
  GET    /tenants/current/members        — list members in active tenant (admin)
  POST   /tenants/current/members        — add/invite member to active tenant (admin)
  PUT    /tenants/current/members/{uid}  — update member role in active tenant (admin)
  DELETE /tenants/current/members/{uid}  — remove member from active tenant (admin)
  GET    /tenants/current/keys           — list current user's API keys in active tenant
  POST   /tenants/current/keys           — create a new API key in active tenant
  DELETE /tenants/current/keys/{key_id}  — revoke an API key in active tenant
"""

from __future__ import annotations

import logging
import re
import secrets
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, func, select

from team_memory.auth.provider import DbApiKeyAuth, User, _hash_password
from team_memory.storage.database import get_session
from team_memory.storage.models import ApiKey, Tenant, TenantMember
from team_memory.storage.tenant_context import DEFAULT_TENANT_ID, parse_tenant_uuid
from team_memory.web import app as app_module
from team_memory.web.app import _get_db_url
from team_memory.web.auth_session import get_current_user
from team_memory.web.dependencies import require_admin

logger = logging.getLogger("team_memory.web.tenants")

router = APIRouter(prefix="/tenants", tags=["tenants"])

_VALID_TENANT_ROLES = ("admin", "member", "viewer")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug[:80] or f"tenant-{secrets.token_hex(4)}"


# ============================================================
# Schemas
# ============================================================


class TenantCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    slug: str | None = Field(default=None, max_length=100)
    settings_json: dict[str, Any] = Field(default_factory=dict)


class TenantUpdateRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    settings_json: dict[str, Any] | None = None


class TenantMemberAddRequest(BaseModel):
    user_id: str = Field(..., min_length=2, max_length=100)
    role: str = Field(default="member")
    password: str | None = Field(default=None, min_length=6)
    generate_api_key: bool = False

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v == "editor":
            return "member"
        if v not in _VALID_TENANT_ROLES:
            raise ValueError("role must be admin, member, or viewer")
        return v


class TenantMemberUpdateRequest(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v == "editor":
            return "member"
        if v not in _VALID_TENANT_ROLES:
            raise ValueError("role must be admin, member, or viewer")
        return v


class TenantApiKeyCreateRequest(BaseModel):
    label: str | None = Field(default=None, max_length=100)
    role: str | None = None

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if v == "editor":
            return "member"
        if v not in _VALID_TENANT_ROLES:
            raise ValueError("role must be admin, member, or viewer")
        return v


# ============================================================
# Tenant CRUD & Listing
# ============================================================


@router.get("")
async def list_my_tenants(user: User = Depends(get_current_user)):
    """List all tenants the current user belongs to."""
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            select(Tenant, TenantMember.role)
            .join(TenantMember, TenantMember.tenant_id == Tenant.id)
            .where(TenantMember.user_id == user.name)
            .order_by(Tenant.created_at.asc())
        )
        rows = res.all()
        items = [
            {
                "id": str(tenant.id),
                "slug": tenant.slug,
                "name": tenant.name,
                "role": member_role,
                "settings_json": tenant.settings_json or {},
                "created_at": tenant.created_at.isoformat() if tenant.created_at else None,
            }
            for tenant, member_role in rows
        ]
        # Fallback for non-DB auth modes (e.g., NoAuth / in-memory ApiKeyAuth)
        if not items:
            t_res = await session.execute(select(Tenant).where(Tenant.id == DEFAULT_TENANT_ID))
            default_t = t_res.scalar_one_or_none()
            if default_t is not None:
                items.append(
                    {
                        "id": str(default_t.id),
                        "slug": default_t.slug,
                        "name": default_t.name,
                        "role": "member" if user.role == "editor" else user.role,
                        "settings_json": default_t.settings_json or {},
                        "created_at": default_t.created_at.isoformat() if default_t.created_at else None,
                    }
                )
        return {"tenants": items, "current_tenant_id": str(user.tenant_id)}


@router.post("", status_code=201)
async def create_tenant(
    req: TenantCreateRequest,
    user: User = Depends(get_current_user),
):
    """Create a new tenant workspace; the creator is automatically added as admin."""
    slug = _slugify(req.slug or req.name)
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        existing = await session.execute(select(Tenant).where(Tenant.slug == slug))
        if existing.scalar_one_or_none() is not None:
            raise HTTPException(status_code=409, detail=f"Tenant slug '{slug}' already exists")

        tenant = Tenant(
            slug=slug,
            name=req.name.strip(),
            settings_json=req.settings_json or {},
        )
        session.add(tenant)
        await session.flush()

        member = TenantMember(
            tenant_id=tenant.id,
            user_id=user.name,
            role="admin",
        )
        session.add(member)
        await session.flush()

        return {
            "tenant": {
                **tenant.to_dict(),
                "role": "admin",
            }
        }


@router.get("/current")
async def get_current_tenant(user: User = Depends(get_current_user)):
    """Return metadata for the active tenant."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(select(Tenant).where(Tenant.id == tid))
        tenant = res.scalar_one_or_none()
        if tenant is None:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return {
            "tenant": {
                **tenant.to_dict(),
                "role": "member" if user.role == "editor" else user.role,
            }
        }


@router.put("/current")
async def update_current_tenant(
    req: TenantUpdateRequest,
    user: User = Depends(require_admin()),
):
    """Update current tenant name or settings_json (Tenant Admin only)."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(select(Tenant).where(Tenant.id == tid))
        tenant = res.scalar_one_or_none()
        if tenant is None:
            raise HTTPException(status_code=404, detail="Tenant not found")
        if req.name is not None:
            tenant.name = req.name.strip()
        if req.settings_json is not None:
            merged = dict(tenant.settings_json or {})
            merged.update(req.settings_json)
            tenant.settings_json = merged
        await session.flush()
        return {"tenant": {**tenant.to_dict(), "role": "admin"}}


# ============================================================
# Tenant Member Management (Tenant Admin only)
# ============================================================


@router.get("/current/members")
async def list_tenant_members(user: User = Depends(require_admin())):
    """List all members of the active tenant (Tenant Admin only)."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            select(TenantMember)
            .where(TenantMember.tenant_id == tid)
            .order_by(TenantMember.created_at.asc())
        )
        members = res.scalars().all()

        # Enrich with active API key count in this tenant
        key_counts_res = await session.execute(
            select(ApiKey.user_name, func.count(ApiKey.id))
            .where(
                ApiKey.tenant_id == tid,
                ApiKey.is_active == True,  # noqa: E712
                ApiKey.key_hash.isnot(None),
            )
            .group_by(ApiKey.user_name)
        )
        key_counts = dict(key_counts_res.all())

        return {
            "members": [
                {
                    **m.to_dict(),
                    "active_key_count": int(key_counts.get(m.user_id, 0)),
                }
                for m in members
            ]
        }


@router.post("/current/members", status_code=201)
async def add_tenant_member(
    req: TenantMemberAddRequest,
    user: User = Depends(require_admin()),
):
    """Invite an existing user or create a new user in the active tenant (Tenant Admin only)."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    target_uid = req.user_id.strip()
    db_url = _get_db_url()
    _auth = app_module._auth

    async with get_session(db_url, bypass_rls=True) as session:
        existing_member = await session.execute(
            select(TenantMember).where(
                TenantMember.tenant_id == tid,
                TenantMember.user_id == target_uid,
            )
        )
        if existing_member.scalar_one_or_none() is not None:
            raise HTTPException(
                status_code=409,
                detail=f"User '{target_uid}' is already a member of this tenant",
            )

        member = TenantMember(tenant_id=tid, user_id=target_uid, role=req.role)
        session.add(member)

        raw_key: str | None = None
        if isinstance(_auth, DbApiKeyAuth):
            # Check whether this username already has a password/account row
            any_account = await session.execute(
                select(ApiKey).where(ApiKey.user_name == target_uid).limit(1)
            )
            has_account = any_account.scalar_one_or_none() is not None

            if req.generate_api_key or (not has_account and req.password):
                key_hash: str | None = None
                key_prefix: str | None = None
                key_suffix: str | None = None
                if req.generate_api_key:
                    raw_key = secrets.token_hex(32)
                    key_hash = _auth.hash_key(raw_key)
                    key_prefix = raw_key[:4]
                    key_suffix = raw_key[-4:]
                pwd_hash = _hash_password(req.password) if req.password else None
                db_key = ApiKey(
                    tenant_id=tid,
                    key_hash=key_hash,
                    user_name=target_uid,
                    role=req.role,
                    is_active=True,
                    password_hash=pwd_hash,
                    key_prefix=key_prefix,
                    key_suffix=key_suffix,
                    label="default" if raw_key else None,
                )
                session.add(db_key)
                if key_hash:
                    _auth._keys[key_hash] = User(
                        name=target_uid, role=req.role, tenant_id=str(tid)
                    )

        await session.flush()
        out: dict[str, Any] = {"member": member.to_dict()}
        if raw_key is not None:
            out["api_key"] = raw_key
        return out


@router.put("/current/members/{target_user_id}")
async def update_tenant_member(
    target_user_id: str,
    req: TenantMemberUpdateRequest,
    user: User = Depends(require_admin()),
):
    """Update a member's role in the active tenant (Tenant Admin only)."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    _auth = app_module._auth

    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            select(TenantMember).where(
                TenantMember.tenant_id == tid,
                TenantMember.user_id == target_user_id,
            )
        )
        member = res.scalar_one_or_none()
        if member is None:
            raise HTTPException(status_code=404, detail="Member not found in this tenant")

        # Prevent demoting the last admin
        if member.role == "admin" and req.role != "admin":
            admin_cnt_res = await session.execute(
                select(func.count(TenantMember.id)).where(
                    TenantMember.tenant_id == tid,
                    TenantMember.role == "admin",
                )
            )
            if (admin_cnt_res.scalar() or 0) <= 1:
                raise HTTPException(
                    status_code=400, detail="Cannot demote the last admin of the tenant"
                )

        member.role = req.role
        # Sync cached keys for this user+tenant
        keys_res = await session.execute(
            select(ApiKey).where(
                ApiKey.tenant_id == tid,
                ApiKey.user_name == target_user_id,
            )
        )
        for k in keys_res.scalars().all():
            if k.role != "viewer" or req.role == "viewer":
                k.role = req.role
            if k.key_hash and isinstance(_auth, DbApiKeyAuth):
                _auth._keys.pop(k.key_hash, None)

        await session.flush()
        return {"member": member.to_dict()}


@router.delete("/current/members/{target_user_id}")
async def remove_tenant_member(
    target_user_id: str,
    user: User = Depends(require_admin()),
):
    """Remove a member from the active tenant and revoke their keys in this tenant."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    _auth = app_module._auth

    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            select(TenantMember).where(
                TenantMember.tenant_id == tid,
                TenantMember.user_id == target_user_id,
            )
        )
        member = res.scalar_one_or_none()
        if member is None:
            raise HTTPException(status_code=404, detail="Member not found in this tenant")

        if member.role == "admin":
            admin_cnt_res = await session.execute(
                select(func.count(TenantMember.id)).where(
                    TenantMember.tenant_id == tid,
                    TenantMember.role == "admin",
                )
            )
            if (admin_cnt_res.scalar() or 0) <= 1:
                raise HTTPException(
                    status_code=400, detail="Cannot remove the last admin of the tenant"
                )

        # Deactivate keys scoped to this tenant for the removed member
        keys_res = await session.execute(
            select(ApiKey).where(
                ApiKey.tenant_id == tid,
                ApiKey.user_name == target_user_id,
            )
        )
        for k in keys_res.scalars().all():
            k.is_active = False
            if k.key_hash and isinstance(_auth, DbApiKeyAuth):
                _auth._keys.pop(k.key_hash, None)

        await session.execute(
            delete(TenantMember).where(
                TenantMember.tenant_id == tid,
                TenantMember.user_id == target_user_id,
            )
        )
        await session.flush()
        return {"message": f"Member '{target_user_id}' removed from tenant"}


# ============================================================
# Per-Tenant Personal API Keys
# ============================================================


@router.get("/current/keys")
async def list_my_tenant_keys(user: User = Depends(get_current_user)):
    """List current user's active API keys in the active tenant."""
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            select(ApiKey)
            .where(
                ApiKey.tenant_id == tid,
                ApiKey.user_name == user.name,
                ApiKey.is_active == True,  # noqa: E712
                ApiKey.key_hash.isnot(None),
            )
            .order_by(ApiKey.created_at.desc())
        )
        keys = res.scalars().all()
        return {
            "keys": [
                {
                    "id": k.id,
                    "label": k.label or "default",
                    "role": "member" if k.role == "editor" else k.role,
                    "masked": f"{k.key_prefix}****{k.key_suffix}"
                    if k.key_prefix and k.key_suffix
                    else "••••****••••",
                    "tenant_id": str(k.tenant_id),
                    "created_at": k.created_at.isoformat() if k.created_at else None,
                }
                for k in keys
            ]
        }


@router.post("/current/keys", status_code=201)
async def create_my_tenant_key(
    req: TenantApiKeyCreateRequest,
    user: User = Depends(get_current_user),
):
    """Create a new API key for the current user bound to the active tenant.

    Viewers can only mint read-only (viewer) keys. Non-admins cannot mint admin keys.
    """
    _auth = app_module._auth
    if not isinstance(_auth, DbApiKeyAuth):
        raise HTTPException(status_code=400, detail="API key creation requires db_api_key mode")

    user_norm_role = "member" if user.role == "editor" else user.role
    requested_role = req.role or user_norm_role

    # Enforce privilege ceiling: cannot mint a key with higher privileges than own role
    role_rank = {"viewer": 0, "member": 1, "admin": 2}
    if role_rank.get(requested_role, 0) > role_rank.get(user_norm_role, 0):
        raise HTTPException(
            status_code=403,
            detail=f"Role '{user_norm_role}' cannot create a '{requested_role}' API key",
        )

    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()
    async with get_session(db_url, bypass_rls=True) as session:
        result = await _auth.register_key_db(
            session=session,
            user_name=user.name,
            role=requested_role,
            generate_api_key=True,
            tenant_id=str(tid),
            label=(req.label or "personal").strip(),
            allow_existing_user=True,
        )
        return {"key": result, "api_key": result["api_key"]}


@router.delete("/current/keys/{key_id}")
async def revoke_tenant_key(
    key_id: int,
    user: User = Depends(get_current_user),
):
    """Revoke an API key in the active tenant (own key, or any key if admin)."""
    _auth = app_module._auth
    tid = parse_tenant_uuid(user.tenant_id) or DEFAULT_TENANT_ID
    db_url = _get_db_url()

    async with get_session(db_url, bypass_rls=True) as session:
        res = await session.execute(
            select(ApiKey).where(ApiKey.id == key_id, ApiKey.tenant_id == tid)
        )
        db_key = res.scalar_one_or_none()
        if db_key is None:
            raise HTTPException(status_code=404, detail="API key not found")
        if db_key.user_name != user.name and user.role != "admin":
            raise HTTPException(status_code=403, detail="Cannot revoke another user's API key")

        db_key.is_active = False
        if db_key.key_hash and isinstance(_auth, DbApiKeyAuth):
            _auth._keys.pop(db_key.key_hash, None)
        await session.flush()
        return {"message": "API key revoked"}
