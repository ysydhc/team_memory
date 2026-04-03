"""Pydantic request/response models for the web API.

Extracted from web/app.py for focused, importable schema definitions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class LoginRequest(BaseModel):
    api_key: str | None = None
    username: str | None = None
    password: str | None = None


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    success: bool
    user: str = ""
    role: str = ""
    message: str = ""


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


class ForgotPasswordResetRequest(BaseModel):
    username: str
    api_key: str
    new_password: str


class AdminResetPasswordRequest(BaseModel):
    username: str
    new_password: str


class ExperienceCreate(BaseModel):
    title: str = Field(..., max_length=500)
    problem: str = Field(..., max_length=2_560)
    solution: str | None = Field(None, max_length=6_400)
    tags: list[str] = Field(default_factory=list)
    status: str = "published"
    visibility: str = "project"
    skip_dedup_check: bool = False
    experience_type: str = "general"
    project: str | None = None
    group_key: str | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        for tag in v:
            if len(tag) > 50:
                raise ValueError(f"Tag too long (max 50 chars): {tag[:20]}...")
        return v


class ExperienceUpdate(BaseModel):
    title: str | None = Field(None, max_length=500)
    problem: str | None = Field(None, max_length=2_560)
    solution: str | None = Field(None, max_length=6_400)
    tags: list[str] | None = None
    experience_type: str | None = None
    exp_status: str | None = None
    visibility: str | None = None
    solution_addendum: str | None = Field(None, max_length=6_400)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if len(v) > 20:
            raise ValueError("Maximum 20 tags allowed")
        for tag in v:
            if len(tag) > 50:
                raise ValueError(f"Tag too long (max 50 chars): {tag[:20]}...")
        return v


class FeedbackCreate(BaseModel):
    rating: int
    comment: str | None = None

    @field_validator("rating")
    @classmethod
    def rating_range(cls, v: int) -> int:
        if not (1 <= v <= 5):
            raise ValueError("rating must be between 1 and 5")
        return v


class SearchRequest(BaseModel):
    query: str
    tags: list[str] | None = None
    max_results: int | None = None
    min_similarity: float = 0.5
    grouped: bool = True
    top_k_children: int | None = None
    project: str | None = None
    include_archives: bool = False


class ApiKeyCreateRequest(BaseModel):
    user_name: str
    role: str = "editor"
    password: str | None = None
    generate_api_key: bool = False

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("admin", "editor", "viewer"):
            raise ValueError("role must be admin, editor, or viewer")
        return v


class ApiKeyUpdateRequest(BaseModel):
    role: str | None = None
    is_active: bool | None = None
    generate_api_key: bool | None = None
