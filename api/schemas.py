"""Pydantic v2 request/response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class TransferIn(BaseModel):
    """Internal (Mojaloop-derived) transfer schema accepted by POST /v1/score."""

    transfer_id: str = Field(min_length=1, max_length=128)
    quote_id: str | None = None
    ts: int = Field(gt=0, description="event time, epoch milliseconds")
    payer_id: str = Field(min_length=1, max_length=128)
    payer_alias: str | None = ""
    payer_fsp: str = Field(min_length=1, max_length=64)
    payer_type: Literal["CONSUMER", "MERCHANT", "BUSINESS"] = "CONSUMER"
    payee_id: str = Field(min_length=1, max_length=128)
    payee_alias: str | None = ""
    payee_fsp: str = Field(min_length=1, max_length=64)
    payee_type: Literal["CONSUMER", "MERCHANT", "BUSINESS"] = "CONSUMER"
    amount: float = Field(gt=0, lt=1e9)
    currency: str = Field(default="INR", min_length=3, max_length=3)
    scenario: Literal["TRANSFER", "PAYMENT", "REFUND", "REVERSAL", "DEPOSIT", "WITHDRAWAL"] = "TRANSFER"
    initiator: Literal["PAYER", "PAYEE"] = "PAYER"
    channel: str | None = ""
    device_id: str | None = ""
    ip: str | None = ""
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    device_bound_at: int | None = None
    merchant_category: str | None = ""
    original_transfer_id: str | None = ""

    @field_validator("payee_id")
    @classmethod
    def _not_self(cls, v, info):
        if v == info.data.get("payer_id"):
            raise ValueError("payer and payee must differ")
        return v


class Reason(BaseModel):
    feature: str
    value: float | None
    contribution: float
    direction: str
    source: str
    reason: str


class ScoreOut(BaseModel):
    transfer_id: str
    risk_score: float
    decision: Literal["ALLOW", "REVIEW", "BLOCK"]
    top_reasons: list[Reason]
    model_version: str
    latency_ms: float
    degraded: bool
    degraded_reasons: list[str] = []
    combo: str
    predicted_type: str | None = None
    explanation_pending: bool = False
    duplicate: bool = False
    timings_ms: dict[str, float] | None = None


class FeedbackIn(BaseModel):
    transfer_id: str = Field(min_length=1)
    label: Literal["fraud", "legit"]
    notes: str | None = Field(default=None, max_length=2000)


class CaseIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    transfer_ids: list[str] = []
    ring_id: str | None = None
    priority: Literal["low", "medium", "high", "critical"] = "medium"
    assignee: str | None = None
    note: str | None = None


class CasePatch(BaseModel):
    status: Literal["open", "investigating", "closed"] | None = None
    assignee: str | None = None
    priority: Literal["low", "medium", "high", "critical"] | None = None
    resolution: str | None = None
    note: str | None = Field(default=None, max_length=4000)
    add_transfer_ids: list[str] = []


class RetrainIn(BaseModel):
    reason: str | None = None


Json = dict[str, Any]
