"""Synchronous pre-settlement scoring."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException

from api.auth import operator
from api.schemas import ScoreOut, TransferIn
from api.state import STATE
from pipeline.mojaloop_adapter import AdapterError, from_fspiop

router = APIRouter(prefix="/v1", tags=["scoring"])


@router.post("/score", response_model=ScoreOut, summary="Score one transfer before settlement")
def score(tx: TransferIn, _op: str = Depends(operator)) -> dict:
    return STATE.scorer.score(tx.model_dump())


@router.post(
    "/score/fspiop",
    response_model=ScoreOut,
    summary="Score a Mojaloop FSPIOP transfer-prepare message (or bare POST /transfers body)",
)
def score_fspiop(msg: dict = Body(...), _op: str = Depends(operator)) -> dict:
    try:
        tx = from_fspiop(msg)
    except AdapterError as e:
        raise HTTPException(422, str(e)) from e
    try:
        TransferIn(**{k: v for k, v in tx.items() if v is not None})
    except Exception as e:
        raise HTTPException(422, f"invalid transfer after mapping: {e}") from e
    return STATE.scorer.score(tx)
