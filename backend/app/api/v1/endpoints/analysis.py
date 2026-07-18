from __future__ import annotations

import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.db.models import Analysis
from app.schemas.analysis import AnalysisRequest, AnalysisResult, AnalysisStatusResponse
from app.services import alignment as alignment_service

router = APIRouter()


@router.post("/analyze", response_model=AnalysisStatusResponse, status_code=202, tags=["analysis"])
def submit_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> AnalysisStatusResponse:
    """
    Submit a curriculum for analysis.
    Returns immediately with an analysis_id; poll GET /analyze/{id} for results.
    """
    analysis_id = uuid.uuid4()
    db_analysis = Analysis(
        id=analysis_id,
        mode=request.mode,
        status="pending",
        input_data=request.model_dump(),
    )
    db.add(db_analysis)
    db.commit()

    background_tasks.add_task(
        alignment_service.run_analysis,
        request=request,
        analysis_id=analysis_id,
        db=db,
    )

    return AnalysisStatusResponse(
        analysis_id=str(analysis_id),
        status="pending",
        progress_pct=0.0,
    )


@router.get("/analyze/{analysis_id}", response_model=AnalysisStatusResponse, tags=["analysis"])
def get_analysis(
    analysis_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> AnalysisStatusResponse:
    """Poll analysis status or retrieve completed result."""
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    result: AnalysisResult | None = None
    if analysis.status == "done" and analysis.result:
        result = AnalysisResult(**analysis.result)

    return AnalysisStatusResponse(
        analysis_id=str(analysis_id),
        status=analysis.status,
        error=analysis.error_message,
        result=result,
    )
