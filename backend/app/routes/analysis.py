from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.database import get_db
from app.models.analysis import Analysis
from app.models.user import User


router = APIRouter(
    prefix="/api/analyses",
    tags=["Analysis History"],
)


@router.get("")
def get_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Return analysis history for the authenticated user.
    """

    analyses = (
        db.query(Analysis)
        .filter(Analysis.user_id == current_user.id)
        .order_by(Analysis.created_at.desc())
        .all()
    )

    return {
        "count": len(analyses),
        "analyses": [
            {
                "id": analysis.id,
                "filename": analysis.filename,
                "image_format": analysis.image_format,
                "image_size": {
                    "width": analysis.image_width,
                    "height": analysis.image_height,
                },
                "model": analysis.model_name,
                "detected_conditions": analysis.detected_conditions,
                "all_predictions": analysis.all_predictions,
                "created_at": analysis.created_at,
            }
            for analysis in analyses
        ],
    }