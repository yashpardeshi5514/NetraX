from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from PIL import Image, UnidentifiedImageError
from io import BytesIO
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.models.user import User
from app.database import get_db
from app.models.analysis import Analysis
from app.services.inference import netrax_inference


router = APIRouter(
    prefix="/api",
    tags=["Prediction"],
)


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
}


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Analyze a retinal image using the trained NetraX model
    and save the analysis for the authenticated user.
    """

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Use PNG, JPEG, or WebP."
            ),
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    try:
        image = Image.open(BytesIO(contents))
        image.load()

    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    try:
        result = netrax_inference.predict(image)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        )

    analysis = Analysis(
        user_id=current_user.id,
        filename=file.filename or "uploaded_image",
        image_format=image.format or "UNKNOWN",
        image_width=image.width,
        image_height=image.height,
        model_name="NetraX_EfficientNetB0_FineTuned",
        detected_conditions=result["detected_conditions"],
        all_predictions=result["all_predictions"],
    )

    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return {
        "analysis_id": analysis.id,
        "filename": file.filename,
        "image_format": image.format,
        "image_size": {
            "width": image.width,
            "height": image.height,
        },
        "model": "NetraX_EfficientNetB0_FineTuned",
        "results": result,
    }
