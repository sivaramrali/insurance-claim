
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure the app directory is importable when started from the project root.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from explanations import generate_counterfactuals


app = FastAPI(
    title="Insurance Claim Transparency API",
    description="API for insurance claim prediction and counterfactual explanations.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class ClaimExplanationRequest(BaseModel):
    professional_fee: float = Field(ge=0)
    facility_fee: float = Field(ge=0)
    service_category: str = Field(min_length=1, max_length=100)
    total_cfs: int = Field(default=3, ge=1, le=5)


@app.get("/")
def root():
    return {
        "message": "Insurance Claim Transparency API",
        "status": "running",
    }


@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "insurance-claim-transparency-api",
    }


@app.post("/api/v1/explanations")
def create_explanation(request: ClaimExplanationRequest):
    try:
        result = generate_counterfactuals(
            professional_fee=request.professional_fee,
            facility_fee=request.facility_fee,
            service_category=request.service_category,
            total_cfs=request.total_cfs,
        )
        return result

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        # Log the full exception in the server terminal for debugging.
        import logging
        logging.exception("Counterfactual explanation failed")
        raise HTTPException(
            status_code=500,
            detail="Could not generate an explanation. Check the API terminal.",
        ) from exc
