
from pathlib import Path
import logging
import sys

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Ensure the app directory is importable when started from the project root.
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from explanations import generate_counterfactuals
from vehicle_explanations import generate_vehicle_counterfactuals


app = FastAPI(
    title="Insurance Claim Transparency API",
    description=(
        "API for insurance claim prediction and counterfactual "
        "explanations for health and vehicle insurance demonstrations."
    ),
    version="0.3.0",
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


class VehicleClaimExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vehicle_age: int = Field(ge=0, le=100)
    driver_age: int = Field(ge=18, le=120)
    vehicle_value: float = Field(gt=0)
    claim_amount: float = Field(ge=0)
    repair_cost: float = Field(ge=0)
    previous_claims: int = Field(ge=0, le=100)
    policy_duration_months: int = Field(ge=0, le=1200)
    documents_complete: int = Field(ge=0, le=1)
    police_report_available: int = Field(ge=0, le=1)
    accident_severity: str = Field(min_length=1, max_length=30)
    total_cfs: int = Field(default=3, ge=1, le=5)

    @model_validator(mode="after")
    def validate_claim_amounts(self):
        if self.repair_cost > self.claim_amount:
            raise ValueError(
                "repair_cost cannot exceed claim_amount."
            )

        if self.claim_amount > self.vehicle_value:
            raise ValueError(
                "claim_amount cannot exceed vehicle_value."
            )

        return self


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
        return generate_counterfactuals(
            professional_fee=request.professional_fee,
            facility_fee=request.facility_fee,
            service_category=request.service_category,
            total_cfs=request.total_cfs,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logging.exception("Health counterfactual explanation failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate a health-claim explanation. "
                "Check the API terminal."
            ),
        ) from exc


@app.post("/api/v1/vehicle/explanations")
def create_vehicle_explanation(
    request: VehicleClaimExplanationRequest,
):
    try:
        claim = request.model_dump(
            exclude={"total_cfs"}
        )

        return generate_vehicle_counterfactuals(
            claim=claim,
            total_cfs=request.total_cfs,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logging.exception("Vehicle counterfactual explanation failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not generate a vehicle-claim explanation. "
                "Check the API terminal."
            ),
        ) from exc
