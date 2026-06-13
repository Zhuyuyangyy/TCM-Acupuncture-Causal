"""Analysis API router — causal inference & uplift endpoints."""
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List

from backend.models.causal_forest import CausalForest
from backend.models.acupoint_combination import AcupointMiner
from backend.analysis.uplift_modeling import TwoModelUplift

router = APIRouter(prefix="/analysis", tags=["analysis"])


# ---- Request / Response schemas ----

class CausalForestRequest(BaseModel):
    """Fit a CausalForest and return ITE / ATE estimates."""
    X: List[List[float]] = Field(..., description="Feature matrix (n x p)")
    treatment: List[int] = Field(..., description="Binary treatment indicator")
    y: List[float] = Field(..., description="Outcome vector")
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_depth: int = Field(default=5, ge=1, le=20)
    n_bootstrap: int = Field(default=50, ge=10, le=500)
    ci_level: float = Field(default=0.95, ge=0.5, le=0.999)


class CausalForestResponse(BaseModel):
    ate: float
    ate_ci_lower: float
    ate_ci_upper: float
    ite_mean: float
    ite_std: float
    n_samples: int
    n_treated: int
    n_control: int


class UpliftRequest(BaseModel):
    """Fit a TwoModel uplift estimator."""
    X: List[List[float]] = Field(..., description="Feature matrix")
    treatment: List[int] = Field(..., description="Binary treatment indicator")
    y: List[float] = Field(..., description="Outcome vector")
    learner: str = Field(default="gbm", description="'gbm' or 'rf'")
    n_estimators: int = Field(default=200, ge=10, le=1000)
    max_depth: int = Field(default=4, ge=1, le=20)


class UpliftResponse(BaseModel):
    ate: float
    ite_mean: float
    ite_std: float
    n_samples: int


class RuleMiningRequest(BaseModel):
    """Mine association rules from prescriptions."""
    prescriptions: List[List[str]] = Field(..., description="List of prescriptions (each is list of acupoint codes)")
    min_support: float = Field(default=0.01, ge=0.0, le=1.0, description="Min relative support")
    min_confidence: float = Field(default=0.1, ge=0.0, le=1.0, description="Min confidence")
    min_lift: float = Field(default=1.0, ge=0.0, description="Min lift")
    max_rules: int = Field(default=50, ge=1, le=500)


class AssociationRule(BaseModel):
    antecedent: List[str]
    consequent: List[str]
    support: float
    confidence: float
    lift: float
    count: int


class RuleMiningResponse(BaseModel):
    n_prescriptions: int
    n_rules: int
    rules: List[AssociationRule]


# ---- Endpoints ----

@router.post("/causal-forest", response_model=CausalForestResponse)
async def run_causal_forest(body: CausalForestRequest):
    """Fit a CausalForest and return ATE with CI + ITE summary."""
    X = np.array(body.X)
    treatment = np.array(body.treatment)
    y = np.array(body.y)

    if X.shape[0] != len(treatment) or X.shape[0] != len(y):
        raise HTTPException(status_code=400, detail="X, treatment, y must have same length")
    if len(set(treatment) - {0, 1}) > 0:
        raise HTTPException(status_code=400, detail="treatment must be binary (0/1)")

    cf = CausalForest(
        n_estimators=body.n_estimators,
        max_depth=body.max_depth,
        n_bootstrap=body.n_bootstrap,
        ci_level=body.ci_level,
        random_state=42,
    )
    cf.fit(X, treatment, y)

    ate_res = cf.ate_ci(X)
    ite = cf.predict_ite(X)

    return CausalForestResponse(
        ate=round(float(ate_res["ate"]), 4),
        ate_ci_lower=round(float(ate_res["ci_lower"]), 4),
        ate_ci_upper=round(float(ate_res["ci_upper"]), 4),
        ite_mean=round(float(np.mean(ite)), 4),
        ite_std=round(float(np.std(ite)), 4),
        n_samples=len(X),
        n_treated=int(treatment.sum()),
        n_control=int((1 - treatment).sum()),
    )


@router.post("/uplift", response_model=UpliftResponse)
async def run_uplift(body: UpliftRequest):
    """Fit a TwoModel uplift estimator and return ITE summary."""
    X = np.array(body.X)
    treatment = np.array(body.treatment)
    y = np.array(body.y)

    if X.shape[0] != len(treatment) or X.shape[0] != len(y):
        raise HTTPException(status_code=400, detail="X, treatment, y must have same length")

    model = TwoModelUplift(
        learner=body.learner,
        n_estimators=body.n_estimators,
        max_depth=body.max_depth,
        random_state=42,
    )
    model.fit(X, treatment, y)

    ite = model.predict_ite(X)

    return UpliftResponse(
        ate=round(model.ate(X), 4),
        ite_mean=round(float(np.mean(ite)), 4),
        ite_std=round(float(np.std(ite)), 4),
        n_samples=len(X),
    )


@router.post("/rules", response_model=RuleMiningResponse)
async def mine_association_rules(body: RuleMiningRequest):
    """Mine association rules from acupoint prescriptions."""
    if not body.prescriptions:
        raise HTTPException(status_code=400, detail="prescriptions cannot be empty")

    miner = AcupointMiner()
    miner.mine_frequent(body.prescriptions, min_support=1, max_k=3)
    rules = miner.generate_rules(
        min_support=body.min_support,
        min_confidence=body.min_confidence,
        min_lift=body.min_lift,
    )

    return RuleMiningResponse(
        n_prescriptions=len(body.prescriptions),
        n_rules=min(len(rules), body.max_rules),
        rules=[AssociationRule(**r) for r in rules[:body.max_rules]],
    )
