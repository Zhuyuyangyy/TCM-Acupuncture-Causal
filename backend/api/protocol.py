"""Protocol management API router."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from backend.models.target_trial import AcupunctureProtocol, AcupunctureTargetTrial

router = APIRouter(prefix="/protocol", tags=["protocol"])

# In-memory protocol store (could be replaced with DB)
_protocols: Dict[str, AcupunctureProtocol] = {}


class ProtocolCreate(BaseModel):
    """Request body for creating a new acupuncture protocol."""
    condition: str = Field(..., description="Clinical condition, e.g. 'chronic_pain'")
    eligibility: Dict = Field(default_factory=dict, description="Inclusion/exclusion criteria")
    treatment: str = Field(..., description="Treatment definition (acupuncture arm)")
    control: str = Field(..., description="Control arm definition")
    primary_outcome: str = Field(..., description="Primary outcome measure")
    follow_up_weeks: int = Field(default=12, ge=1, le=52)


class ProtocolResponse(BaseModel):
    id: str
    condition: str
    eligibility: Dict
    treatment: str
    control: str
    primary_outcome: str
    follow_up_weeks: int


@router.post("/", response_model=ProtocolResponse, status_code=201)
async def create_protocol(body: ProtocolCreate):
    """Create a new acupuncture target trial protocol."""
    pid = f"proto_{len(_protocols) + 1:04d}"
    proto = AcupunctureProtocol(
        condition=body.condition,
        eligibility=body.eligibility,
        treatment=body.treatment,
        control=body.control,
        primary_outcome=body.primary_outcome,
        follow_up_weeks=body.follow_up_weeks,
    )
    _protocols[pid] = proto
    return ProtocolResponse(id=pid, **body.dict())


@router.get("/", response_model=List[ProtocolResponse])
async def list_protocols():
    """List all registered protocols."""
    results = []
    for pid, p in _protocols.items():
        results.append(ProtocolResponse(
            id=pid,
            condition=p.condition,
            eligibility=p.eligibility,
            treatment=p.treatment,
            control=p.control,
            primary_outcome=p.primary_outcome,
            follow_up_weeks=p.follow_up_weeks,
        ))
    return results


@router.get("/{protocol_id}", response_model=ProtocolResponse)
async def get_protocol(protocol_id: str):
    """Get a specific protocol by ID."""
    if protocol_id not in _protocols:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id!r} not found")
    p = _protocols[protocol_id]
    return ProtocolResponse(
        id=protocol_id,
        condition=p.condition,
        eligibility=p.eligibility,
        treatment=p.treatment,
        control=p.control,
        primary_outcome=p.primary_outcome,
        follow_up_weeks=p.follow_up_weeks,
    )


@router.delete("/{protocol_id}", status_code=204)
async def delete_protocol(protocol_id: str):
    """Delete a protocol."""
    if protocol_id not in _protocols:
        raise HTTPException(status_code=404, detail=f"Protocol {protocol_id!r} not found")
    del _protocols[protocol_id]
