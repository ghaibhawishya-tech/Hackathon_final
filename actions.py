from pydantic import BaseModel, Field
from typing import Optional, Literal, Union

# Action Payloads

class InspectModelCatalog(BaseModel):
    action_type: Literal["inspect_model_catalog"] = "inspect_model_catalog"

class EstimateComplexityAndRisk(BaseModel):
    action_type: Literal["estimate_complexity_and_risk"] = "estimate_complexity_and_risk"
    complexity_reasoning: str
    risk_reasoning: str

class ChooseModel(BaseModel):
    action_type: Literal["choose_model"] = "choose_model"
    model_name: Literal["small", "medium", "large"]
    justification: str

class SubmitRoute(BaseModel):
    action_type: Literal["submit_route"] = "submit_route"

# Combined Action Type
Action = Union[
    InspectModelCatalog,
    EstimateComplexityAndRisk,
    ChooseModel,
    SubmitRoute
]
