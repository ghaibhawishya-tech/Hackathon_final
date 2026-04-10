from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ModelMetadata(BaseModel):
    model_name: str
    tier: Literal["small", "medium", "large"]
    relative_cost: float
    relative_latency: float
    quality_capability: str
    strengths: List[str]
    weaknesses: List[str]
    safe_for_high_risk: bool

class TaskInfo(BaseModel):
    task_id: str
    user_request: str
    task_type: str
    domain: str
    risk_level: Literal["low", "medium", "high"]
    estimated_input_length: int
    expected_output_length: int
    quality_requirement: Literal["low", "medium", "high"]

class Observation(BaseModel):
    task_id: str
    user_request: str
    task_type: str
    domain: str
    # Certain things might be hidden initially until inspected
    risk_level: Optional[str] = None
    estimated_input_length: Optional[int] = None
    expected_output_length: Optional[int] = None
    latency_budget: str
    cost_budget: str
    quality_requirement: str
    
    # Context
    models_catalog_inspected: bool = False
    available_models: Optional[List[ModelMetadata]] = None
    
    current_step: int
    max_steps: int
    chosen_model_so_far: Optional[str] = None
    last_action_error: Optional[str] = None
    done: bool

class Info(BaseModel):
    score: float
    step_rewards: List[float]
    final_choice: Optional[str] = None
    route_justification: Optional[str] = None
