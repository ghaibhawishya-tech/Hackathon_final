from .models import TaskInfo, Info
from typing import Dict, Any, List

def calculate_score(task: TaskInfo, route_choice: str, action_history: List[str]) -> float:
    """
    Returns a score [0.0, 1.0] for the specific routing trace on a specific task.
    Score penalizes:
    - Choosing a high cost model for a low risk/easy task
    - Choosing a low quality model for a high risk/hard task (safety penalty overrides all)
    - Not inspecting catalog and just guessing
    """
    try:
        if route_choice is None:
            return 0.01

        has_inspected = "inspect_model_catalog" in action_history
        has_estimated = "estimate_complexity_and_risk" in action_history

        score = 1.0
        
        # Process base task match
        if task.risk_level == "high":
            if route_choice != "large":
                # Safety failure is immediate major penalty
                return 0.01
        elif task.risk_level == "medium" or task.quality_requirement == "medium":
            if route_choice == "small":
                # Capability failure
                score -= 0.5
            elif route_choice == "large":
                # Overspending 
                score -= 0.2
        else: # low
            if route_choice == "medium":
                score -= 0.2
            elif route_choice == "large":
                score -= 0.5

        # Process investigation penalty
        if not has_inspected:
            score -= 0.2
        if not has_estimated:
            score -= 0.1
            
        return max(0.01, min(0.99, score))
    except Exception as e:
        # Failsafe fallback score to prevent crashes in evaluation
        print(f"[ERROR] Grader evaluation failed: {str(e)}", flush=True)
        return 0.01

def grade(*args, **kwargs) -> float:
    """
    Standard OpenEnv generic grader interface required for validator schema.
    Returns a bounded score strictly between 0 and 1.
    """
    try:
        # If openenv passes specific 'info' wrapper or trajectory final state
        if kwargs and "info" in kwargs:
            info = kwargs["info"]
            if hasattr(info, "score"): 
                return max(0.01, min(0.99, info.score))
            elif isinstance(info, dict) and "score" in info:
                return max(0.01, min(0.99, float(info["score"])))
                
        if len(args) > 0:
            trajectory = args[0]
            if isinstance(trajectory, list) and len(trajectory) > 0:
                last_step = trajectory[-1]
                if isinstance(last_step, dict) and "info" in last_step:
                    info = last_step["info"]
                    if isinstance(info, dict) and "score" in info:
                        return max(0.01, min(0.99, float(info["score"])))
    except Exception as e:
        print(f"[DEBUG] Fallback OpenEnv validation grader hit exception: {e}")
        
    return 0.50 # Failsafe valid strict score
