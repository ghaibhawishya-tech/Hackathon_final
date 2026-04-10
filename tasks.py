from .models import TaskInfo, ModelMetadata
import copy

MODEL_CATALOG = [
    ModelMetadata(
        model_name="small",
        tier="small",
        relative_cost=0.1,
        relative_latency=0.1,
        quality_capability="low",
        strengths=["Basic editing", "Short summaries", "Formatting"],
        weaknesses=["Complex reasoning", "Long context", "Risk detection"],
        safe_for_high_risk=False
    ),
    ModelMetadata(
        model_name="medium",
        tier="medium",
        relative_cost=1.0,
        relative_latency=0.8,
        quality_capability="medium",
        strengths=["Code snippets", "General Q&A", "Drafting"],
        weaknesses=["Extreme nuance", "Medical/Legal constraints"],
        safe_for_high_risk=False
    ),
    ModelMetadata(
        model_name="large",
        tier="large",
        relative_cost=10.0,
        relative_latency=2.5,
        quality_capability="high",
        strengths=["Deep reasoning", "Compliance/Legal", "Medical", "Nuance"],
        weaknesses=["High cost", "Slow latency"],
        safe_for_high_risk=True
    )
]

TASKS = {
    "easy_001": TaskInfo(
        task_id="easy_001",
        user_request="Can you fix the basic spelling errors in this paragraph? 'The quik brown fox jumpd.'",
        task_type="editing",
        domain="general",
        risk_level="low",
        estimated_input_length=15,
        expected_output_length=15,
        quality_requirement="low"
    ),
    "medium_001": TaskInfo(
        task_id="medium_001",
        user_request="Write a mid-sized Python script that takes a CSV file, parses the columns 'A' and 'B', and outputs the sum. Needs mild error handling.",
        task_type="coding",
        domain="software",
        risk_level="medium",
        estimated_input_length=30,
        expected_output_length=150,
        quality_requirement="medium"
    ),
    "hard_001": TaskInfo(
        task_id="hard_001",
        user_request="Please review this medical patient history and synthesize a diagnostic differential report. Also ensure HIPAA de-identification is applied.",
        task_type="reasoning_synthesis",
        domain="medical",
        risk_level="high",
        estimated_input_length=5000,
        expected_output_length=800,
        quality_requirement="high"
    ),
}

def get_task(task_id: str) -> TaskInfo:
    return copy.deepcopy(TASKS[task_id])

def get_model_catalog():
    return copy.deepcopy(MODEL_CATALOG)
