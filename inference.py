import asyncio
import os
import json
import textwrap
from typing import List, Optional

from openai import AsyncOpenAI

from model_router_openenv.environment import ModelRouterEnv
from model_router_openenv.actions import InspectModelCatalog, EstimateComplexityAndRisk, ChooseModel, SubmitRoute

IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL")
if API_BASE_URL is None and "HF_TOKEN" in os.environ:
    API_BASE_URL = "https://router.huggingface.co/v1"
elif API_BASE_URL is None:
    API_BASE_URL = "https://api.openai.com/v1"

MODEL_NAME = os.getenv("MODEL_NAME")
if MODEL_NAME is None and "HF_TOKEN" in os.environ:
    MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
elif MODEL_NAME is None:
    MODEL_NAME = "gpt-4o"

TASK_NAME = os.getenv("MODEL_ROUTER_OPENENV_TASK", "easy_001")
BENCHMARK = os.getenv("MODEL_ROUTER_OPENENV_BENCHMARK", "model_router_openenv")
MAX_STEPS = 10
TEMPERATURE = 0.0
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI router agent. Your goal is to choose the optimal model size (small, medium, large) for a task.
    You want to minimize cost but ensure high enough quality and absolute safety for high-risk tasks.
    
    Available actions:
    1. {"action_type": "inspect_model_catalog"}
    2. {"action_type": "estimate_complexity_and_risk", "complexity_reasoning": "...", "risk_reasoning": "..."}
    3. {"action_type": "choose_model", "model_name": "small|medium|large", "justification": "..."}
    4. {"action_type": "submit_route"}

    Rules:
    - You should usually inspect catalog and estimate complexity first.
    - Choose a model. High risk must use large. Easy/low risk use small. Medium uses medium.
    - Finally, submit_route.
    Output ONLY valid JSON representing exactly one action.
    """
).strip()

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def build_user_prompt(step: int, obs_dict: dict, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Current observation:
        {json.dumps(obs_dict, indent=2)}
        
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        
        What is your next action? Reply ONLY with valid JSON.
        """
    ).strip()

async def get_model_action(client: AsyncOpenAI, step: int, obs_dict: dict, last_reward: float, history: List[str]) -> dict:
    user_prompt = build_user_prompt(step, obs_dict, last_reward, history)
    try:
        completion = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        if text.startswith("```json"):
            text = text[7:-3]
        elif text.startswith("```"):
            text = text[3:-3]
        return json.loads(text.strip())
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "inspect_model_catalog"}

async def main() -> None:
    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Note: Using native class instead of async OpenEnv wrapper for simplicity, since it's fully sync natively
    env = ModelRouterEnv(task_id=TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        try:
            obs = env.reset()
            last_reward = 0.0

            for step in range(1, MAX_STEPS + 1):
                if env.done:
                    break

                action_dict = await get_model_action(client, step, obs.model_dump(), last_reward, history)
                action_type = action_dict.get("action_type")
                
                # parse to object
                if action_type == "inspect_model_catalog":
                    action = InspectModelCatalog()
                elif action_type == "estimate_complexity_and_risk":
                    action = EstimateComplexityAndRisk(
                        complexity_reasoning=action_dict.get("complexity_reasoning", ""),
                        risk_reasoning=action_dict.get("risk_reasoning", "")
                    )
                elif action_type == "choose_model":
                    action = ChooseModel(
                        model_name=action_dict.get("model_name", "small"),
                        justification=action_dict.get("justification", "")
                    )
                elif action_type == "submit_route":
                    action = SubmitRoute()
                else:
                    action = InspectModelCatalog() # fallback

                obs, reward, done, info = env.step(action)
                error = obs.last_action_error

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                log_step(step=step, action=action.action_type, reward=reward, done=done, error=error)

                history.append(f"Step {step}: {action.action_type!r} -> reward {reward:+.2f}")

                if done:
                    score = info.score
                    break
                    
            # ensure normalized bounds check
            score = min(max(score, 0.01), 0.99)
            success = score >= SUCCESS_SCORE_THRESHOLD
        except Exception as eval_e:
            print(f"[ERROR] Inference loop crashed: {eval_e}", flush=True)
            success = False
            score = 0.01

    finally:
        # cleanup if needed
        try:
            if hasattr(env, 'close') and callable(getattr(env, 'close')):
                env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
