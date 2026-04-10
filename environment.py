from typing import Tuple
from pydantic import ValidationError

import openenv
from .models import Observation, Info
from .actions import Action
from .tasks import get_task, get_model_catalog
from .graders import calculate_score

class ModelRouterEnv:
    """
    OpenEnv standard simulation for Model Router Agent.
    """
    def __init__(self, task_id: str = "easy_001"):
        try:
            self.task_id = task_id
            self.task = get_task(task_id)
        except Exception as e:
            raise ValueError(f"Failed to load task '{task_id}': {str(e)}")
        self.max_steps = 10
        self.reset()
        
    def reset(self) -> Observation:
        self.current_step = 0
        self.done = False
        self.chosen_model = None
        self.action_history = []
        self.route_justification = None
        self.step_rewards = []
        self.error_msg = None
        
        self.models_inspected = False
        self.complexity_estimated = False
        
        return self._get_obs()

    def _get_obs(self) -> Observation:
        obs = Observation(
            task_id=self.task.task_id,
            user_request=self.task.user_request,
            task_type=self.task.task_type,
            domain=self.task.domain,
            latency_budget="Standard",
            cost_budget="Strict",
            quality_requirement=self.task.quality_requirement,
            models_catalog_inspected=self.models_inspected,
            available_models=get_model_catalog() if self.models_inspected else None,
            current_step=self.current_step,
            max_steps=self.max_steps,
            chosen_model_so_far=self.chosen_model,
            last_action_error=self.error_msg,
            done=self.done
        )
        # Show risk/complexity if estimated
        if self.complexity_estimated:
            obs.risk_level = self.task.risk_level
            obs.estimated_input_length = self.task.estimated_input_length
            obs.expected_output_length = self.task.expected_output_length
        return obs
        
    def step(self, action: Action) -> Tuple[Observation, float, bool, Info]:
        if self.done:
            return self._get_obs(), 0.0, True, self._get_info()
            
        self.error_msg = None
        self.current_step += 1
        self.action_history.append(action.action_type)
        
        step_reward = 0.0
        
        try:
            if action.action_type == "inspect_model_catalog":
                if self.models_inspected:
                    step_reward -= 0.1 # penalty for repeat
                    self.error_msg = "Catalog already inspected"
                else:
                    self.models_inspected = True
                    step_reward += 0.2
                    
            elif action.action_type == "estimate_complexity_and_risk":
                if self.complexity_estimated:
                    step_reward -= 0.1
                    self.error_msg = "Complexity already estimated"
                else:
                    self.complexity_estimated = True
                    step_reward += 0.2
                    
            elif action.action_type == "choose_model":
                self.chosen_model = action.model_name
                self.route_justification = action.justification
                step_reward += 0.1 # choosing is good
                
            elif action.action_type == "submit_route":
                if self.chosen_model is None:
                    step_reward -= 0.5
                    self.error_msg = "Must choose a model before submitting."
                else:
                    self.done = True
                    
        except Exception as e:
            self.error_msg = str(e)
            step_reward -= 0.1
            
        if self.current_step >= self.max_steps:
            self.done = True
            
        # If terminal, compute final score based on graders
        if self.done:
            score = calculate_score(self.task, self.chosen_model, self.action_history)
            final_reward = score # Keep reward strictly < 1.0
            step_reward += final_reward
            
        self.step_rewards.append(step_reward)
            
        return self._get_obs(), step_reward, self.done, self._get_info()
        
    def _get_info(self) -> Info:
        return Info(
            score=calculate_score(self.task, self.chosen_model, self.action_history),
            step_rewards=self.step_rewards,
            final_choice=self.chosen_model,
            route_justification=self.route_justification
        )
