from vllm import LLM, SamplingParams
import torch
import pandas as pd
import time
import os
from rich.progress import track
import jax
from .model.ProbSolver import ProblemSolver, argmax, argmin
import numpy as np
import openai
import os

openai_api_key = os.environ["OPENAI_API_KEY"]

class AutoToM:
    def __init__(self, model_name: str = "Llama-3.1-8B-Instruct", tensor_parallel_size: int = 1, dtype: torch.dtype = torch.bfloat16, gpu_memory_utilization: float = 0.90, group: bool = False, partnr: bool = False):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization

        self.action_to_name = {
            0: "stay",
            1: "right",
            2: "left",
            3: "down",
            4: "up",
            5: "interact"
        }
        self.name_to_action = {v: k for k, v in self.action_to_name.items()}
        if group:
            self.dataset_name = "group_partnr_dataset"
        else:
            self.dataset_name = "single_agent_partnr_dataset"
    
    def convert_state_to_text(self, state):
        text = ""
        text += f"Scene Graph: {state['scene_graph']}\n"
        text += f"Agent State: {state['agent_state']}\n"
        return text
    
    def convert_states_actions_to_text(self, states, actions):
        state_strings = []
        action_strings = []
        tools = set()
        tool_descriptions = ""
        for i in range(len(actions)):
            state = states[i]
            action = actions[i]
            state_string = self.convert_state_to_text(state)
            state_strings.append(state_string)
            action_string = f"{i+1}. Agent 0 Action: {action}"
            action_strings.append(action_string)
            tools = tools.union(set(state['tool_list']))
            tool_descriptions += state['tool_descriptions']
        state_action_strings = [f"{s} {a}" for s, a in zip(state_strings[:-2], action_strings[:-2])]
        state_action_strings.append(state_strings[-2])
        return "\n-------\n".join(state_action_strings), tools, tool_descriptions
    

    def predict_action(self, states, actions, training=False, episode_id=0, return_probs=True, agent_id=0):
        episode_name = f"{self.dataset_name}_{episode_id}"
        text, tools, tool_descriptions = self.convert_states_actions_to_text(states, actions)
        question = f"What tool will agent {agent_id} use next?"
        num_agents = 1
        choices = [a for a in tools]
        choices.sort()
        
        verbose = False
        
        solver = ProblemSolver(
                story=text,
                question=question,
                choices=choices,
                K=1,
                assigned_model=[],
                model_name="automated",
                episode_name=episode_name,
                llm=self.model_name,
                hypo_llm=self.model_name,
                verbose=verbose,
                dataset_name=self.dataset_name,
                hypo_method="guided",
                nested=False,
                video_id=None,
                answerfunc=argmax,
                back_inference=True,
                reduce_hypotheses=True,
                precomputed_states=None,
                precomputed_actions=None,
                prev_hyp=None,
                no_model_adjustment=False,
                recursion_depth=None
            )

        final_probs, model_record = solver.solve()
        probs = np.array(final_probs)
        predicted_action = np.argmax(probs)
        if return_probs:
            return choices[predicted_action], probs, choices
        else:
            return choices[predicted_action]
    
        

