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
    def __init__(self, model_name: str = "Llama-3.1-8B-Instruct", tensor_parallel_size: int = 1, dtype: torch.dtype = torch.bfloat16, gpu_memory_utilization: float = 0.90, group: bool = False, using_mcts: bool = False):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.group = group
        self.action_to_name = {
            0: "stay",
            1: "right",
            2: "left",
            3: "down",
            4: "up",
            5: "interact"
        }
        self.name_to_action = {v: k for k, v in self.action_to_name.items()}
        self.using_mcts = using_mcts
        self.state_to_action_buffer = {}
        if self.group:
            self.dataset_name = "group_dataset"
        else:
            self.dataset_name = "single_agent_dataset"
    
    def convert_state_to_text(self, state):
        text = ""
        text += f"The agents' inventory is {state['agent_inventory']}.\n"
        text += f"The agents' inventory colors are {state['agent_inventory_colors']}.\n"
        text += f"The agents' location is {state['agent_locations']}.\n"
        text += f"The block colors are {state['block_colors']}.\n"
        text += f"The block locations are {state['block_locations']}.\n"
        text += f"The wall locations are {state['wall_locations']}.\n"
        return text
    
    def convert_states_actions_to_text(self, states, actions):
        state_strings = []
        action_strings = []
        for i in range(actions.shape[0]):
            state = jax.tree.map(lambda x: x[i], states)
            state_string = self.convert_state_to_text(state)
            action = [self.action_to_name[int(a)] for a in actions[i]]
            state_strings.append(f"{i+1}. State: {state_string}.")
            action_string = f"{i+1}."
            for aid, a in enumerate(action):
                action_string += f" Agent {aid}'s Action: {a}, "
            action_strings.append(action_string)
        state_action_strings = [f"{s} {a}" for s, a in zip(state_strings[:-1], action_strings[:-1])]
        state_action_strings.append(state_strings[-1])
        return "\n-------\n".join(state_action_strings)
    

    def predict_action(self, states, actions, training=False, episode_id=0, return_probs=True, agent_id=0, timestep=None):
        episode_name = f"{self.dataset_name}_{episode_id}_{agent_id}"
        text = self.convert_states_actions_to_text(states, actions)
        if timestep is not None:
            question = f"What will agent {agent_id} do at timestep {timestep + 1}?"
        else:
            question = f"What will agent {agent_id} do next?"
        num_agents = actions.shape[1]
        choices = list(self.action_to_name.values())
        
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
        choices = np.array([self.name_to_action[c] for c in choices])
        predicted_action = np.argmax(probs)
        if return_probs:
            return choices[predicted_action], probs
        else:
            return choices[predicted_action]
    
        

