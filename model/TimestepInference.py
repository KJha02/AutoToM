from .ElementExtractor import *
from .utils import *
from .BayesianInference import BayesianInferenceModel

def infer_belief_at_timestamp(
    self,
    time_variables,
    i,
    previous_belief,
    belief_name,
    variable_values_with_time,
    all_probs,
    no_observation_hypothesis,
    all_prob_estimations,
):
    # For all time stamps except for the last --> we infer belief with Bayesian Inference
    # print(no_observation_hypothesis)

    if isinstance(time_variables, list):
        var_i = time_variables[i]
    elif isinstance(time_variables, dict):  # variables at a specific timestep
        var_i = time_variables
    now_variables = []
    for key, item in var_i.items():
        if (
            key != belief_name
            and key != "All Actions"
            and key != "Ground Truth State"
        ):
            now_variables.append(item)
    if belief_name in var_i:
        now_variables.append(var_i[belief_name])
    if "BigToM" in self.dataset_name:
        context = self.story
    else:
        context = ""
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        world_rules=self.world_rules,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
    )

    try:
        results, all_prob_estimations, all_node_results = inference_model.infer("Belief", self.model_name, self.episode_name, self.init_belief)
    except Exception as e:
        print(f"Exception {e}")
        return (Variable("Previous Belief", True, False, ["NONE"], np.ones(1)), all_prob_estimations, all_probs)
    
    self.translate_and_add_node_results(self, i, all_node_results)
    previous_belief = deepcopy(var_i[belief_name])
    previous_belief.prior_probs = np.array(results)
    previous_belief.name = "Previous Belief"
    # if self.verbose:
    enh_print(
        f"After time {i}: Belief Probs Updated to {previous_belief.possible_values}, {results}"
    )
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]
    now_probs = {
        "Chunk": chunk,
        f"Probs({self.choices})": results,
    }
    all_probs.append(now_probs)
    return (previous_belief, all_prob_estimations, all_probs)
    # return previous_belief, all_prob_estimations, all_probs

def infer_last_timestamp(
    self,
    time_variables,
    i,
    inf_name,
    inf_var_name,
    now_variables,
    no_observation_hypothesis,
    variable_values_with_time,
    all_probs,
    all_prob_estimations,
):
    # Last time stamp --> we want to infer the variable we are interested in with Bayesian Inference
    if isinstance(time_variables, list):
        var_i = time_variables[i]
    elif isinstance(time_variables, dict):  # variables at a specific timestep
        var_i = time_variables
    for key, item in var_i.items():
        if key != inf_name and key != "All Actions" and key != "Ground Truth State":
            if item != "NONE":
                item.name = key
                now_variables.append(item)

    try:
        now_variables.append(var_i[inf_name])
    except Exception:
        print(f"No {inf_name}!")
        return None

    if self.verbose:
        print("chosen variables: \n\n\n", now_variables, "\n\n\n")

    if "BigToM" in self.dataset_name:
        context = self.story
    else:
        context = "" 
    inference_model = BayesianInferenceModel(
        variables=now_variables,
        context=context,
        llm=self.llm,
        verbose=self.verbose,
        inf_agent=self.inf_agent_name,
        model_name=self.model_name,
        episode_name=self.episode_name,
        dataset_name=self.dataset_name,
        K=self.K,
        answer_choices=self.choices,
        world_rules=self.world_rules,
        all_prob_estimations=all_prob_estimations,
        no_observation_hypothesis=no_observation_hypothesis,
        reduce_hypotheses=self.reduce_hypotheses,
    )

    results, all_prob_estimations, all_node_results = inference_model.infer(inf_var_name, self.model_name, self.episode_name, self.init_belief)
    self.translate_and_add_node_results(self, i, all_node_results)
    
    save_node_results(
        self.intermediate_node_results,
        self.model_name,
        self.episode_name,
        self.back_inference,
        self.reduce_hypotheses,
    )
        
    enh_print(
        f"After time {i}: {inf_name} Probs calculated as {var_i[inf_name].possible_values}, {results}"
    )
    chunk = "NONE"
    if variable_values_with_time is None:
        chunk = "NONE"
    elif i < len(variable_values_with_time) and "Chunk" in variable_values_with_time[i]:
        chunk = variable_values_with_time[i]["Chunk"]
    now_probs = {
        "Chunk": chunk,
        f"Probs({self.choices})": results,
    }
    all_probs.append(now_probs)
    return results, all_prob_estimations, all_probs
