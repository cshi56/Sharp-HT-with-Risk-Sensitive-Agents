import numpy as np

def normalize(array):
    
    ## array should only contain nonnegative entries
    ## normalize the array to a probability vector that sums to 1
    
    if np.sum(array) == 0:
        probability_vector = np.zeros_like(array)
    else:
        probability_vector = array / np.sum(array)
    return probability_vector


def opt_in_proportion(agent_list, contract):
    
    ## return a matrix (n_agents, n_thresholds) where each entry is 
    ## 0 if the agent does not opt in under this threshold
    ## proportion of the agents within all the agents that opt in under this threshold 
    
    # a vector of contains each agent's absolute proportion
    agent_abs_prop = np.array([agent.proportion for agent in agent_list])
    
    # a matrix (n_agents, n_thresholds) where each entry is 1 if opt in and 0 otherwise
    opt_in = np.array([agent.opt_in_contract(contract.tau, contract.R0, contract.R1, contract.C) for agent in agent_list])

    #initialize conditional proportional matrix
    opt_in_prop = np.zeros_like(opt_in, dtype=float)
    n_agent, n_tau = opt_in.shape
    
    for c in range(n_tau):
        opt_in_prop_under_tau = opt_in[:,c] * agent_abs_prop
        opt_in_conditional_prop = normalize(opt_in_prop_under_tau)
        opt_in_prop[:,c] = opt_in_conditional_prop
    
    return opt_in_prop

### Compute FDR bounds for K agents ###

def simulate_single_contract_bound(agent_list, contract, num_tau):

    # agent_list: a list of agents with ascending prior null probability and same utility function
    
    summary = {}
    
    upperTau = contract.tau_upperBound(agent_list[0])
    tau_range = np.linspace(0, upperTau, num_tau)[1:]
    contract.tau = tau_range
    
    opt_in_prop = opt_in_proportion(agent_list, contract)
    
    # a matrix (n_agents, n_thresholds) where each entry is the posnull incurred when agent i opts in under threshold j
    posnull_true_matrix = np.array([contract.true_posterior_null(agent) for agent in agent_list])
    # return the true posnull weighted by the conditional proportion of each opt-in agent under threshold j
    total_posnull_true = np.sum(opt_in_prop * posnull_true_matrix, axis=0)
    
    posnull_oracle_matrix = np.array([contract.oracle_posterior_null(agent) for agent in agent_list])
    total_posnull_oracle = np.mean(posnull_oracle_matrix, axis=0)
    total_posnull_oracle = np.maximum(total_posnull_oracle, 0) ## could be negative due to not opt-in agents
    
    posnull_imple_matrix = np.array([contract.implementable_posterior_null(agent) for agent in agent_list])
    total_posnull_imple = np.mean(posnull_imple_matrix, axis=0)
    total_posnull_imple = np.maximum(total_posnull_imple, 0)
    
    posnull_oldlinear_matrix = np.array([contract.loose_linear_bound() for agent in agent_list])
    total_posnull_oldlinear = np.mean(posnull_oldlinear_matrix, axis=0)
    total_posnull_oldlinear = np.maximum(total_posnull_oldlinear, 0)

    
    summary['tau'] = tau_range
    summary['total_posnull_true'] = total_posnull_true
    summary['total_posnull_oracle'] = total_posnull_oracle
    summary['total_posnull_imple'] = total_posnull_imple
    summary['total_posnull_oldlinear'] = total_posnull_oldlinear

    return summary



## compute all the FDR bounds for 2 agents ##
"""
def simulate_single_contract_bound(agent_g, agent_b, contract, num_tau):

    summary = {}

    upperTau = contract.tau_upperBound(agent_g)
    tau_range = np.linspace(0, upperTau, num_tau)[1:]
    contract.tau = tau_range

    opt_in_g = agent_g.opt_in_contract(contract.tau, contract.R0, contract.R1, contract.C)
    opt_in_b = agent_b.opt_in_contract(contract.tau, contract.R0, contract.R1, contract.C)

    p_g = np.where(opt_in_g+opt_in_b == 2, agent_g.proportion, 1) * opt_in_g
    p_b = np.where(opt_in_g+opt_in_b == 2, agent_b.proportion, 1) * opt_in_b

    posnull_true_g = contract.true_posterior_null(agent_g)
    posnull_true_b = contract.true_posterior_null(agent_b)
    total_posnull_true = p_g * posnull_true_g + p_b * posnull_true_b

    posnull_oracle_g = contract.oracle_posterior_null(agent_g)
    posnull_oracle_b = contract.oracle_posterior_null(agent_b)
    total_posnull_oracle = p_g * posnull_oracle_g + p_b * posnull_oracle_b

    posnull_imple_g = contract.implementable_posterior_null(agent_g)
    posnull_imple_b = contract.implementable_posterior_null(agent_b)
    total_posnull_imple = p_g * posnull_imple_g + p_b * posnull_imple_b

    posnull_oldlinear_g = contract.loose_linear_bound()
    posnull_oldlinear_b = contract.loose_linear_bound()
    total_posnull_oldlinear = p_g * posnull_oldlinear_g + p_b * posnull_oldlinear_b

    summary['tau'] = tau_range
    summary['total_posnull_true'] = total_posnull_true
    summary['total_posnull_oracle'] = total_posnull_oracle
    summary['total_posnull_imple'] = total_posnull_imple
    summary['total_posnull_oldlinear'] = total_posnull_oldlinear

    return summary
"""