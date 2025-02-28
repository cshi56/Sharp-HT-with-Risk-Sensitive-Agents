from Simulation import *
import NormalMeanTest as NMT
from Model import Agent, Contract

### Agents configurations ###

def crra(w, coeff):
    num = w ** (1-coeff)
    denom = (1-coeff)
    result = num/denom
    return result

def nullAppFunc(tau):
    return tau
def altAppFunc(tau):
    return None

def riskNeutral(w):
    return w
def slightlyRiskAverse(w):
    return crra(w, slightlyRiskAverseCoeff)
def highlyRiskAverse(w):
    return crra(w, highlyRiskAverseCoeff)

proportion = 1
pi = None
w0 = 5
maxPower = 1
slightlyRiskAverseCoeff = 0.35
highlyRiskAverseCoeff = 0.7

agent_riskNeutral = Agent(proportion, pi, w0, riskNeutral, nullAppFunc, altAppFunc, maxPower)
agent_slight = Agent(proportion, pi, w0, slightlyRiskAverse, nullAppFunc, altAppFunc, maxPower)
agent_high = Agent(proportion, pi, w0, highlyRiskAverse, nullAppFunc, altAppFunc, maxPower)

### Contract configurations ###

reward_0 = 0.8
reward_1_list = [1, 25, 50]
#cost = 0.3
cost = 0.2

standard_tau = 0.000625
modernized_tau = 0.005
accelerated_tau = 0.05

standard_contract_list = [Contract(standard_tau, reward_0, reward_1, cost) for reward_1 in reward_1_list]
modernized_contract_list = [Contract(modernized_tau, reward_0, reward_1, cost) for reward_1 in reward_1_list]
accelerated_contract_list = [Contract(accelerated_tau, reward_0, reward_1, cost) for reward_1 in reward_1_list]

all_contracts_list = standard_contract_list + modernized_contract_list + accelerated_contract_list


### Compute FDR table ###

n_row = 9
n_col = 4
FDA_table = np.zeros((n_row,n_col))

oldLinearBound = np.zeros(n_row)
riskNeutralBound = np.zeros(n_row)
slightlyRiskAverseBound = np.zeros(n_row)
highlyRiskAverseBound = np.zeros(n_row)


for i in range(len(all_contracts_list)):
    contract = all_contracts_list[i]
    oldLinearBound[i] = contract.loose_linear_bound()
    riskNeutralBound[i] = contract.implementable_posterior_null(agent_riskNeutral)
    slightlyRiskAverseBound[i] = contract.implementable_posterior_null(agent_slight)
    highlyRiskAverseBound[i] = contract.implementable_posterior_null(agent_high)

FDA_table[:,0] = oldLinearBound
FDA_table[:,1] = riskNeutralBound
FDA_table[:,2] = slightlyRiskAverseBound
FDA_table[:,3] = highlyRiskAverseBound

## suppress scientific number format
np.set_printoptions(suppress=True)

## print FDR bounds in the unit of (%)
print(np.round(FDA_table * 100, 2)) 
