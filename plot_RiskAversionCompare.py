import os

import matplotlib.pyplot as plt
plt.ion()

from Simulation import *
import NormalMeanTest as NMT
from Model import Agent, Contract


#######################################################################

def get_full_figname(figname):
    figdir = "./Figure"
    return os.path.join(figdir, figname)


def crra(w, coeff):
    num = w ** (1-coeff)
    denom = (1-coeff)
    result = num/denom
    return result

def nullAppFunc(tau):
    return tau

def altAppFunc(tau):
    return NMT.tau_to_beta(tau, mu, sigma)

def riskNeutral(w):
    return w

def slightlyRiskAverse(w):
    return crra(w, slightlyRiskAverseCoeff)

def highlyRiskAverse(w):
    return crra(w, highlyRiskAverseCoeff)

############################################################################


### Define simulation configurations ###

### agent configurations ###

w0_g = 20
w0_b = 20
maxPower = 1
slightlyRiskAverseCoeff = 0.35
highlyRiskAverseCoeff = 0.7
mu = 1
sigma = 1

#########################################################################

### contract configurations ###

ctype = "flora_contract"
#ctype = "martin_contract"
#ctype = "one_step"
#ctype = "super_close"

if ctype == "flora_contract":

    proportion_g = 0.1
    proportion_b = 1 - proportion_g
    pi_g = 0.3
    pi_b = 0.8
    reward_0 = 100
    reward_1 = 100
    cost = 10
    figname_neutral = "fig_risk_neutral_bound_r100.pdf"
    figname_averse = "fig_risk_averse_bound_r100.pdf"

    
elif ctype == "martin_contract":
    proportion_g = 0.1
    proportion_b = 1 - proportion_g
    pi_g = 0.3
    pi_b = 0.8
    reward_0 = 25
    reward_1 = 25
    cost = 10
    figname_neutral = "fig_risk_neutral_bound_r25.pdf"
    figname_averse = "fig_risk_averse_bound_r25.pdf"


elif ctype == "super_close":
    proportion_g = 0.005
    proportion_b = 1 - proportion_g
    pi_g = 0.3
    pi_b = 0.8
    reward_0 = 25
    reward_1 = 25
    cost = 10
    figname_neutral = "fig_risk_neutral_bound_superclose.pdf"
    figname_averse = "fig_risk_averse_bound_superclose.pdf"

elif ctype == "one_step":   
     proportion_g = 1.0
     proportion_b = 1 - proportion_g
     pi_g = 0.3
     pi_b = 0.3
     reward_0 = 25
     reward_1 = 25
     cost = 10
     figname_neutral = "fig_risk_neutral_bound_onestep.pdf"
     figname_averse = "fig_risk_averse_bound_onestep.pdf"

#####################################################################

# Initialize agents
agent_g_riskNeutral = Agent(proportion_g, pi_g, w0_g, riskNeutral, nullAppFunc, altAppFunc, maxPower)
agent_b_riskNeutral = Agent(proportion_b, pi_b, w0_b, riskNeutral, nullAppFunc, altAppFunc, maxPower)
agent_g_slight = Agent(proportion_g, pi_g, w0_g, slightlyRiskAverse, nullAppFunc, altAppFunc, maxPower)
agent_b_slight = Agent(proportion_b, pi_b, w0_b, slightlyRiskAverse, nullAppFunc, altAppFunc, maxPower)
agent_g_high = Agent(proportion_g, pi_g, w0_g, highlyRiskAverse, nullAppFunc, altAppFunc, maxPower)
agent_b_high = Agent(proportion_b, pi_b, w0_b, highlyRiskAverse, nullAppFunc, altAppFunc, maxPower)

agent_list_riskNeutral = [agent_g_riskNeutral, agent_b_riskNeutral]
agent_list_slight = [agent_g_slight, agent_b_slight]
agent_list_high = [agent_g_high, agent_b_high]

##################################################################

num_tau = 1000
single_contract = Contract(None, reward_0, reward_1, cost)


### Compute FDR Bounds ###

summaryRiskNeutral = simulate_single_contract_bound(agent_list_riskNeutral, single_contract, num_tau)
summarySlightRisk = simulate_single_contract_bound(agent_list_slight, single_contract, num_tau)
summaryHighRisk = simulate_single_contract_bound(agent_list_high, single_contract, num_tau)

##################################################################

### Plot ##

#fig_height = 5
#fig_width = 15
fig_height = 7
fig_width = 7

axislabel_fontsize = 15
axistick_fontsize = 15
legend_fontsize = 12
fontmed = 16
fontlarge = 20


color_oracle_list = ["#0b5c8f", "#1e6ea7", "#2d80bf", "#3c93d8", "#4aa6f2"] ## color gradient decreases
truth_color = "#ffa600"
imple_color = "#c83865"
oldLinear_color = "#7f2d6f"


fig_averse = plt.figure(figsize=(fig_width, fig_height))

line = {"neutral": "-.",
        "slight": "--",
        "oracle": "-",
        "high": "-"}
        
plt.plot(summaryRiskNeutral['tau'],
         summaryRiskNeutral['total_posnull_oracle'],
         color = color_oracle_list[4],
         linestyle = line["neutral"],
         label = r"Known-$\beta$: $\gamma = 0$")
plt.plot(summarySlightRisk['tau'],
         summarySlightRisk['total_posnull_oracle'],
         color = color_oracle_list[2],
         linestyle = line["slight"],
         label = r"Known-$\beta$: $\gamma = 0.35$")
plt.plot(summaryHighRisk['tau'],
         summaryHighRisk['total_posnull_oracle'],
         color = color_oracle_list[0],
         linestyle = line["high"],
         label = r"Known-$\beta$: $\gamma = 0.7$")
plt.plot(summaryHighRisk['tau'],
         summaryHighRisk['total_posnull_true'],
         color = truth_color,
         linestyle = line["oracle"],
         label = r"Exact Bayes FDR: $\gamma = 0.7$")

plt.fill_between(summaryHighRisk['tau'],
                 summaryHighRisk['total_posnull_true'],
                 color = truth_color,
                 alpha = 0.3)

## ensure same font as latex
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rc('axes', titlesize=14, labelsize=12)

plt.xlabel(r"Decision threshold $\tau$", fontsize = fontmed)
plt.ylabel(r"False discovery rate", fontsize = fontmed)
plt.title(r"Risk-averse:  FDR versus $\tau$", fontsize = fontlarge)
plt.grid(linestyle = '--', alpha=0.5)
plt.legend(loc = "upper left", fontsize=legend_fontsize)
plt.tick_params(axis='both', labelsize=axistick_fontsize)

plt.tight_layout()

figname = get_full_figname(figname_averse)
plt.savefig(figname, format="pdf", bbox_inches="tight")
