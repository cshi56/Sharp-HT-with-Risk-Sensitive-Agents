import os

import matplotlib.pyplot as plt
#plt.ion()

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

proportion_g = 0.1
proportion_b = 1 - proportion_g
pi_g = 0.3
pi_b = 0.8

agent_g_high = Agent(proportion_g, pi_g, w0_g, highlyRiskAverse, nullAppFunc, altAppFunc, maxPower)
agent_b_high = Agent(proportion_b, pi_b, w0_b, highlyRiskAverse, nullAppFunc, altAppFunc, maxPower)
agent_list_high = [agent_g_high, agent_b_high]

#########################################################################


### contract configurations ###


reward_0_det = [100, 75, 50, 25]
reward_1_det = 100

cost = 10

#########################################################################

num_tau = 1000
single_contract_list = [Contract(None, reward_0, reward_1_det, cost) for reward_0 in reward_0_det]

### Compute FDR Bounds ###

summary_list = [simulate_single_contract_bound(agent_list_high, single_contract, num_tau) for single_contract in single_contract_list]


#########################################################################

## Plot

figdir = "./Figure/"
figname = "fig_risk_averse_bound_rewardRatio.pdf"

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


fig_rewardRatio = plt.figure(figsize=(fig_width, fig_height))

line = {0: ":",
        1: "-.",
        2: "--",
        3: "-",
        "true": "-"}



for i in range(len(summary_list)):
    summary = summary_list[i]
    plt.plot(summary['tau'], summary['total_posnull_oracle'], color = color_oracle_list[i], linestyle = line[i], label = fr"Known-$\beta$: R0/R1 = {4-i}/4")

plt.plot(summary_list[-1]['tau'], summary_list[-1]['total_posnull_true'], color = truth_color, linestyle = line["true"], label = f"Exact Bayes FDR: R0/R1 = 1/4")
plt.fill_between(summary_list[-1]['tau'],
                 summary_list[-1]['total_posnull_true'], 
                 color = truth_color,
                 alpha = 0.3)

plt.grid(linestyle = '--', alpha=0.5)
plt.tick_params(axis='both', which='major', labelsize = axistick_fontsize)
plt.xlabel(r'Decision threshold $\tau$', fontsize = axislabel_fontsize)
plt.ylabel('False discovery rate', fontsize = axislabel_fontsize)
plt.legend(loc = "lower right", fontsize=legend_fontsize)


plt.title(r"Risk-averse:  FDR versus $\tau$", fontsize = fontlarge)


plt.savefig(figdir+figname, format="pdf", bbox_inches="tight")
plt.show()
