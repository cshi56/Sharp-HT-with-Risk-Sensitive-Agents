import os

import matplotlib.pyplot as plt
#plt.ion()

from Simulation import *
import NormalMeanTest as NMT
from Model import Agent, Contract
from scipy.stats import truncnorm

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

## null reward configuration

reward_0_det = 50

reward_0_lb_med = 20
reward_0_ub_med = 80
reward_0_mu_med, reward_0_sigma_med = 50, 25

reward_0_lb_high = 0
reward_0_ub_high = 100
reward_0_mu_high, reward_0_sigma_high = 50, 35


reward_0_lb_std_med = (reward_0_lb_med - reward_0_mu_med) / reward_0_sigma_med
reward_0_ub_std_med = (reward_0_ub_med - reward_0_mu_med) / reward_0_sigma_med

reward_0_lb_std_high = (reward_0_lb_high - reward_0_mu_high) / reward_0_sigma_high
reward_0_ub_std_high = (reward_0_ub_high - reward_0_mu_high) / reward_0_sigma_high

reward_0_med = truncnorm.rvs(reward_0_lb_std_med, reward_0_ub_std_med, loc=reward_0_mu_med, scale=reward_0_sigma_med, size=10000)
reward_0_high = truncnorm.rvs(reward_0_lb_std_high, reward_0_ub_std_high, loc=reward_0_mu_high, scale=reward_0_sigma_high, size=10000)


# check if mean consistent
#print(np.mean(reward_0_med))
#print(np.mean(reward_0_high))

## non-null reward configuration

reward_1_det = 150

reward_1_lb_med = 120
reward_1_ub_med = 180
reward_1_mu_med, reward_1_sigma_med = 150, 25

reward_1_lb_high = 100
reward_1_ub_high = 200
reward_1_mu_high, reward_1_sigma_high = 150, 35


reward_1_lb_std_med = (reward_1_lb_med - reward_1_mu_med) / reward_1_sigma_med
reward_1_ub_std_med = (reward_1_ub_med - reward_1_mu_med) / reward_1_sigma_med

reward_1_lb_std_high = (reward_1_lb_high - reward_1_mu_high) / reward_1_sigma_high
reward_1_ub_std_high = (reward_1_ub_high - reward_1_mu_high) / reward_1_sigma_high

reward_1_med = truncnorm.rvs(reward_1_lb_std_med, reward_1_ub_std_med, loc=reward_1_mu_med, scale=reward_1_sigma_med, size=10000)
reward_1_high = truncnorm.rvs(reward_1_lb_std_high, reward_1_ub_std_high, loc=reward_1_mu_high, scale=reward_1_sigma_high, size=10000)


# check if mean consistent
#print(np.mean(reward_1_med))
#print(np.mean(reward_1_high))

cost = 10

#########################################################################


num_tau = 1000
single_contract_list = [Contract(None, reward_0_det, reward_1_det, cost),
                        Contract(None, reward_0_med, reward_1_med, cost),
                        Contract(None, reward_0_high, reward_1_high, cost)]

### Compute FDR Bounds ###

summary_list = [simulate_single_contract_bound(agent_list_high, single_contract, num_tau) for single_contract in single_contract_list]

#########################################################################

### Plot ##

figdir = "./Figure/"
figname = "fig_risk_averse_bound_randomReward.pdf"

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

det_color = "#d7301f"


fig_random = plt.figure(figsize=(fig_width, fig_height))

line = {0: "-.",
        1: "--",
        2: "-",
        "true": "-"}

stochastic_level = ["deterministic", "slightly stochastic", "highly stochastic"]


for i in range(len(summary_list)):
    summary = summary_list[i]
    plt.plot(summary['tau'], summary['total_posnull_oracle'], color = color_oracle_list[i], linestyle = line[i], label = rf"Known-$\beta$: {stochastic_level[i]}")

plt.plot(summary_list[2]['tau'], summary_list[2]['total_posnull_true'], color = truth_color, linestyle = line["true"], label = rf"Exact Bayes FDR: {stochastic_level[2]}")

plt.fill_between(summary_list[2]['tau'],
                 summary_list[2]['total_posnull_true'], 
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