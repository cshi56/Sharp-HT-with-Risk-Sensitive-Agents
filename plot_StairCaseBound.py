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


def create_agent_prop(n_agent, ratio, start):
    prop_list = np.zeros(n_agent)
    prop_list[0] = start
    for i in range(1, n_agent):
        cumsum = np.sum(prop_list[:i])
        prop_list[i] = (cumsum * ratio)/(1-ratio)
    result = normalize(prop_list)
    return result

############################################################################

############################################################################


### Define simulation configurations ###

### agent configurations ###

w0 = 20
maxPower = 1
slightlyRiskAverseCoeff = 0.35
highlyRiskAverseCoeff = 0.7
mu = 1
sigma = 1

#########################################################################

### contract configurations ###


reward_0 = 25
reward_1 = 25
cost = 10

#####################################################################

## initialize agents

n_agent = 20 # 40
ratio = 0.99
pi_start = 0.02
pi_end = 0.97

agent_prop_list = create_agent_prop(n_agent, ratio, start=1)

agent_pi_list = np.linspace(pi_start, pi_end, n_agent)

agent_list = [Agent(agent_prop_list[i], agent_pi_list[i], w0, riskNeutral, nullAppFunc, altAppFunc, maxPower) for i in range(n_agent)]

#####################################################################

num_tau = 1000
single_contract = Contract(None, reward_0, reward_1, cost)

### Compute FDR Bounds ###

summaryRiskNeutral = simulate_single_contract_bound(agent_list, single_contract, num_tau)

#####################################################################


### Plot ##

figname = f"fig_risk_neutral_bound_K{n_agent}.pdf"

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

# Do risk neutral figure
fig_neutral = plt.figure(figsize=(fig_width, fig_height))

line = {"bates": "-.",
        "imple": "--",
        "oracle": "-",
        "true": "-"}


plt.plot(summaryRiskNeutral['tau'],
         summaryRiskNeutral['total_posnull_oldlinear'],
         color = oldLinear_color,
         linestyle = line["bates"],
         label = r"BatesEtAl23")

                         
plt.plot(summaryRiskNeutral['tau'],
         summaryRiskNeutral['total_posnull_imple'],
         color = imple_color,
         linestyle = line["imple"],
         label = r"Bound II: Unknown-$\beta$")
                         
plt.plot(summaryRiskNeutral['tau'],
          summaryRiskNeutral['total_posnull_oracle'],
          color = color_oracle_list[0],
         linestyle = line["oracle"],
          label = r"Bound I: Known-$\beta$")
                         
plt.plot(summaryRiskNeutral['tau'],
         summaryRiskNeutral['total_posnull_true'],
         color = truth_color,
         linestyle = line["true"],
         label = r"Exact Bayes FDR")
plt.fill_between(summaryRiskNeutral['tau'],
                 summaryRiskNeutral['total_posnull_true'],
                 color = truth_color,
                 alpha = 0.3)


## ensure same font as latex
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
                 
plt.legend(loc = "upper left", fontsize = legend_fontsize)
plt.grid(linestyle = '--', alpha=0.5)
plt.tick_params(axis='both', labelsize = axistick_fontsize)

plt.xlabel(r"Decision threshold $\tau$", fontsize = fontmed)
plt.ylabel(r"False discovery rate", fontsize = fontmed)
plt.title(r"Risk-neutral:  FDR versus $\tau$", fontsize = fontlarge)

plt.tight_layout()

figname = get_full_figname(figname)
plt.savefig(figname, format="pdf", bbox_inches="tight")

