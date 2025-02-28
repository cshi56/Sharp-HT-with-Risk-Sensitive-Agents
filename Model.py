
import numpy as np

class Agent:
    def __init__(self, proportion, pi, w0, utilFunc, nullAppFunc, altAppFunc, maxPower):
        self.proportion = proportion
        self.pi = pi 
        self.w0 = w0
        self.utilFunc = utilFunc
        self.altAppFunc = altAppFunc
        self.nullAppFunc = nullAppFunc
        self.maxPower = maxPower

    def utilLossFunc(self, C):
        result = self.utilFunc(self.w0) - self.utilFunc(self.w0 - C)
        return result 
    
    def nullUtilGainFunc(self, R0, C):
        # if R0 is random, input a vector with simulated values
        nulUtilGain = self.utilFunc(self.w0 + R0 - C)
        if isinstance(nulUtilGain, np.ndarray):
            expectedNulUtilGain = np.mean(nulUtilGain)
        else:
            expectedNulUtilGain = nulUtilGain
        result = expectedNulUtilGain - self.utilFunc(self.w0 - C)
        return result
    
    def altUtilGainFunc(self, R1, C):
        # if R1 is random, input a vector with simulated values
        altUtilGain = self.utilFunc(self.w0 + R1 - C)
        if isinstance(altUtilGain, np.ndarray):
            expectedAltUtilGain = np.mean(altUtilGain)
        else:
            expectedAltUtilGain = altUtilGain
        result = expectedAltUtilGain - self.utilFunc(self.w0 - C)
        return result
    
    def compute_utility(self, tau, R0, R1, C):
        altApp = self.altAppFunc(tau)
        nullApp = self.nullAppFunc(tau)

        nullUtilGain = self.nullUtilGainFunc(R0, C)
        altUtilGain = self.altUtilGainFunc(R1, C)
        utilLoss = self.utilLossFunc(C)

        utility = self.pi * (nullApp * nullUtilGain - altApp * altUtilGain) + (altApp * altUtilGain - utilLoss)
        return utility
    
    def opt_in_contract(self, tau, R0, R1, C):
        utility = self.compute_utility(tau, R0, R1, C)
        opt_in = np.where(utility >= 0, 1, 0)
        return opt_in



class Contract:
    def __init__(self, tau, R0, R1, C):
        
        self.tau = tau
        self.R0 = np.array(R0) if isinstance(R0, list) else R0
        self.R1 = np.array(R1) if isinstance(R1, list) else R1
        self.C = C

    def tau_upperBound(self, agent):

        nullUtilGain = agent.nullUtilGainFunc(self.R0, self.C)
        utilLoss = agent.utilLossFunc(self.C)

        upperBound = utilLoss/nullUtilGain

        return upperBound

    def true_posterior_null(self, agent):

        nullApp = agent.nullAppFunc(self.tau)
        altApp = agent.altAppFunc(self.tau)

        numerator = agent.pi * nullApp
        denominator = agent.pi * nullApp + (1-agent.pi) * altApp
        posnull_true = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))

        return posnull_true
    
    def oracle_posterior_null(self, agent):
        
        nullApp = agent.nullAppFunc(self.tau)
        altApp = agent.altAppFunc(self.tau)
        
        nullUtilGain = agent.nullUtilGainFunc(self.R0, self.C)
        altUtilGain = agent.altUtilGainFunc(self.R1, self.C)
        utilLoss = agent.utilLossFunc(self.C)
        
        numerator = nullApp * altApp * altUtilGain - nullApp * utilLoss
        denominator = nullApp * altApp * (altUtilGain - nullUtilGain) + (altApp - nullApp) * utilLoss
        posnull_oracle = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))

        return posnull_oracle
    

    def implementable_posterior_null(self, agent):
        
        nullApp = agent.nullAppFunc(self.tau)
        altApp = np.ones_like(nullApp) * agent.maxPower
        
        nullUtilGain = agent.nullUtilGainFunc(self.R0, self.C)
        altUtilGain = agent.altUtilGainFunc(self.R1, self.C)
        utilLoss = agent.utilLossFunc(self.C)
        
        numerator = nullApp * altApp * altUtilGain - nullApp * utilLoss
        denominator = nullApp * altApp * (altUtilGain - nullUtilGain) + (altApp - nullApp) * utilLoss
        posnull_imple = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))

        return posnull_imple
    
    def loose_linear_bound(self):
        if isinstance(self.R1, np.ndarray):
            numerator = self.tau * np.mean(self.R1)
        else:
            numerator = self.tau * self.R1
        denominator = self.C
        posnull_loose = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0))
        
        return posnull_loose