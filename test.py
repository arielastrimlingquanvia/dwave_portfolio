import pandas as pd
import numpy as np
from itertools import product
import json
import random

from dimod import Integer
from dimod import quicksum
from dimod import ConstrainedQuadraticModel, DiscreteQuadraticModel
from dimod import ExactCQMSolver


df = pd.read_csv('data/crypto_data.csv').set_index('Month')
budget = 1000
bin_size=10
alpha=0.005
max_risk=None
min_return=None
stocks = df.columns
model = {}
print(df)

max_num_shares = (budget/df.iloc[-1]).astype(int)
print('max shares')
print(max_num_shares)

shares_intervals = {}
for stock in stocks:
    if max_num_shares[stock]+1 <= bin_size:
        shares_intervals[stock] = list(range(max_num_shares[stock] + 1))
    else:
        span = (max_num_shares[stock]+1) / bin_size
        shares_intervals[stock] = [int(i*span) for i in range(bin_size)]
print(shares_intervals)

price = df.iloc[-1]
monthly_returns = df[list(stocks)].pct_change().iloc[1:]
avg_monthly_returns = monthly_returns.mean(axis=0)
covariance_matrix = monthly_returns.cov()
print(avg_monthly_returns)
print(covariance_matrix)

# Instantiating the CQM object
cqm = ConstrainedQuadraticModel()

# Defining and adding variables to the CQM model
x = {s: Integer("%s" %s, lower_bound=0,
                upper_bound=max_num_shares[s]) for s in stocks}

# Defining risk expression
risk = 0
for s1, s2 in product(stocks, stocks):
    coeff = (covariance_matrix[s1][s2] * price[s1] * price[s2])
    risk = risk + coeff*x[s1]*x[s2]

# Defining the returns expression
returns = 0
for s in stocks:
    returns = returns + price[s] * avg_monthly_returns[s] * x[s]

print('risk and returns')
print(risk)
print(returns)

# Adding budget constraint
cqm.add_constraint(quicksum([x[s]*price[s] for s in stocks])
                   <= budget, label='upper_budget')
cqm.add_constraint(quicksum([x[s]*price[s] for s in stocks])
                   >= 0.997*budget, label='lower_budget')

if max_risk:
    # Adding maximum risk constraint
    cqm.add_constraint(risk <= max_risk, label='max_risk')

    # Objective: maximize return
    cqm.set_objective(-1*returns)
elif min_return:
    # Adding minimum returns constraint
    cqm.add_constraint(returns >= min_return, label='min_return')

    # Objective: minimize risk
    cqm.set_objective(risk)
else:
    # Objective: minimize mean-variance expression
    cqm.set_objective(alpha*risk - returns)

cqm.substitute_self_loops()

model['CQM'] = cqm

print('model')
print(cqm)
# print(cqm.sample)

# sampler = ExactCQMSolver()
# sampler_set = sampler.sample_cqm(cqm)
# print(sampler_set.lowest())
# res_df = pd.DataFrame(sampler_set)
# res_df.to_csv('results_exact_cqm.csv')

# import dimod
# bqm, invert = dimod.cqm_to_bqm(cqm)
# sampler_set = dimod.ExactSolver().sample(bqm)
# res_df = pd.DataFrame(sampler_set)
# res_df.to_csv('results_bqm.csv')

# TODO works hybrid
# from dwave.system import  LeapHybridCQMSampler
# sampler = LeapHybridCQMSampler()
# sampler_set = sampler.sample_cqm(cqm)
# print(sampler_set)
# res_df = pd.DataFrame(sampler_set)
# res_df.to_csv('results.csv')

# from dwave.system.samplers import DWaveSampler
#
# sampler = DWaveSampler()
#
# sample_set = sampler.sample(cqm)
#
# n_samples = len(sample_set.record)
#
# feasible_samples = sample_set.filter(lambda d: d.is_feasible)
#
# print(feasible_samples)