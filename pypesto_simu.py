import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pypesto
import pypesto.optimize as optimize

# SEIR model function
def seir_model(y, t, beta, sigma, gamma):
    S, E, I, R = y
    N = S + E + I + R
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

# Read data from CSV file
data = pd.read_csv('German_case_wave_1to4.csv')
infected = data['n_confirmed'].to_numpy()
recovered = data['n_recovered'].to_numpy()
dead = data['n_death'].to_numpy()

# Total population
N = 10000

# Time vector (~3 months, 100 days)
t = np.linspace(0, 99,100)

# Initial number of infected, recovered, and dead individuals
I0 = infected[0]
R0 = recovered[0]
D0 = dead[0]
E0 = 0
S0 = N - I0 - R0 - D0

# Initial conditions vector
y0 = S0, E0, I0, R0

# Estimate parameters using PyPESTO
objective = pypesto.Objective(fun=lambda params: np.sum((odeint(seir_model, y0, t, args=tuple(params))[:, 2] - infected)**2 +
                                                         (odeint(seir_model, y0, t, args=tuple(params))[:, 3] - recovered)**2 +
                                                         (odeint(seir_model, y0, t, args=tuple(params))[:, 3] - dead)**2))
problem = pypesto.Problem(objective, lb=[0.001,0.001,0.001], ub=[5,5,5])
#optimizer = pypesto.optimize()
optimizer = optimize.ScipyOptimizer(method="ls_trf")
result = pypesto.optimize.minimize(problem=problem, optimizer=optimizer)

# Estimated parameters
params= result.optimize_result.get_for_key('x')
print("Optimized parameters:", params)

# Simulate SEIR model with estimated parameters
solution = odeint(seir_model, y0, t, args=(beta, sigma, gamma))
S, E, I, R = solution.T

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.plot(t, D, label='Dead')
plt.xlabel('Time (days)')
plt.ylabel('Number of cases')
plt.title('SEIR Model Simulation')
plt.legend()
plt.grid(True)
plt.show()
