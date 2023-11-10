import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import minimize  # Import optimization function
from derivative import derivative_rhs

data = pd.read_csv(r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\German_case_.csv')

#---------------------------------------------------------------------------------------------------

# some constants for initialization
contacts = 6  # number of contacts in standard case# max 16 # wave 1
transmission_prob = 0.716  # transmission probability
exposed_period = 4.6  # 5.2 incubation period #Mean exposed period
asymptomatic_period = 2.17  # from A to I (2.67-4.0) upto(6.1-8.9)#Mean asymptomatic infectious period# reference 46 p9
infectious_period = 5  #(5.6-8.4) #Mean infectious period
isolated_period = 10  # Mean isolation
reducing_transmission = 0.859
prob_asymptomatic = 0.9  # probability of being asymptomatic once exposed# reference 46 p9
prob_isolated_asy = 0.2  # probability of being
prob_dead = 0.1  # probability of dying
test_asy = 0.135
dev_symp = 0.135  # which makes the (1 - test_asy - dev_symp)=0.73
prob_quarant_inf = 0.6  # proportion from I into F(quarantine) ~1 ; 0.9? ; reference 46 p9
mortality_isolated = 0.015  # make the rest goes to recovered# reference 58 p7
mortality_infected = 0.3#? (1 - mortality_infected - prob_quarant_inf)
#tmax = 90  # maximum simulation day

#fslarge = 20
#fssmall = 16
total_population = 50000  # Total number of individuals
E0 = 20  # Initial number of exposed individuals
A0 = 0  # Initial number of asymptomatic infectious individuals
I0 = 10  # Initial number of infected symptomatic individuals
F0 = 0  # Initial number of individuals in isolation
R0 = 0  # Initial number of recovered individuals
D0 = 0  # Initial number of death cases
S0 = total_population - E0 - A0 - I0 - F0 - R0 - D0  # Susceptible individuals
t=data['Date']
initial_conditions = [S0, E0, A0, I0, F0, R0, D0]
#-----------------------
solution_seaifrd = integrate.solve_ivp(derivative_rhs, t,initial_conditions, t_eval=t,
                                        args=(contacts, transmission_prob, total_population, reducing_transmission,
                                              exposed_period, asymptomatic_period, infectious_period,
                                              isolated_period, prob_asymptomatic,
                                              prob_quarant_inf, test_asy, dev_symp, mortality_isolated,mortality_infected), method='RK45')
#---------------------------------------------------------------------------------------------
#t = np.linspace(0, tmax, (tmax + 1) * 10)


params=[transmission_prob, total_population, reducing_transmission,exposed_period, asymptomatic_period, infectious_period, isolated_period,prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected]

def objective_function(transmission_prob, total_population, reducing_transmission,exposed_period, asymptomatic_period, infectious_period, isolated_period,prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):

    predictions =  derivative_rhs(t, initial_conditions, contacts, transmission_prob, total_population, reducing_transmission,exposed_period, asymptomatic_period, infectious_period, isolated_period,prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)
    #model(data['Date'], beta, gamma, other_params)  # Calculate model predictions

    squared_diff = np.sum((data['Infection_case'] - predictions['Infection_case']) ** 2)
    squared_diff += np.sum((data['n_recovered'] - predictions['n_recovered']) ** 2)
    squared_diff += np.sum((data['n_death'] - predictions['n_death']) ** 2)
    return squared_diff
#------------------------------------------------------------------------------------------------

initial_guess = [0.716,50000,0.859,4.6,2.17,5,10,0.9,0.6,0.15,0.135,0.015,0.3]  # Initial values for parameters

#params, _ = curve_fit(objective_function, data['Date'], np.concatenate((data['Infection_case'], data['n_recovered'], data['n_death']),axis=0), p0=initial_guess)
params, _ = curve_fit(objective_function, data['Date'], np.concatenate((data['Infection_case'], data['n_recovered'], data['n_death'])), p0=initial_guess)



#transmission_prob, total_population, reducing_transmission,exposed_period, asymptomatic_period, infectious_period, isolated_period,prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected = params[:13]  # Extract beta and gamma from estimated parameters

#
print("Estimated parameters (transmission_prob, total_population, reducing_transmission,exposed_period, asymptomatic_period, infectious_period, isolated_period,prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):", params)