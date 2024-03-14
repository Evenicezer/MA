import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import integrate
from iprocessor import add_day_name_column,smooth_sundays_rolling_w7_l,add_date_name_column
from utility_intilization import get_value_by_date
#from numerical_simulation import param_dict_r as param_dict,t_fit

#---------------------------------------------
df = pd.read_csv(r'C:\Users\Evenezer kidane\PycharmProjects\MA\Ma\German_case_wave_1to4.csv')

# -------------------------------------------------------------------------------Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# ------------------------------------------------------------------------------  Modification
# Add the 'days' column
df = add_day_name_column(df)
df = add_date_name_column(df)
# ----------------------------------------------------------------------------- second modification with w7_l
df_observed = smooth_sundays_rolling_w7_l(df)
#df_observed.columns
"""'Date', 'Confirmed', 'Deaths', 'Recovered', 'n_confirmed', 'n_death',
       'n_recovered', 'Infection_case', 'date_name', 'days', 'rolling_mean_r',
       'rolling_mean_c', 'rolling_mean_d'"""
exposed_value_date = get_value_by_date('2020-04-30',df_observed,'n_confirmed')
symptomatic_infected_value_date = get_value_by_date('2020-05-05',df_observed,'n_confirmed')
recovered_values_date = get_value_by_date('2020-05-05',df_observed,'n_recovered')
death_values_date = get_value_by_date('2020-05-05',df_observed,'n_death')
#--------------------------------------------
def derivative_rhs( X,t):
    S, E, A, I, F, R, D = X
    derivS = - contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population
    derivE = contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population - E / exposed_period
    derivA = prob_asymptomatic * E / exposed_period - A / asymptomatic_period
    derivI = (
                         1 - prob_asymptomatic) * E / exposed_period + dev_symp * A / asymptomatic_period - I / infectious_period  # +
    derivF = prob_quarant_inf * I / infectious_period - F / isolated_period + test_asy * A / asymptomatic_period  # prob_isolated_asy*A/asymptomatic_period
    derivR = (1 - prob_quarant_inf - mortality_infected) * I / infectious_period + (
                1 - mortality_isolated) * F / isolated_period + (
                         1 - dev_symp - test_asy) * A / asymptomatic_period  # (1-prob_isolated_asy)*A / asymptomatic_period
    derivD = (mortality_infected) * I / infectious_period + mortality_isolated * F / isolated_period
    return np.array([derivS, derivE, derivA, derivI, derivF, derivR, derivD])



#if __name__ == "__main__":




total_population = 84000000  # Total number of individuals
E0 = exposed_value_date * 1.33 # [1.17,1.33]
A0 = 0
I0 = symptomatic_infected_value_date
F0 = 0
R0 = recovered_values_date
D0 = death_values_date
S0 = total_population - E0 - A0 - I0 - F0 - R0 - D0 # Susceptible individuals


'''contacts = param_dict['contacts']
transmission_prob = param_dict['transmission_prob']
reducing_transmission = param_dict['reducing_transmission']
exposed_period = param_dict['exposed_period']
asymptomatic_period = param_dict['asymptomatic_period']
infectious_period = param_dict['infectious_period']
isolated_period = param_dict['isolated_period']
prob_asymptomatic = param_dict['prob_asymptomatic']
prob_quarant_inf = param_dict['prob_quarant_inf']
test_asy = param_dict['test_asy']
dev_symp = param_dict['dev_symp']
mortality_isolated = param_dict['mortality_isolated']
mortality_infected = param_dict['mortality_infected']'''
# ------------------------------------------------------replacing values
contacts = 4.6 # mean value, from the covimod
transmission_prob = 0.11
#total_population = 8000000
reducing_transmission = 0.859
exposed_period = 5.2  # this is the incubation period
asymptomatic_period = 14  #[3,14]
infectious_period = 7 # [5.6,8.5]
isolated_period = 14 #21
prob_asymptomatic = 0.25
prob_quarant_inf = 0.81 #0.5
test_asy = 0.2
dev_symp = 0.75 # [0.75,0.85]
mortality_isolated = 0.02
mortality_infected = 0.039

#tmax = t_fit  # maximum simulation day
tmax = 269
fslarge = 20
fssmall = 16




#t = tmax
t = np.linspace(0, tmax, (tmax+1)*10)
fig, ax = plt.subplots(figsize=[12, 7])


solution_seaifrd = integrate.odeint(derivative_rhs, [S0, E0, A0, I0, F0, R0, D0],t)
total_infections = total_population - solution_seaifrd[-1,0]
print('Simulation completed successfully.')
#ifr = solution_seaifrd[-1,5] / total_infections
#print('The infection fatality rate is ' +  str(np.round(100*ifr,2)) + ' %')
#print('The case fatality rate is ' +  str(np.round(100*ifr/(1-relrisk_infected),2)) + ' %')



ax.plot(t, solution_seaifrd[:,0], label='Susceptibles', linestyle='-', marker='o', color='blue', markersize=4)
ax.plot(t, solution_seaifrd[:, 1], label='Exposed', linestyle='--', marker='^', color='orange', markersize=3)
ax.plot(t, solution_seaifrd[:, 2], label='Asymptomatic_Infected', linestyle='-', marker='o', color='green', markersize=4)
ax.plot(t, solution_seaifrd[:, 3], label='Symptomatic_Infected', linestyle='--', marker='<', color='red', markersize=4)
ax.plot(t, solution_seaifrd[:, 4], label='Isolated', linestyle='-', marker='>', color='purple', markersize=4)
ax.plot(t, solution_seaifrd[:,5], label='Recovered', linestyle='--', marker='o', color='cyan', markersize=4)
ax.plot(t, solution_seaifrd[:, 6], label='Dead', linestyle='-', marker='s', color='black', markersize=3)


ax.set_xlabel('Days', fontsize=fslarge)
ax.set_ylabel('Number of individuals', fontsize=fslarge)
ax.set_title('Simulation of an ODE-SEAIFRD model', fontsize=fslarge)
ax.legend(fontsize=fssmall, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()
plt.show()