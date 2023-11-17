import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from iprocessor import add_day_name_column, plot_data_dcr, add_date_name_column, smooth_sundays_ssmt,smooth_sundays_ssm,\
    smooth_sundays_rolling_ssm_w3,smooth_sundays_rolling_ssm_w5,smooth_sundays_rolling_w5_r,smooth_sundays_rolling_w5_l,smooth_sundays_rolling_w7_l,\
        smooth_sundays_rolling_w7_r,smooth_sundays_rolling_ssm_w3_smt,plot_data_dcr_multi

# Sample parameters,
contacts = 1.0
transmission_prob = 0.1
total_population = 1000000
reducing_transmission = 0.859
exposed_period = 5
asymptomatic_period = 14
infectious_period = 14
isolated_period = 21
prob_asymptomatic = 0.3
prob_quarant_inf = 0.5
test_asy = 0.2
dev_symp = 0.1
mortality_isolated = 0.02
mortality_infected = 0.01

# Sample initial conditions-------------------------------------------------------
S0 = total_population - 1
E0 = 1
A0 = 0
I0 = 0
F0 = 0
R0 = 0
D0 = 0
initial_conditions = [S0, E0, A0, I0, F0, R0, D0]


# Dataframe-------------------------------------------------
df = pd.read_csv(r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\German_case_.csv')

# -------------------------------------------------------------------------------Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


#------------------------------------------------------------------------------  Modification
# Add the 'days' column
df= add_day_name_column(df)
df = add_date_name_column(df)
#----------------------------------------------------------------------------- second modification with w7_l
df_observed = smooth_sundays_rolling_w7_l(df)
#-----------------------------------------------------------------------------------------------------
# Taking 'days' time column from dataframe
t = np.array(df_observed['days'])
tmax = len(t)
def derivative_rhs(t, X, contacts, transmission_prob, total_population, reducing_transmission,
                   exposed_period, asymptomatic_period, infectious_period, isolated_period,
                   prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    S, E, A, I, F, R, D = X
    derivS = - contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population
    derivE = contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population - E / exposed_period
    derivA = prob_asymptomatic * E / exposed_period - A / asymptomatic_period
    derivI = (1 - prob_asymptomatic) * E / exposed_period + dev_symp * A / asymptomatic_period - I / infectious_period  # +
    derivF = prob_quarant_inf * I / infectious_period - F / isolated_period + test_asy * A / asymptomatic_period  # prob_isolated_asy*A/asymptomatic_period
    derivR = (1 - prob_quarant_inf - mortality_infected) * I / infectious_period + (1 - mortality_isolated) * F / isolated_period + (1 - dev_symp - test_asy) * A / asymptomatic_period  # (1-prob_isolated_asy)*A / asymptomatic_period
    derivD = (mortality_infected) * I / infectious_period + mortality_isolated * F / isolated_period
    return np.array([derivS, derivE, derivA, derivI, derivF, derivR, derivD])

#print(derivative_rhs(t, initial_conditions, contacts, transmission_prob, total_population, reducing_transmission,
 #                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
  #                 prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected))


def seaifrd_model(t, contacts, initial_conditions,transmission_prob, total_population, reducing_transmission,
                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
                  prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    def derivative(t, initial_conditions):
        return derivative_rhs(t, initial_conditions, contacts, transmission_prob, total_population, reducing_transmission,
                              exposed_period, asymptomatic_period, infectious_period,
                              isolated_period, prob_asymptomatic,
                              prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    solution = solve_ivp(derivative, [0, tmax], initial_conditions, t_eval=t, method='RK45')
    return solution#.y.flatten()
print(seaifrd_model(t, contacts,initial_conditions, transmission_prob, total_population, reducing_transmission,
                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
                  prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected))
def fit_seaifrd_model(t, initial_conditions,confirmed, recovered, death):
    def objective(t, contacts, initial_conditions,transmission_prob, total_population, reducing_transmission,
                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
                  prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
        return seaifrd_model(t, contacts, initial_conditions,transmission_prob, total_population, reducing_transmission,
                             exposed_period, asymptomatic_period, infectious_period,
                             isolated_period, prob_asymptomatic,
                             prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    # Initial parameter guesses
    initial_guess = [1.0,initial_conditions, 0.1, 1000000, 0.859, 5, 14, 14, 21, 0.3, 0.5, 0.2, 0.1, 0.02, 0.01]

    # curve_fit to estimate parameters
    params,_ = curve_fit(objective,t, np.concatenate([confirmed, recovered, death]), p0=initial_guess)# since curve_fit is expecting target values ydata as a single array; and x=time

    return params

# Estimate parameters
estimated_params = fit_seaifrd_model(t, initial_conditions,df_observed['n_confirmed'], df_observed['n_recovered'], df_observed['n_death'])

# Print the estimated parameters
print("Estimated Parameters:", estimated_params)

# Integrate the SEAIRFD model with estimated parameters
solution_seaifrd = solve_ivp(
    derivative_rhs,
    [0, tmax],
    initial_conditions,
    t_eval=t,
    args=estimated_params,
    method='RK45'
)


