import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from iprocessor import add_day_name_column, plot_data_dcr, add_date_name_column, smooth_sundays_ssmt, \
    smooth_sundays_ssm, \
    smooth_sundays_rolling_ssm_w3, smooth_sundays_rolling_ssm_w5, smooth_sundays_rolling_w5_r, \
    smooth_sundays_rolling_w5_l, smooth_sundays_rolling_w7_l, \
    smooth_sundays_rolling_w7_r, smooth_sundays_rolling_ssm_w3_smt, plot_data_dcr_multi
import lmfit
from lmfit import Parameters, minimize, report_fit

# Sample parameters
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

# Sample initial conditions
S0 = total_population - 20
E0 = 10
A0 = 0
I0 = 10
F0 = 0
R0 = 0
D0 = 0
initial_conditions = [S0, E0, A0, I0, F0, R0, D0]

# Dataframe
df = pd.read_csv(r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\German_case_.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Add the 'days' column
df = add_day_name_column(df)
df = add_date_name_column(df)

# Second modification with w7_l
df_observed = smooth_sundays_rolling_w7_l(df)

# Taking 'days' time column from dataframe
t = np.array(df_observed['days'])
tmax = len(t)

# Define the lmfit model
def lmfit_objective(params, t, array_recovered):
    contacts = params['contacts']
    initial_conditions = params['initial_conditions']
    transmission_prob = params['transmission_prob']
    total_population = params['total_population']
    reducing_transmission = params['reducing_transmission']
    exposed_period = params['exposed_period']
    asymptomatic_period = params['asymptomatic_period']
    infectious_period = params['infectious_period']
    isolated_period = params['isolated_period']
    prob_asymptomatic = params['prob_asymptomatic']
    prob_quarant_inf = params['prob_quarant_inf']
    test_asy = params['test_asy']
    dev_symp = params['dev_symp']
    mortality_isolated = params['mortality_isolated']
    mortality_infected = params['mortality_infected']

    temp = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                         exposed_period, asymptomatic_period, infectious_period,
                         isolated_period, prob_asymptomatic,
                         prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    recovered = temp.y[5]
    dead = temp.y[6]
    daily_recovered_ = []
    for t_index in range(1, len(t)):
        # for recovered
        recovered_t = recovered[t_index]
        recovered_t_minus_1 = recovered[t_index - 1]
        daily_recovered_.append(recovered_t - recovered_t_minus_1)
    return np.concatenate(daily_recovered_)

# Set up lmfit parameters
params = Parameters()
params.add('contacts', value=1.0)
params.add('initial_conditions', value=[S0, E0, A0, I0, F0, R0, D0], vary=False)
params.add('transmission_prob', value=0.1)
params.add('total_population', value=1000000)
params.add('reducing_transmission', value=0.859)
params.add('exposed_period', value=5)
params.add('asymptomatic_period', value=14)
params.add('infectious_period', value=14)
params.add('isolated_period', value=21)
params.add('prob_asymptomatic', value=0.3)
params.add('prob_quarant_inf', value=0.5)
params.add('test_asy', value=0.2)
params.add('dev_symp', value=0.1)
params.add('mortality_isolated', value=0.02)
params.add('mortality_infected', value=0.01)

# Perform lmfit minimization
result = minimize(lmfit_objective, params, args=(t, array_recovered))

# Print lmfit result
report_fit(result)

# Extract lmfit parameters
lmfit_params = [result.params[param].value for param in params]

# Print lmfit parameters
print("Estimated Parameters:", lmfit_params)
