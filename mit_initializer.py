import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from utility_intilization import get_value_by_date
from iprocessor import add_day_name_column, plot_data_dcr, add_date_name_column, smooth_sundays_ssmt, \
    smooth_sundays_ssm, \
    smooth_sundays_rolling_ssm_w3, smooth_sundays_rolling_ssm_w5, smooth_sundays_rolling_w5_r, \
    smooth_sundays_rolling_w5_l, smooth_sundays_rolling_w7_l, \
    smooth_sundays_rolling_w7_r, smooth_sundays_rolling_ssm_w3_smt, plot_data_dcr_multi

# Dataframe------------------------------------------------------------------------------------------------------------

df = pd.read_csv(r'C:\Users\Evenezer kidane\PycharmProjects\MA\Ma\German_case_period_4th.csv')
df_for_init = pd.read_csv(r'C:\Users\Evenezer kidane\PycharmProjects\MA\Ma\German_case_period_jan_aug.csv')
# -------------------------------------------------------------------------------Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df_for_init['Date'] = pd.to_datetime(df_for_init['Date'], format='%Y-%m-%d')

# ------------------------------------------------------------------------------  Modification
# Add the 'days' column
df = add_day_name_column(df)
df = add_date_name_column(df)

df_for_init = add_day_name_column(df_for_init)
df_for_init = add_date_name_column(df_for_init)

# ----------------------------------------------------------------------------- second modification with w7_l
df_observed = smooth_sundays_rolling_w7_l(df)
df_init_observed = smooth_sundays_rolling_w7_l(df_for_init)
#df_observed.columns
"""'Date', 'Confirmed', 'Deaths', 'Recovered', 'n_confirmed', 'n_death',
       'n_recovered', 'Infection_case', 'date_name', 'days', 'rolling_mean_r',
       'rolling_mean_c', 'rolling_mean_d'"""
#some_random_value_deaths = get_value_by_date('2020-05-20',df_observed,'Deaths')
exposed_value_date = get_value_by_date('2020-04-30',df_observed,'n_confirmed')
symptomatic_infected_value_date = get_value_by_date('2020-05-05',df_observed,'n_confirmed')
recovered_values_date = get_value_by_date('2020-05-05',df_observed,'n_recovered')
death_values_date = get_value_by_date('2020-05-05',df_observed,'n_death')
#print(f'{some_random_value_deaths} value for death random')
# -----------------------------------------------------------------------------------------------------

# Sample parameters,
contacts = 5.5 # mean value, from the covimod
transmission_prob = 0.31
total_population = 82000000
reducing_transmission = 0.859
exposed_period = 5.2  # this is the incubation period
asymptomatic_period = 6  #[3, 14]
infectious_period = 7.05 # [5.6, 8.5]
isolated_period = 7 #21
prob_asymptomatic = 0.25
prob_quarant_inf = 0.81 #0.5
test_asy = 0.2 # [0.171, 0.2]
dev_symp = 0.19 # [0.1, 0.85]
mortality_isolated = 0.02
mortality_infected = 0.039 #[0.015, 0.039]

# Sample initial conditions-------------------------------------------------------

#E0  = (1 / (1 - prob_asymptomatic)) * (get_value_by_date('2020-05-25', df_init_observed, 'Confirmed') - get_value_by_date('2020-05-22', df_init_observed, 'Confirmed'))
#A0  = (prob_asymptomatic / (1 - prob_asymptomatic)) * (get_value_by_date('2020-05-22', df_init_observed, 'Confirmed') - get_value_by_date('2020-05-17', df_init_observed, 'Confirmed'))
#I0 = get_value_by_date('2020-05-20', df_init_observed, 'n_confirmed')
#F0 = prob_quarant_inf * I0
#R0 = get_value_by_date('2020-05-20',df_init_observed,'n_recovered')
#D0 = get_value_by_date('2020-05-20',df_init_observed,'n_death')
#S0 = total_population - E0 - A0 - I0 - R0 - D0 - F0

# Print the results
#print(f'exposed_value: {E0}, asymptomatic_value: {A0 }, infection_value: {I0}, recovered:{R0}, death:{D0}, isolated:{F0}, susceptible:{S0}')
#initial_conditions = [S0, E0, A0, I0, F0, R0, D0]
#---------------------------------------------------------------------------------------------------------
# Sample initial conditions-------------------------------------------------------

init_date = pd.to_datetime('2020-05-02')#2020-05-14

# check, if the choosen init_date exists in the dataframe

if init_date < df_init_observed['Date'].min() or init_date > df_init_observed['Date'].max():
    raise ValueError('The date is out of the range of the dataframe')

# ---initialization of the compartments

date_init = init_date.strftime('%Y-%m-%d')

date_ec = (init_date + pd.Timedelta(days=exposed_period + asymptomatic_period)).strftime('%Y-%m-%d')

date_c = (init_date + pd.Timedelta(days=asymptomatic_period)).strftime('%Y-%m-%d')

date_i = (init_date - pd.Timedelta(days=infectious_period)).strftime('%Y-%m-%d')

date_if = (init_date - pd.Timedelta(days=isolated_period + infectious_period)).strftime('%Y-%m-%d')

E0 = (1 / (1 - prob_asymptomatic)) * (
            get_value_by_date(date_ec, df_init_observed, 'Confirmed') - get_value_by_date(date_c, df_init_observed, 'Confirmed'))

A0 = (1 / (1 - prob_asymptomatic)) * (
            get_value_by_date(date_c, df_init_observed, 'Confirmed') - get_value_by_date(date_init, df_init_observed,
                                                                                    'Confirmed'))

I0 = prob_quarant_inf * (
            get_value_by_date(date_i, df_init_observed, 'Confirmed') - get_value_by_date(date_if, df_init_observed, 'Confirmed'))

F0 = get_value_by_date(init_date, df_init_observed, 'Confirmed') - get_value_by_date(date_i, df_init_observed, 'Confirmed')

R0 = get_value_by_date(init_date, df_init_observed, 'Recovered')

D0 = get_value_by_date(init_date, df_init_observed, 'Deaths')

S0 = total_population - E0 - A0 - I0 - R0 - D0 - F0
print(f'exposed_value: {E0}, asymptomatic_value: {A0 }, infection_value: {I0}, recovered:{R0}, death:{D0}, isolated:{F0}, susceptible:{S0}')
initial_conditions = [S0, E0, A0, I0, F0, R0, D0]
#------------------------------------------------------------------------------------------------------
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
    derivR = (1 - prob_quarant_inf - mortality_infected) * I / infectious_period + (1 - mortality_isolated) * F / isolated_period + (
                         1 - dev_symp - test_asy) * A / asymptomatic_period  # (1-prob_isolated_asy)*A / asymptomatic_period
    derivD = (mortality_infected) * I / infectious_period + mortality_isolated * F / isolated_period
    return np.array([derivS, derivE, derivA, derivI, derivF, derivR, derivD])



def seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
                  prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    def derivative(t, initial_conditions):
        return derivative_rhs(t, initial_conditions, contacts, transmission_prob, total_population,
                              reducing_transmission,
                              exposed_period, asymptomatic_period, infectious_period,
                              isolated_period, prob_asymptomatic,
                              prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    solution = solve_ivp(derivative, [0, tmax], initial_conditions, t_eval=t, method='RK45')
    print(solution)
    return solution  # .y.flatten()




def objective(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
              exposed_period, asymptomatic_period, infectious_period, isolated_period,
              prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    temp = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                         exposed_period, asymptomatic_period, infectious_period,
                         isolated_period, prob_asymptomatic,
                         prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)
    # print(f' the name of the columns is: {type(temp)}')
    return temp  # [temp.y[5],temp.y[6]]




def parse_ivp_ode(return_from_objective):
    """takes the function from the ode; and the index 5 and 6; represent the compartment
    recovered and dead
    return: np array of recovered and dead for dialy not commulative; so that we can fit"""
    recovered, dead = return_from_objective.y[5], return_from_objective.y[6]
    time_array = return_from_objective.t
    recovered_difference = []
    dead_difference = []
    for t_index in range(1, len(time_array)):
        # for recovered
        recovered_t = recovered[t_index]
        recovered_t_minus_1 = recovered[t_index - 1]
        recovered_difference.append(recovered_t - recovered_t_minus_1)
        # for dead
        death_t = dead[t_index]
        death_t_minus_1 = dead[t_index - 1]
        dead_difference.append(death_t - death_t_minus_1)
    return [np.array(recovered_difference), np.array(dead_difference)]


return_from_objective = objective(t, contacts, initial_conditions, transmission_prob, total_population,
                                  reducing_transmission,
                                  exposed_period, asymptomatic_period, infectious_period, isolated_period,
                                  prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated,
                                  mortality_infected)

# print(f' here are the recovered and dead: {parse_ivp_ode(return_from_objective)}')
array_recovered, array_dead = parse_ivp_ode(return_from_objective)
# print(f'from recovered array{array_recovered}')
# print(f'from dead array{array_dead}')



def objective_function_(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                        exposed_period, asymptomatic_period, infectious_period, isolated_period,
                        prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated,
                        mortality_infected):
    temp = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                         exposed_period, asymptomatic_period, infectious_period,
                         isolated_period, prob_asymptomatic,
                         prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)

    recovered = temp.y[5]
    dead = temp.y[6]
    daily_recovered = recovered(t) - recovered(t - 1)
    daily_dead = dead(t) - dead(t - 1)
    return [daily_recovered, daily_dead]


def objective_function(t,  contacts, mortality_infected):
    #initial_conditions = [S0, E0, A0, I0, F0, R0, D0]
    #contacts = 4.6  # mean value, from the covimod
    # contacts = 4.6 # mean value, from the covimod
    transmission_prob = 0.31 #[0.11, 0.365]
    total_population = 82000000
    reducing_transmission = 0.859
    exposed_period = 5.2  # this is the incubation period
    asymptomatic_period = 6  # [3, 14] mean 8.5
    infectious_period = 7.05  # [5.6, 8.5]mean
    isolated_period = 7  # 21
    prob_asymptomatic = 0.25
    prob_quarant_inf = 0.81  # 0.5
    test_asy = 0.2  # [0.171, 0.2]
    dev_symp = 0.19  # [0.1, 0.85]
    mortality_isolated = 0.02
    #mortality_infected = 0.039  # [0.015, 0.039]
    temp = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                         exposed_period, asymptomatic_period, infectious_period,
                         isolated_period, prob_asymptomatic,
                         prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)


    recovered = temp.y[5]
    dead = temp.y[6]
    daily_recovered_ = []
    daily_death_ = []
    for t_index in range(1, len(t)):
        # for recovered
        recovered_t = recovered[t_index]
        recovered_t_minus_1 = recovered[t_index - 1]
        daily_recovered_.append(recovered_t - recovered_t_minus_1)
        # for dead
        death_t = dead[t_index]
        death_t_minus_1 = dead[t_index - 1]
        daily_death_.append(death_t - death_t_minus_1)
    return daily_recovered_
def objective_function_dead(t,  contacts, mortality_infected):
    #initial_conditions = [S0, E0, A0, I0, F0, R0, D0]
    # contacts = 4.6 # mean value, from the covimod
    transmission_prob = 0.31  # [0.11, 0.365]
    total_population = 82000000
    reducing_transmission = 0.859
    exposed_period = 5.2  # this is the incubation period
    asymptomatic_period = 6  # [3, 14] mean 8.5
    infectious_period = 7.05  # [5.6, 8.5]mean
    isolated_period = 7  # 21
    prob_asymptomatic = 0.25
    prob_quarant_inf = 0.81  # 0.5
    test_asy = 0.2  # [0.171, 0.2]
    dev_symp = 0.19  # [0.1, 0.85]
    mortality_isolated = 0.02
    #mortality_infected = 0.039  # [0.015, 0.039]
    temp = seaifrd_model(t, contacts, initial_conditions, transmission_prob, total_population, reducing_transmission,
                         exposed_period, asymptomatic_period, infectious_period,
                         isolated_period, prob_asymptomatic,
                         prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected)


    recovered = temp.y[5]
    dead = temp.y[6]
    daily_recovered_ = []
    daily_death_ = []
    for t_index in range(1, len(t)):
        # for recovered
        recovered_t = recovered[t_index]
        recovered_t_minus_1 = recovered[t_index - 1]
        daily_recovered_.append(recovered_t - recovered_t_minus_1)
        # for dead
        death_t = dead[t_index]
        death_t_minus_1 = dead[t_index - 1]
        daily_death_.append(death_t - death_t_minus_1)
    return daily_death_
#daily_recovered_,daily_death_= objective_function()

#result_vector=(np.concatenate([x,y]))
#print(result_vector)

# curve_fit to estimate parameters
#print(len(t),len(array_dead))
t_end = df_observed['days'].iloc[-1]

# Create a sequence from 0 to t_end
t_fit = np.arange(0, tmax, 1)
#t_fitt = np.arange(0,t,1)
# Concatenate the sequence with itself to repeat from 0 to t_end
#t_fit_ = np.concatenate([t_fit, t_fit])
#t_fit_ = t_fit_[2:]

recov_dead = np.concatenate([array_recovered,array_dead])

#print(f'length of the time t_fit_{len(t_fit_)}, tmax{(tmax)}, time t {len(t)},time t_fit_{len(t_fit)}, recov_dead{len(recov_dead)}')
params_r, _ = curve_fit(objective_function, t_fit, array_recovered)  # since curve_fit is expecting target values ydata as a single array; and x=time
params_d, _ = curve_fit(objective_function_dead, t_fit, array_dead)
#params_rd, _ = curve_fit(objective_function, t_fit_, recov_dead)
#print(f'params:{params},\n,{type(params)}')
print(f'from_recovered{params_r}: from_dead{params_d}')
# assigning back
# List of names corresponding to each value in params
##param_names = ['contacts','mortality_infected']

##param_dict_r = {}

##for name, value in zip(param_names, params_r):
    #param_dict_r[name] = value

##print(param_dict_r)
#formatted_string = '\n'.join(f'{key} = {value}' for key, value in param_dict.items())
#print(formatted_string)

# for dead
##param_dict_d = {}
##for name, value in zip(param_names, params_d):
 #   param_dict_d[name] = value

#print(param_dict_d)


#print(param_dict_rd)
#print(f'length of the time t_fit_{len(t_fit)}, tmax{(tmax)}, time t {len(t)},time t_fit_{len(t_fit)}, recov_dead{len(recov_dead)}')