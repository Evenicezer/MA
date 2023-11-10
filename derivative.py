import numpy as np
from typing import Protocol


def derivative_rhs(t, X, contacts, transmission_prob, total_population, reducing_transmission,
               exposed_period, asymptomatic_period, infectious_period, isolated_period,
               prob_asymptomatic, prob_quarant_inf, test_asy, dev_symp, mortality_isolated, mortality_infected):
    '''
    An ODE-SEAIFRD model with possibly reduced risk from asymptomatic infected individuals.

    @param X Array of values for S, E, A, I, F, R, and D.
    @param t Time point.

    @param transmission_prob Transmission probability.
    @param reducing_transmission factor that reduce the transmission of asymptomatic ~ 0.859
    @param total_population Total number of individuals.
    @param exposed_period Mean exposed period.
    @param asymptomatic_period Mean asymptomatic infectious period.
    @param infectious_period Mean infectious period.
    @param isolated_period Mean quarantined period.
    @param prob_asymptomatic Probability of being asymptomatic once exposed.
    @param prob_isolated Probability of being isolated once infected.
    @param prob_dead Probability of dying
    @dev_symp proportion for asymptomatic to develop symptoms and goes to symptomatic compartment
    @param test_asy proportion that Asymptomatic individuals gets tested and found positive to be isolated
    @param mortality_isolated the rate of COVID-19 mortality for individuals in the quarantined compartment
    @return Returns derivative of S, E, A, I, F, R, and D at t.
    '''
    S, E, A, I, F, R, D = X
    derivS = - contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population
    derivE = contacts * transmission_prob * S * (I + reducing_transmission * A) / total_population - E / exposed_period
    derivA = prob_asymptomatic * E / exposed_period - A / asymptomatic_period
    derivI = (1 - prob_asymptomatic) * E / exposed_period + dev_symp * A / asymptomatic_period - I / infectious_period  # +
    derivF = prob_quarant_inf * I / infectious_period - F / isolated_period + test_asy * A / asymptomatic_period  # prob_isolated_asy*A/asymptomatic_period
    derivR = (1 - prob_quarant_inf - mortality_infected) * I / infectious_period + (1 - mortality_isolated) * F / isolated_period + (1 - dev_symp - test_asy) * A / asymptomatic_period  # (1-prob_isolated_asy)*A / asymptomatic_period
    derivD = (mortality_infected) * I / infectious_period + mortality_isolated * F / isolated_period
    return np.array([derivS, derivE, derivA, derivI, derivF, derivR, derivD])

