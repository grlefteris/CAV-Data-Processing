import numpy as np
from multiprocessing import Pool
from functools import partial
import data_and_kalman_specific_funcs as dks

'''Functions used for Genetic Algorithm Calibration'''

def initial_population(size,params_upper,params_lower,param_number):

    Population = []
    for i in range(size):
        indiv = {}
        indiv['Params'] = np.round(np.random.random(param_number) * (params_upper - params_lower) + params_lower)
        Population.append(indiv)

    return Population

def fitness(Population,kalman_input,w1,w2,p):

    for i, f in enumerate(p.map(partial(dks.Kalman_error,kalman_input=kalman_input,w1=w1,w2=w2), Population)):
        Population[i]['fit'] = f

    return Population


def select_parent_determ_tournament(size,tournament):
    temp = np.random.choice(np.arange(size),tournament)
    return np.min(temp)

def select_parent_random_tournament(size):
    temp = np.random.choice(np.arange(size),5)
    temp.sort()
    while np.random.rand()>0.6 and len(temp)>1:
        temp = np.delete(temp,0)
    return temp[0]

def mutation(boarders):
    value = np.random.random()*(boarders[0] - boarders[1]) + boarders[1]
    return value

def crossover(Population,params_upper,params_lower,tournament,mutation_rate=0.05):

    size = len(Population)
    p=[]
    p.append(Population[select_parent_determ_tournament(size,tournament)])
    p.append(Population[select_parent_determ_tournament(size,tournament)])

    child_pars = []

    for i,param in enumerate(p[0]['Params']):
        c = np.random.randint(0,2)
        child_pars.append(p[c]['Params'][i])
        if np.random.rand()<mutation_rate:
            child_pars[-1] = mutation([params_upper[i],params_lower[i]])

    return {'Params':np.array(child_pars)}


def crossover_convex(Population,params_upper,params_lower,tournament,mutation_rate=0.3):

    size = len(Population)
    p=[]
    p.append(Population[select_parent_determ_tournament(size,tournament)])
    p.append(Population[select_parent_determ_tournament(size,tournament)])

    child_pars = []

    for i,param in enumerate(p[0]['Params']):
        a = np.random.rand()
        child_pars.append(a*p[0]['Params'][i]+(1-a)*p[1]['Params'][i])
        if np.random.rand() < mutation_rate:
            child_pars[-1] = mutation([params_upper[i], params_lower[i]])

    return {'Params':child_pars}

def sum_fitness(Init):

    fit_sum = 0
    for i in Init:
        fit_sum += 1/i['fit']
    return fit_sum