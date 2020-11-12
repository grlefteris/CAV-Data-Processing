import pandas as pd
import numpy as np
import GA_functions as gaf
from operator import itemgetter
import matplotlib.pyplot as plt
from os import listdir
from os import getcwd
from os.path import isfile, join
import collections
from multiprocessing import Pool
import numpy as np
import xlrd
import pandas as pd
import scipy as sp
import csv
import filters as filt
from scipy.interpolate import CubicSpline


def kalman_gen_alg(size,iterations,elit,mutation_rate,tournament,speed_upper_bound,speed_lower_bound,ivs_upper_bound,ivs_lower_bound,w1,w2,data_path,N):
    '''Calibrate Kalman with GA'''

    '''read data'''
    kalman_input = read_data(data_path, N)

    '''chose details'''
    # size = 150
    # iterations = 40
    # elit = 50
    tol = 0
    np.random.seed(100)
    # mutation_rate = 0.05
    # tournament = 50
    crossover = gaf.crossover
    Leaders = []
    #
    # '''Kalman specific'''
    # speed_upper_bound = 3
    # speed_lower_bound = 0.01
    first_number = N
    # ivs_upper_bound = 1000
    # ivs_lower_bound = 50
    second_number = N-1
    # w1 = 0.5
    # w2 = 0.5

    params_upper = np.array([speed_upper_bound] * first_number + [ivs_upper_bound] * second_number)
    params_lower = np.array([speed_lower_bound] * first_number + [ivs_lower_bound] * second_number)
    param_number = first_number + second_number

    # if __name__ == '__main__':
    p = Pool()

    Init = gaf.initial_population(size, params_upper, params_lower, param_number)
    # Fitnetss = [*map(functools.partial(cfo.fcn2minGA, instance = instance,model=model,tp=tp,lsp=lsp,iv_dist=distance,start_dist=start_dist,start_speed=start_speed), Init)]

    Init = gaf.fitness(Init, kalman_input, w1, w2, p)
    Init = sorted(Init, key=itemgetter('fit'))

    Population = []
    j = 0
    while j < iterations and Init[0]['fit'] > tol:
        j += 1
        Population = Init[:elit]
        for i in range(len(Population)):
            del (Population[i]['fit'])
        for i in range(size - elit):
            child = crossover(Init, params_upper, params_lower, tournament, mutation_rate)
            Population.append(child)

        Population = gaf.fitness(Population, kalman_input, w1, w2, p)
        Population = sorted(Population, key=itemgetter('fit'))

        ###print best
        print(Population[0])
        Init = Population[:]

        Population = []

    Leaders.append(Init[0])
        # Output_df = Output_df.append(Leaders)
        # Output_df.to_csv(str(model_name) + '_Result_lap_RT_Noise.csv')

    return Leaders


def read_data(path,num_of_vehicles):

    x = [None] * num_of_vehicles
    y = [None] * num_of_vehicles
    z = [None] * num_of_vehicles
    u=[None]*num_of_vehicles
    ud = [None] * num_of_vehicles
    ivs = [None] * num_of_vehicles
    sxy = [None] * num_of_vehicles
    sz = [None] * num_of_vehicles



    xls = xlrd.open_workbook(path, on_demand=True)
    stats = xls.sheet_names()
    df = pd.read_excel(path, sheet_name=stats[0])

    for j in range(num_of_vehicles):
        x[j] = np.array(df[f'x{j+1}'])
        y[j] = np.array(df[f'y{j+1}'])
        z[j] = np.array(df[f'z{j+1}'])
        u[j] = np.array(df[f'u{j+1}'])
        ud[j] = np.array(df[f'ud{j + 1}'])
        sxy[j] = np.array(df[f'sxy{j+1}'])
        sz[j] = np.array(df[f'sz{j+1}'])

    for j in range(num_of_vehicles-1):
        ivs[j] = np.array(df[f'ivs{j + 1}'])


    kalman_input=[x,y,z,u,ud,ivs,sxy,sz]

    return kalman_input



def Kalman_error(params,kalman_input,w1,w2):

    x = kalman_input[0]
    y = kalman_input[1]
    z = kalman_input[2]
    u = kalman_input[3]
    ud = kalman_input[4]
    ivs = kalman_input[5]
    sxy = kalman_input[6]
    sz = kalman_input[7]

    N=len(x)
    k=x[0].size
    dt=0.1
    t=np.arange(0,k*dt,dt)

    distP = [None] * N
    cdux = [None] * N
    cduy = [None] * N
    dUkal = [None] * N
    xuk = [None] * N
    yuk = [None] * N
    rmseUkal_Udop = [None] * N
    ivsuk = [None]*(N-1)
    err = [None] * (N - 1)
    # for i in range(N):
        # distP[i]=np.zeros(k)
        # dUkal[i] = np.zeros(k-1)
        # for l in np.arange(1,k):
        #     distP[i][l]=distP[i][l-1]+np.sqrt((x[i][l]-x[i][l-1])**2+(y[i][l]-y[i][l-1])**2+(z[i][l]-z[i][l-1])** 2)
        #     if distP[i][l] <= distP[i][l-1]:
        #         distP[i][l]=distP[i][l]+0.0000001
        #
        # cdux[i] = CubicSpline(distP[i], x[i])
        # cduy[i] = CubicSpline(distP[i], y[i])



    Ukal,UkalS= filt.kalman(x, y, z, u, ivs, sxy, sz, N, dt,k, params['Params'][0:N], params['Params'][N:2*N-1])
    for i in range(N):
        rmseUkal_Udop[i] = np.sqrt(np.sum((np.abs(np.diff(UkalS[i][2:-2])/dt) - np.abs(np.diff(ud[i][2:-3])/dt)) ** 2) / (k -5))

        # for l in np.arange(1, k- 1):
        #     dUkal[i][l] = sp.integrate.trapz(UkalS[i][0:l + 1], t[0:l + 1])
        #
        # xuk[i] = cdux[i](dUkal[i])
        # yuk[i] = cduy[i](dUkal[i])

    for i in range(N - 1):
        ivsuk[i] = np.zeros(k - 1)
        ivsuk[i][0] = ivs[i][0]
        for l in np.arange(1, k - 1):
            ivsuk[i][l] = ivsuk[i][0] + sp.integrate.trapz(UkalS[i][0:l + 1],t[0:l + 1]) - sp.integrate.trapz(UkalS[i + 1][0:l + 1], t[0:l + 1])

        # ivsuk[i] = np.sqrt((xuk[i] - xuk[i + 1]) ** 2 + (yuk[i] - yuk[i + 1]) ** 2)
        err[i] = np.sqrt(np.sum((ivs[i][2:-3] - ivsuk[i][2:-2]) ** 2) / (k - 5)) / np.mean(ivs[i][2:-3])


    # print(params)

    error= w1*np.mean(err) + w2*np.mean(rmseUkal_Udop)

    return error


def evalbut(uu,ud,Tnew,IVS,k,N,dt,w1,w2,f):
    u = [None] * N
    e2 = [None] * N
    e1 = [None] * (N-1)
    ivsu = [None] * (N-1)
    for i in range(N):
        u[i]=filt.butrw(uu[i], 1, f)
        e2[i] = np.sqrt(np.sum((np.abs(np.diff(u[i]) / dt) - np.abs(np.diff(ud[i]) / dt)) ** 2) / k)

    for i in range(N - 1):
        ivsu[i] = np.zeros(k - 1)
        ivsu[i][0]=IVS[i][0]
        for l in np.arange(1, k - 1):
            ivsu[i][l] = ivsu[i][0] + sp.integrate.trapz(u[i][0:l + 1], Tnew[0:l + 1]) - sp.integrate.trapz(u[i + 1][0:l + 1], Tnew[0:l + 1])
        e1[i] = np.sqrt(np.sum((IVS[i][:-1] - ivsu[i]) ** 2) / (k - 1)) / np.mean(IVS[i][:-1])
    return w1*np.mean(e1)+w2*np.mean(e2)



def optibut(uu,ud,Tnew,IVS,k,N,dt,w1,w2,minf, maxf, step):
    e = np.zeros((np.arange(minf, maxf, step).size))
    ii = np.zeros((np.arange(minf, maxf, step).size))
    j = 0
    for i in np.arange(minf, maxf, step):
        e[j] = evalbut(uu,ud,Tnew,IVS,k,N,dt,w1,w2,i)
        ii[j] = i
        j = j + 1

    minerr = np.min(e)
    eind = np.where(e == minerr)
    optf = ii[eind]

    return minerr, optf[0]




def evalmav(uu,ud,Tnew,IVS,k,N,dt,w1,w2,Ns,Nf):
    u = [None] * N
    e2 = [None] * N
    e1 = [None] * (N-1)
    ivsu = [None] * (N-1)
    for i in range(N):
        u[i]=filt.medmav(uu[i],Ns,Nf)
        e2[i] = np.sqrt(np.sum((np.abs(np.diff(u[i]) / dt) - np.abs(np.diff(ud[i]) / dt)) ** 2) / k)
    for i in range(N - 1):
        ivsu[i] = np.zeros(k - 1)
        ivsu[i][0] = IVS[i][0]
        for l in np.arange(1,k-1):
            ivsu[i][l]=ivsu[i][0] + sp.integrate.trapz(u[i][0:l + 1],Tnew[0:l + 1]) - sp.integrate.trapz(u[i + 1][0:l + 1], Tnew[0:l + 1])
        e1[i] = np.sqrt(np.sum((IVS[i][:-1] - ivsu[i]) ** 2) / (k - 1)) / np.mean(IVS[i][:-1])
    return w1*np.mean(e1)+w2*np.mean(e2)


def optimav(uu,ud,Tnew,IVS,k,N,dt,w1,w2,minNs, maxNs,minNf, maxNf):
    e = np.zeros((np.arange(minNs, maxNs, 1).size) * (np.arange(minNf, maxNf, 2).size))
    ii = np.zeros((np.arange(minNs, maxNs, 1).size) * (np.arange(minNf, maxNf, 2).size))
    ll = np.zeros((np.arange(minNs, maxNs, 1).size) * (np.arange(minNf, maxNf, 2).size))
    j = 0
    for i in np.arange(minNs, maxNs, 1):
        for l in np.arange(minNf, maxNf, 2):
            e[j] = evalmav(uu,ud,Tnew,IVS,k,N,dt,w1,w2,i,l)
            ii[j] = i
            ll[j] = l
            j = j + 1

    minerr=np.min(e)
    eind=np.where(e==minerr)
    optNs=ii[eind]
    optNf = ll[eind]

    return minerr, int(optNs[0]), int(optNf[0])


def doppler_correction(ud,dist,T,N):
    udn = [None]*N
    distud = [None] * N
    for i in range(N):
        distud[i] = sp.integrate.trapz(ud[i], T)
        udn[i] = ud[i] + dfun(T, T[-1], dist[i][-1], distud[i])

    return udn


def dfun(t,T,dist,distl):
    b=(dist-distl)/(((T**2)/2)-((T**2)/3))
    a=-b/T
    return a*(t**2)+b*t