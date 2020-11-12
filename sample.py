import data_read as dr
import save_car_data_to_xl as svc
import filter_optimize as fop
import filters as filt
import metrics as mt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# path=r'D:\Apps\CF_Data_Processing\Experimental Data\JRC_28Feb\Vicolungo_JRC'

path=r'/Users/lefteris/Desktop/untitled1/jrc_athens/Experimental Data/JRC_28Feb/JRC_Vicolungo'

start=0  #9200   17190  3100  8500  10500  12600  16000
stop=30000   #10200  18190  3200  9500  11500  13600  18000
dt=0.1
k=stop-start
output_file='kinputs3.xls'


### kalman optimization parameters ###
size = 30
iterations = 40
elit = 10
np.random.seed(100)
mutation_rate = 0.2
tournament = 15
speed_upper_bound = 1 # 2
speed_lower_bound = 0.0   #0.01
first_number = 5
ivs_upper_bound = 500# 1200
ivs_lower_bound = 0#  20
second_number = 4
wk1=0.5
wk2=0.5
### kalman optimization parameters ###


### moving average optimization parameters ###
minNs=141#3
maxNs=145
minNf=5#5     #should be odd
maxNf=7#35    #should be odd
wma1=0.5
wma2=0.5
### moving average optimization parameters ###



### butterworth optimization parameters ###
minf=0.001#0.01
maxf=0.007#0.2
step=0.001#0.01
wbu1=0.5
wbu2=0.5
### butterworth optimization parameters ###



########################################## Data Processing ################################################
x,y,z,ud,u,ivs,dist,T,sxy,sz,su,N=dr.car_data(path,start,stop,dt,10)
print('Data are read, processed and synchronized')
svc.save_data(x,y,z,u,ud,sxy,sz,ivs,N,stop-start,output_file)
print(r'Unfiltered processed data are saved into', output_file)
########################################## Data Processing ################################################

#
#
# ########################################## Optimization ################################################
# if __name__ == '__main__':
#     kalman_parameters=fop.kalman_gen_alg(size,iterations,elit,mutation_rate,tournament,speed_upper_bound,\
#                                           speed_lower_bound,ivs_upper_bound,ivs_lower_bound,wk1,wk2,\
#                                          r'/Users/lefteris/Desktop/untitled1/jrc_athens/kinputs3.xls',N)
#     gamma_u=kalman_parameters[0]['Params'][0:N]
#     gamma_ivs = kalman_parameters[0]['Params'][N:2*N-1]
#     kalman_expected_error=kalman_parameters[0]['fit']
# print('Kalman smoother is now optimized')
#
# moving_average_expected_error,Ns,Nf=fop.optimav(u,ud,T,ivs,stop-start,N,dt,wma1,wma2,minNs, maxNs, minNf, maxNf)
# print('Moving average filter is now optimized')
#
# butterworth_expected_error,f=fop.optibut(u,ud,T,ivs,stop-start,N,dt,wbu1,wbu2,minf,maxf,step)
# print('Butterworth filter is now optimized')
#
# gamma_u=np.array([0.01162527,0.864095,0.03257831,0.60301148,0.07627791])
# gamma_ivs=np.array([262.70896123,72.69085042,64.90056348,8.08489916])
# ########################################## Optimization ################################################
#
#
#
# ######################################### Evaluate Speeds ##############################################
# uks=[None]*N
# uk, ukst =filt.kalman(x, y, z, u, ivs, sxy, sz, N, dt, k, gamma_u, gamma_ivs)
# for i in range(N):
#     uks[i]=np.append(ukst[i],2*ukst[i][-2]-ukst[i][-1])
# print('Kalman speed has been computed')
#
#
# uma=[None]*N
# for i in range(N):
#     uma[i]=filt.medmav(u[i],Ns,Nf)
# print('Moving average speed has been computed')
#
#
# ubut=[None]*N
# for i in range(N):
#     ubut[i]=filt.butrw(u[i],1,f)
# print('Butterworth speed has been computed')
#
#
# udn=fop.doppler_correction(ud,dist,T,N)
# print('Doppler speed is now corrected')
# ######################################### Evaluate Speeds ##############################################
#
#
#
# ########################################## Compute Errors ###############################################
# err_min=[None]*5
# err_max=[None]*5
# err_mean=[None]*5
# err_rmse=[None]*5
#
# err_dop=[None]*3
# err_total=[None]*3
#
# err_rmse_uks, err_min_uks, err_max_uks, err_mean_uks = mt.ivs_error(uks,ivs,T,N,k)
# err_rmse_uks = np.mean(err_rmse_uks)
# err_min_uks = np.mean(err_min_uks)
# err_max_uks = np.mean(err_max_uks)
# err_mean_uks = np.mean(err_mean_uks)
#
#
# err_rmse_uma, err_min_uma, err_max_uma, err_mean_uma = mt.ivs_error(uma,ivs,T,N,k)
# err_rmse_uma = np.mean(err_rmse_uma)
# err_min_uma = np.mean(err_min_uma)
# err_max_uma = np.mean(err_max_uma)
# err_mean_uma = np.mean(err_mean_uma)
#
# err_rmse_ubut, err_min_ubut, err_max_ubut, err_mean_ubut = mt.ivs_error(ubut,ivs,T,N,k)
# err_rmse_ubut = np.mean(err_rmse_ubut)
# err_min_ubut = np.mean(err_min_ubut)
# err_max_ubut = np.mean(err_max_ubut)
# err_mean_ubut = np.mean(err_mean_ubut)
#
# err_rmse_ud, err_min_ud, err_max_ud, err_mean_ud = mt.ivs_error(ud,ivs,T,N,k)
# err_rmse_ud = np.mean(err_rmse_ud)
# err_min_ud = np.mean(err_min_ud)
# err_max_ud = np.mean(err_max_ud)
# err_mean_ud = np.mean(err_mean_ud)
#
# err_rmse_udn, err_min_udn, err_max_udn, err_mean_udn = mt.ivs_error(udn,ivs,T,N,k)
# err_rmse_udn = np.mean(err_rmse_udn)
# err_min_udn = np.mean(err_min_udn)
# err_max_udn = np.mean(err_max_udn)
# err_mean_udn = np.mean(err_mean_udn)
#
#
# err_dop_uks=np.mean(mt.doppler_compare(uks,ud,dt,N,k))
#
# err_dop_uma=np.mean(mt.doppler_compare(uma,ud,dt,N,k))
#
# err_dop_ubut=np.mean(mt.doppler_compare(ubut,ud,dt,N,k))
#
# err_uks=0.5*err_rmse_uks + 0.5*err_dop_uks
# err_uma=0.5*err_rmse_uma + 0.5*err_dop_uma
# err_ubut=0.5*err_rmse_ubut + 0.5*err_dop_ubut
#
#
# err_min[0]=err_min_ud
# err_min[1]=err_min_udn
# err_min[2]=err_min_uma
# err_min[3]=err_min_ubut
# err_min[4]=err_min_uks
#
# err_max[0]=err_max_ud
# err_max[1]=err_max_udn
# err_max[2]=err_max_uma
# err_max[3]=err_max_ubut
# err_max[4]=err_max_uks
#
# err_mean[0]=err_mean_ud
# err_mean[1]=err_mean_udn
# err_mean[2]=err_mean_uma
# err_mean[3]=err_mean_ubut
# err_mean[4]=err_mean_uks
#
# err_rmse[0]=err_rmse_ud
# err_rmse[1]=err_rmse_udn
# err_rmse[2]=err_rmse_uma
# err_rmse[3]=err_rmse_ubut
# err_rmse[4]=err_rmse_uks
#
# err_dop[0]=err_dop_uma
# err_dop[1]=err_dop_ubut
# err_dop[2]=err_dop_uks
#
# err_total[0]=err_uma
# err_total[1]=err_ubut
# err_total[2]=err_uks
#
# ################### Internal Consistency#####################
# icuks, duks = mt.internal_consistency(uks,dist,T,N,k)
#
# icuma, duma = mt.internal_consistency(uma,dist,T,N,k)
#
# icubut, dubut = mt.internal_consistency(ubut,dist,T,N,k)
#
# icud, dud = mt.internal_consistency(ud,dist,T,N,k)
#
# icudn, dudn = mt.internal_consistency(udn,dist,T,N,k)
# ################### Internal Consistency#####################
#
#
# ########################################## Compute Errors ###############################################
#
#
# # for i in range(N):
# #     plt.figure(i)
# #     plt.plot(T,u[i],'b',label='measured speed')
# #     plt.plot(T,uks[i],'r',label='Kalman')
# #     plt.plot(T,uma[i],'g',label='Moving Average')
# #     plt.plot(T,ubut[i],'k',label='Butterworth')
# #     plt.plot(T,udn[i],'y',label='Corrected Doppler')
# #     plt.legend()






