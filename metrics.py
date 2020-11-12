import numpy as np
import scipy as sp

def ivs_error(u,IVS,T,N,k):
    IVSnew=[None]*(N-1)
    ivs_rmse = [None] * (N - 1)
    ivs_err_min = [None] * (N - 1)
    ivs_err_max = [None] * (N - 1)
    ivs_err_mean = [None] * (N - 1)
    for i in range(N-1):
        IVSnew[i] = np.zeros(k - 1)
        IVSnew[i][0] = IVS[i][0]
        for l in np.arange(1, k - 1):
            IVSnew[i][l] = IVSnew[i][0] + sp.integrate.trapz(u[i][0:l + 1], T[0:l + 1]) - sp.integrate.trapz(
                u[i + 1][0:l + 1], T[0:l + 1])
        ivs_rmse[i] = np.sqrt(np.sum((IVS[i][2:-3] - IVSnew[i][2:-2]) ** 2) / (k - 5)) / np.mean(IVS[i][2:-3])
        ivs_err_min[i]=np.min(np.abs((IVS[i][2:-3] - IVSnew[i][2:-2])))
        ivs_err_max[i] = np.max(np.abs((IVS[i][2:-3] - IVSnew[i][2:-2])))
        ivs_err_mean[i] = np.mean(np.abs((IVS[i][2:-3] - IVSnew[i][2:-2])))

    return ivs_rmse, ivs_err_min, ivs_err_max, ivs_err_mean



def doppler_compare(u,ud,dt,N,k):
    rmse_dop=[None]*(N-1)
    for i in range(N-1):
        rmse_dop[i] = np.sqrt(
            np.sum((np.abs(np.diff(u[i][2:-3]) / dt) - np.abs(np.diff(ud[i][2:-3]) / dt)) ** 2) / (k - 5))

    return rmse_dop


def internal_consistency(u,dist,T,N,k):
    distu=[None]*N
    err = [None] * N
    for i in range(N):
        distu[i] = np.zeros(k - 1)
        for l in np.arange(1, k - 1):
            distu[i][l] = sp.integrate.trapz(u[i][0:l + 1], T[0:l + 1])
        err[i] = np.sqrt(np.sum((dist[i][:-1] - distu[i]) ** 2) / (k - 1))

    return err, distu