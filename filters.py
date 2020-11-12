import numpy as np
from scipy import signal

def kalman(X, Y, Z, U, ivs, sigma_xy, sigma_z, N, dt, k, w_u_sigma, w_ivs_sigma):
    ####################################################
    ####################################################
    ####################################################
    ################# KALMAN FILTER ####################
    ####################################################
    ####################################################
    ####################################################
    n = 2 * N - 1  # number of state variables
    w_u_mean = 0
    # w_u_sigma = 0.6
    w_ivs_mean = 0
    # w_ivs_sigma = 10
    # w = np.block([np.random.normal(w_u_mean, w_u_sigma, 5), np.random.normal(w_ivs_mean, w_ivs_sigma, 4)])
    # w = np.block([np.array([1,1,1,1,1])*w_u_sigma, np.array([1,1,1,1])*w_ivs_sigma])
    w = np.block([ w_u_sigma, w_ivs_sigma])

    ############## STATIONARY MATRICES #############
    C = np.eye(n)
    D = np.eye(n)
    A = np.eye(n)

    for i in np.arange(N, n, 1):
        A[i, i - N] = dt
        A[i, i - N + 1] = -dt
    ############## STATIONARY MATRICES #############

    Pp = ((0)*np.random.rand(n)+1) * np.eye(n)

    ############## CALCULATION OF CO-VARIANCES   /    INITIALIZE STATE-OUTPUT VECTORS #############
    Xstate = np.zeros([n, k])
    Xpredict = np.zeros([n, k])
    res = np.zeros([n, k])
    zeta = np.zeros(n)
    varIVS = np.zeros([N - 1, k])
    varU = np.zeros([N, k])
    for i in range(N):
        Xpredict[i,0]=U[i][0]
        Xstate[i, 0] = U[i][0]
    for i in np.arange(N, n):
        Xpredict[i,0]=ivs[i-N][0]
        Xstate[i, 0] = ivs[i - N][0]

    # varU=[None]*N
    dUdX0 = [None] * N
    dUdX1 = [None] * N
    dUdY0 = [None] * N
    dUdY1 = [None] * N
    dUdZ0 = [None] * N
    dUdZ1 = [None] * N

    # varIVS=[None]*(N-1)
    dDSdXn = [None] * (N - 1)
    dDSdXp = [None] * (N - 1)
    dDSdYn = [None] * (N - 1)
    dDSdYp = [None] * (N - 1)
    dDSdZn = [None] * (N - 1)
    dDSdZp = [None] * (N - 1)

    for j in range(N):
        utol = 0.00000001
        zero_u = np.where(U[j][0:-1] < utol)
        U[j][zero_u]=0.00001

        dUdX1[j] = (1 / dt) * np.diff(X[j]) / U[j][0:-1]
        # dUdX1[j][zero_u] = 0
        dUdX0[j] = -dUdX1[j]
        dUdY1[j] = (1 / dt) * np.diff(Y[j]) / U[j][0:-1]
        # dUdY1[j][zero_u] = 0
        dUdY0[j] = -dUdY1[j]
        dUdZ1[j] = (1 / dt) * np.diff(Z[j]) / U[j][0:-1]
        # dUdZ1[j][zero_u] = 0
        dUdZ0[j] = -dUdZ1[j]

    for j in range(N - 1):
        dDSdXn[j] = (X[j] - X[j + 1]) / ivs[j]
        dDSdXp[j] = -dDSdXn[j]
        dDSdYn[j] = (Y[j] - Y[j + 1]) / ivs[j]
        dDSdYp[j] = -dDSdYn[j]
        dDSdZn[j] = (Z[j] - Z[j + 1]) / ivs[j]
        dDSdZp[j] = -dDSdZn[j]
    ############## CALCULATION OF CO-VARIANCES   /    INITIALIZE STATE-OUTPUT VECTORS #############


    R = [None] * (k - 1)  # time-variant R-matrix
    P1 = [None] * (k - 1)  # time-variant covariance-matrix
    P2 = [None] * (k - 1)  # time-variant covariance-matrix
    Ks = [None] * (k - 1)  # time-variant Kalman_gain-matrix
    P1[0]=Pp
    P2[0] = Pp

    for r in range(1, k - 1):

        ################################################ ERROR MEASUREMENT COVARIANCE MATRIX##############################################################
        R[r] = np.zeros([n, n])

        for i in range(N):
            R[r][i, i] = (dUdX1[i][r] ** 2) * (sigma_xy[i][r + 1]) + (dUdX0[i][r] ** 2) * (sigma_xy[i][r]) + \
                         (dUdY1[i][r] ** 2) * (sigma_xy[i][r + 1]) + (dUdY0[i][r] ** 2) * (sigma_xy[i][r]) + \
                         (dUdZ1[i][r] ** 2) * (sigma_z[i][r + 1]) + (dUdZ0[i][r] ** 2) * (
                         sigma_z[i][r])  # MAIN DIAGONAL SPEED-VARIANCES

            varU[i, r] = R[r][i, i]
            zeta[i] = (varU[i, r])

        for i in np.arange(N + 1, n - 1):
            R[r][i, i] = (dDSdXn[i - N][r] ** 2) * (sigma_xy[i - N][r]) + (dDSdXp[i - N][r] ** 2) * (
            sigma_xy[i - N + 1][r]) + \
                         (dDSdYn[i - N][r] ** 2) * (sigma_xy[i - N][r]) + (dDSdYp[i - N][r] ** 2) * (
                         sigma_xy[i - N + 1][r]) + \
                         (dDSdZn[i - N][r] ** 2) * (sigma_z[i - N][r]) + (dDSdZp[i - N][r] ** 2) * (
                         sigma_z[i - N + 1][r])  # MAIN DIAGONAL IVS-VARIANCES

            varIVS[i - N, r] = R[r][i, i]
            zeta[i] = (varIVS[i - N, r])

            R[r][i, i - 1] = dDSdXn[i - N][r] * dDSdXp[i - N - 1][r] * (sigma_xy[i - N][r]) + \
                             dDSdYn[i - N][r] * dDSdYp[i - N - 1][r] * (sigma_xy[i - N][r]) + \
                             dDSdZn[i - N][r] * dDSdZp[i - N - 1][r] * (
                             sigma_z[i - N][r])  # SUBDIAGONAL CONSECUTIVE IVS-COVARIANCES

            R[r][i - 1, i] = R[r][i, i - 1]  # SUPERDIAGONAL CONSECUTIVE IVS-COVARIANCES

            R[r][i - N, i] = dUdX0[i - N][r] * dDSdXn[i - N][r] * (sigma_xy[i - N][r]) + \
                             dUdY0[i - N][r] * dDSdYn[i - N][r] * (sigma_xy[i - N][r]) + \
                             dUdZ0[i - N][r] * dDSdZn[i - N][r] * (sigma_z[i - N][r])

            R[r][i - N, i - 1] = dUdX0[i - N][r] * dDSdXn[i - N - 1][r] * (sigma_xy[i - N][r]) + \
                                 dUdY0[i - N][r] * dDSdYn[i - N - 1][r] * (sigma_xy[i - N][r]) + \
                                 dUdZ0[i - N][r] * dDSdZn[i - N - 1][r] * (sigma_z[i - N][r])

            R[r][i, i - N] = R[r][i - N, i]

            R[r][i - 1, i - N] = R[r][i - N, i - 1]

        R[r][n - 1, n - 1] = (dDSdXn[n - 1 - N][r] ** 2) * (sigma_xy[n - 1 - N][r]) + (dDSdXp[n - 1 - N][r] ** 2) * (
        sigma_xy[n - N][r]) + \
                             (dDSdYn[n - 1 - N][r] ** 2) * (sigma_xy[n - 1 - N][r]) + (dDSdYp[n - 1 - N][r] ** 2) * (
                             sigma_xy[n - N][r]) + \
                             (dDSdZn[n - 1 - N][r] ** 2) * (sigma_z[n - 1 - N][r]) + (dDSdZp[n - 1 - N][r] ** 2) * (
                             sigma_z[n - N][r])  # MAIN DIAGONAL IVS-VARIANCES LAST ELEMENT

        R[r][n - 1, n - 2] = dDSdXn[n - 1 - N][r] * dDSdXp[n - N - 2][r] * (sigma_xy[n - 1 - N][r]) + \
                             dDSdYn[n - 1 - N][r] * dDSdYp[n - N - 2][r] * (sigma_xy[n - 1 - N][r]) + \
                             dDSdZn[n - 1 - N][r] * dDSdZp[n - N - 2][r] * (
                             sigma_z[n - 1 - N][r])  # SUBDIAGONAL CONSECUTIVE IVS-COVARIANCES LAST ROW

        R[r][n - 2, n - 1] = R[r][n - 1, n - 2]  # SUPERDIAGONAL CONSECUTIVE IVS-COVARIANCES LAST COLUMN

        R[r][N, N] = (dDSdXn[0][r] ** 2) * (sigma_xy[0][r]) + (dDSdXp[0][r] ** 2) * (sigma_xy[1][r]) + \
                     (dDSdYn[0][r] ** 2) * (sigma_xy[0][r]) + (dDSdYp[0][r] ** 2) * (sigma_xy[1][r]) + \
                     (dDSdZn[0][r] ** 2) * (sigma_z[0][r]) + (dDSdZp[0][r] ** 2) * (
                     sigma_z[1][r])  # MAIN DIAGONAL IVS-VARIANCES FIRST ELEMENT (N,N)

        varIVS[0, r] = R[r][N, N]
        varIVS[N - 2, r] = R[r][n - 1, n - 1]
        zeta[N] = (varIVS[0, r])
        zeta[n - 1] = (varIVS[N - 2, r])

        R[r][n - 1 - N, n - 1] = dUdX0[n - 1 - N][r] * dDSdXn[n - 1 - N][r] * (sigma_xy[n - 1 - N][r]) + \
                                 dUdY0[n - 1 - N][r] * dDSdYn[n - 1 - N][r] * (sigma_xy[n - 1 - N][r]) + \
                                 dUdZ0[n - 1 - N][r] * dDSdZn[n - 1 - N][r] * (sigma_z[n - 1 - N][r])

        R[r][n - 1 - N, n - 1 - 1] = dUdX0[n - 1 - N][r] * dDSdXn[n - 1 - N - 1][r] * (sigma_xy[n - 1 - N][r]) + \
                                     dUdY0[n - 1 - N][r] * dDSdYn[n - 1 - N - 1][r] * (sigma_xy[n - 1 - N][r]) + \
                                     dUdZ0[n - 1 - N][r] * dDSdZn[n - 1 - N - 1][r] * (sigma_z[n - 1 - N][r])

        R[r][n - N, n - 1] = dUdX0[n - N][r] * dDSdXn[n - N - 1][r] * (sigma_xy[n - N][r]) + \
                             dUdY0[n - N][r] * dDSdYn[n - N - 1][r] * (sigma_xy[n - N][r]) + \
                             dUdZ0[n - N][r] * dDSdZn[n - N - 1][r] * (sigma_z[n - N][r])

        R[r][0, N] = dUdX0[0][r] * dDSdXn[0][r] * (sigma_xy[0][r]) + \
                     dUdY0[0][r] * dDSdYn[0][r] * (sigma_xy[0][r]) + \
                     dUdZ0[0][r] * dDSdZn[0][r] * (sigma_z[0][r])

        R[r][n - 1, n - N] = R[r][n - N, n - 1]

        R[r][n - 1, n - 1 - N] = R[r][n - 1 - N, n - 1]

        R[r][n - 1 - 1, n - 1 - N] = R[r][n - 1 - N, n - 1 - 1]

        R[r][N, 0] = R[r][0, N]
        ################################################ ERROR MEASUREMENT COVARIANCE MATRIX##############################################################

        # w_u_mean = 0
        # # w_u_sigma = 0.6
        # w_ivs_mean = 0
        # # w_ivs_sigma = 10
        # w = np.block([np.random.normal(w_u_mean, w_u_sigma, 5), np.random.normal(w_ivs_mean, w_ivs_sigma, 4)])

        # w1=0.8
        # w2=0.1
        # w=[w1,w1,w1,w1,w1,w2,w2,w2,w2]*np.eye(9)

        Xpredict[:, r] = np.dot(A, Xstate[:, r - 1])

        Yout = [U[0][r], U[1][r], U[2][r], U[3][r], U[4][r], ivs[0][r], ivs[1][r], ivs[2][r], ivs[3][r]] #+ zeta

        K = np.dot(Pp, np.linalg.inv(Pp + R[r]))
        Ks[r]=K


        Xstate[:, r] = Xpredict[:, r] + np.dot(K, (Yout - Xpredict[:, r]))

        res[:,r]=Yout - Xpredict[:, r]

        Pf = np.dot(np.eye(n) - K, Pp)
        P2[r]=Pf

        Pp = np.dot(A, np.dot(Pf, np.transpose(A))) + np.diag(w ** 2)
        P1[r] = Pp

    ####################################################
    ####################################################
    ####################################################
    ################# KALMAN Smoother ##################
    ####################################################
    ####################################################
    ####################################################

    Xsmooth = np.zeros([n, k-1])
    Xsmooth[:,k-2]=Xstate[:,k-2]
    Ps=P2[k-2]
    for r in range(k-3,-1,-1):

        K=np.dot(P2[r], np.dot(np.transpose(A), np.linalg.inv(P1[r+1])))

        Ps=P2[r]-np.dot(K, np.dot(P1[r+1]-Ps, np.transpose(K)))

        Xsmooth[:,r]=Xstate[:,r] + np.dot(K,Xsmooth[:,r+1]-Xpredict[:,r+1])

    # Xsmooth[:, 0]=2*Xsmooth[:,1]-Xsmooth[:,2]
    # Xsmooth[:, 0] = (7/3)*Xsmooth[:, 1] - Xsmooth[:, 2] - Xsmooth[:, 3] + (2/3)*Xsmooth[:, 4]
    # Xsmooth[:, 0] = Xsmooth[:, 1]

    # for i in range(N):
    #     Xsmooth[i,:]=np.append(Xsmooth[i,:],2*Xsmooth[i,N-2]-Xsmooth[i,N-1])

    return Xstate[:,:-1],Xsmooth



def medmav(Ssignal,N,Nm):
    '''

    Smoothing of a signal. We use a median filter and a moving average

    :param Ssignal:
    :param N: smoothing parameter
    :param Nm: filtering parameter
    :return: res : Smoothed signal
    '''

    B = np.ones((N, 1), dtype='float64').flatten()
    A = N * np.ones((1, 1), dtype='float64').flatten()

    # Apply median filtering to remove outliers
    res = signal.medfilt(Ssignal, Nm)

    # Apply smoothing
    res = signal.filtfilt(B, A, res)

    return res


def butrw(ssignal,order,cfreq):

    B, A = signal.butter(order, cfreq, output='ba')
    res = signal.filtfilt(B, A, ssignal)

    return res