from os import listdir
from os.path import isfile, join
import numpy as np
from scipy.interpolate import *
import csv



def car_data(path,start,stop,dt,ud_ind):

    k=stop-start

    count_csv = 0
    dd = []
    mypath = path
    cf_data_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for i in range(len(cf_data_files)):
        cf = ''.join(cf_data_files[i])
        if cf[len(cf) - 1] == 'v':
            # p2c.convert(mypath + '\\' + cf_data_files[i])
            count_csv = count_csv + 1
            dd.append(count_csv)
            dd[count_csv - 1] = mypath + '/' + cf_data_files[i]

    # sort data / set leader and followers according to correct position
    data_path = [None] * count_csv
    for j in range(count_csv):
        ind = int(dd[j][len(dd[j]) - 5])
        data_path[ind - 1] = dd[j]
    # sort data / set leader and followers according to correct position
    N = count_csv  # number of cars

    data = [None] * N
    h = [None] * N
    x = [None] * N
    y = [None] * N
    z = [None] * N
    t = [None] * N
    s = [None] * N
    X = [None] * N
    Y = [None] * N
    Z = [None] * N
    T = [None] * N
    SC = [None] * N
    H = [None] * N
    Ux = [None] * N
    Uy = [None] * N
    Uz = [None] * N
    U = [None] * N  # speed from measured position
    Ui = [None] * N  # speed from doppler interpolated
    Uxma = [None] * N
    Uyma = [None] * N
    Uzma = [None] * N
    Uma = [None] * N  # filtered speed (moving average)
    ax = [None] * N
    ay = [None] * N
    az = [None] * N
    a = [None] * N
    xx = [None] * N
    yy = [None] * N
    zz = [None] * N
    axma = [None] * N
    ayma = [None] * N
    azma = [None] * N
    ama = [None] * N
    ak = [None] * N
    ux = [None] * N
    uy = [None] * N
    uz = [None] * N
    uxx = [None] * N
    uyy = [None] * N
    uzz = [None] * N
    u = [None] * N  # measured speed
    uu = [None] * N
    sigma_u = [None] * N
    sigma_xy = [None] * N
    sigma_z = [None] * N
    csx = [None] * N
    csy = [None] * N
    csz = [None] * N
    css = [None] * N
    csux = [None] * N
    csuy = [None] * N
    csuz = [None] * N
    csu = [None] * N
    cssu = [None] * N
    cssxy = [None] * N
    cssz = [None] * N
    csxp1 = [None] * N
    csyp1 = [None] * N
    csxp0 = [None] * N
    csyp0 = [None] * N
    chead = [None] * N
    Ttemp1 = np.zeros(N)
    Ttemp2 = np.zeros(N)
    kk = [None] * N
    IVS = [None] * N
    distP = [None] * N
    car_start_T = np.zeros(N)
    car_stop_T = np.zeros(N)

    SL = 435.31  # speed limit outliers

    for j in range(N):
        with open(data_path[j], 'r') as f:
            reader = csv.reader(f, delimiter=',')
            headers = next(reader)
            data[j] = list(reader)
            data[j] = np.array(data[j]).astype(float)
            u[j] = data[j][:, 12][data[j][:, 12] < SL]  # measured speed
            a[j] = np.diff(u[j])/dt
            uy[j] = data[j][:, 9][data[j][:, 12] < SL]  # measured speed
            ux[j] = data[j][:, 10][data[j][:, 12] < SL]  # measured speed
            uz[j] = data[j][:, 11][data[j][:, 12] < SL]  # measured speed
            x[j] = data[j][:, 16][data[j][:, 12] < SL]  # -data[N-1][0, 16]  # /1000 #x-coordinate
            y[j] = data[j][:, 17][data[j][:, 12] < SL]  # -data[N-1][0, 17]  # /1000 #y-coordinate
            z[j] = data[j][:, 18][data[j][:, 12] < SL]  # -data[N-1][0, 18]  # /1000 #z-coordinate
            h[j] = data[j][:, 13][data[j][:, 12] < SL]  # heading
            t[j] = data[j][:, 2][data[j][:, 12] < SL] / 1000  # time-coordinate
            sigma_u[j] = (data[j][:, 14][data[j][:, 12] < SL] / 1000) ** 1  # speed accuracy
            sigma_xy[j] = (data[j][:, 7][data[j][:, 12] < SL] / 1000) ** 1  # horizontal accuracy
            sigma_z[j] = (data[j][:, 8][data[j][:, 12] < SL] / 1000) ** 1  # vertical accuracy
            #################### fix monotonicity of data #####################
            tind = np.where(t[j] > t[j][-1])
            t[j] = np.delete(t[j], tind[0])
            x[j] = np.delete(x[j], tind[0])
            y[j] = np.delete(y[j], tind[0])
            z[j] = np.delete(z[j], tind[0])
            u[j] = np.delete(u[j], tind[0])
            s[j] = np.delete(s[j], tind[0])
            ux[j] = np.delete(ux[j], tind[0])
            uy[j] = np.delete(uy[j], tind[0])
            uz[j] = np.delete(uz[j], tind[0])
            h[j] = np.delete(h[j], tind[0])
            sigma_u[j] = np.delete(sigma_u[j], tind[0])
            sigma_xy[j] = np.delete(sigma_xy[j], tind[0])
            sigma_z[j] = np.delete(sigma_z[j], tind[0])

            tind1 = np.where(np.logical_or(a[j] >= ud_ind,a[j] <= -ud_ind))
            print(tind1)
            t[j] = np.delete(t[j], tind1[0])
            x[j] = np.delete(x[j], tind1[0])
            y[j] = np.delete(y[j], tind1[0])
            z[j] = np.delete(z[j], tind1[0])
            u[j] = np.delete(u[j], tind1[0])
            s[j] = np.delete(s[j], tind1[0])
            ux[j] = np.delete(ux[j], tind1[0])
            uy[j] = np.delete(uy[j], tind1[0])
            uz[j] = np.delete(uz[j], tind1[0])
            h[j] = np.delete(h[j], tind1[0])
            sigma_u[j] = np.delete(sigma_u[j], tind1[0])
            sigma_xy[j] = np.delete(sigma_xy[j], tind1[0])
            sigma_z[j] = np.delete(sigma_z[j], tind1[0])
            #################### fix monotonicity of data #####################

            h[j] = (h[j] * np.pi) / 180

            Ttemp1[j] = t[j][0]
            Ttemp2[j] = t[j][-1]

    for j in range(N):
        T[j] = np.arange(t[j][0], t[j][-1], dt)  # augmented time coordinate (for interpolation)
        T[j] = np.floor(T[j] * 10) / 10  # remove extra digits

        csx[j] = CubicSpline(t[j], x[j])
        csy[j] = CubicSpline(t[j], y[j])
        csz[j] = CubicSpline(t[j], z[j])
        csu[j] = CubicSpline(t[j], u[j])
        cssxy[j] = CubicSpline(t[j], sigma_xy[j])
        cssz[j] = CubicSpline(t[j], sigma_z[j])
        cssu[j] = CubicSpline(t[j], sigma_u[j])
        chead[j] = CubicSpline(t[j], h[j])

        X[j] = csx[j](T[j])
        Y[j] = csy[j](T[j])
        Z[j] = csz[j](T[j])
        U[j] = csu[j](T[j])
        sigma_xy[j] = cssxy[j](T[j])
        sigma_u[j] = cssu[j](T[j])
        sigma_z[j] = cssz[j](T[j])
        H[j] = chead[j](T[j])
        Ux[j] = np.diff(X[j]) / dt
        Uy[j] = np.diff(Y[j]) / dt
        Uz[j] = np.diff(Z[j]) / dt
        u[j] = np.sqrt((Ux[j] ** 2) + (Uy[j] ** 2) + (Uz[j] ** 2))

        car_start_T[j] = T[j][0]  # find time when all vehicles start moving
        car_stop_T[j] = T[j][-1]  # find time when all vehicles stop moving

    Tsync_start = np.ceil((np.max(car_start_T) + 0 * dt) * 10) / 10  # synchronized time-frame start
    Tsync_stop = np.floor((np.min(car_stop_T) - 0 * dt) * 10) / 10  # synchronized time-frame stop

    Tsync_start_index = [None] * N
    Tsync_stop_index = [None] * N
    dist_to_end = np.zeros(N)
    for j in range(N):
        temp1 = np.where(T[j] == Tsync_start)
        temp2 = np.where(T[j] == Tsync_stop)
        Tsync_start_index[j] = np.max(temp1)
        Tsync_stop_index[j] = np.max(temp2)

        X[j] = X[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        Y[j] = Y[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        Z[j] = Z[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        T[j] = T[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        U[j] = U[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        H[j] = H[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        u[j] = u[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        sigma_xy[j] = sigma_xy[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        sigma_z[j] = sigma_z[j][Tsync_start_index[j]:Tsync_stop_index[j]]
        sigma_u[j] = sigma_u[j][Tsync_start_index[j]:Tsync_stop_index[j]]

    del T
    T = np.arange(Tsync_start, Tsync_stop, dt)


    T = T[start:stop]-T[start]

    for j in range(N):
        distP[j] = np.zeros(k)
        X[j] = X[j] - X[N - 1][0]
        Y[j] = Y[j] - Y[N - 1][0]
        Z[j] = Z[j] - Z[N - 1][0]
        X[j] = X[j][start:stop]
        Y[j] = Y[j][start:stop]
        Z[j] = Z[j][start:stop]
        U[j] = U[j][start:stop]
        u[j] = u[j][start:stop]
        sigma_xy[j] = (sigma_xy[j][start:stop])**2
        sigma_z[j] = (sigma_z[j][start:stop])**2
        sigma_u[j] = (sigma_u[j][start:stop])**2

        for l in np.arange(1, k):
            distP[j][l] = distP[j][l - 1] + np.sqrt(
                (X[j][l] - X[j][l - 1]) ** 2 + (Y[j][l] - Y[j][l - 1]) ** 2 + (
                            Z[j][l] - Z[j][l - 1]) ** 2)
            if distP[j][l] <= distP[j][l - 1]:
                distP[j][l] = distP[j][l] + 0.0000001

    for i in range(N - 1):
        IVS[i] = np.sqrt((X[i] - X[i + 1]) ** 2 + (Y[i] - Y[i + 1]) ** 2 + (Z[i] - Z[i + 1]) ** 2)



    return X,Y,Z,U,u,IVS,distP,T,sigma_xy,sigma_z,sigma_u,N


