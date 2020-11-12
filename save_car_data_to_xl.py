from xlwt import Workbook
import numpy as np

def save_data(x,y,z,u,ud,sxy,sz,ivs,N,k,output_file_name):
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    for j in range(N):
        sheet1.write(0, j,       f"x{j+1}")
        sheet1.write(0, j+N,     f"y{j+1}")
        sheet1.write(0, j+(2*N), f"z{j+1}")
        sheet1.write(0, j+(3*N), f"u{j+1}")
        sheet1.write(0, j+(4*N), f"ud{j+1}")
        sheet1.write(0, j+(5*N), f"sxy{j+1}")
        sheet1.write(0, j+(6*N), f"sz{j+1}")


        for i in range(k):
            sheet1.write(i+1, j, x[j][i])
            sheet1.write(i+1, j + N, y[j][i])
            sheet1.write(i+1, j + (2 * N), z[j][i])
            sheet1.write(i+1, j + (3 * N), u[j][i])
            sheet1.write(i+1, j + (4 * N), ud[j][i])
            sheet1.write(i+1, j + (5 * N), sxy[j][i])
            sheet1.write(i+1, j + (6 * N), sz[j][i])

    for j in range(N-1):
        sheet1.write(0, j + (7 * N), f"ivs{j + 1}")
        for i in range(k):
            sheet1.write(i + 1, j + (7 * N), ivs[j][i])

    wb.save(output_file_name)

    return None



def save_output(x,y,z,u,ud,udn,uks,uma,ubut,dist,dud,dudn,duks,duma,dubut,N,k,dt,output_file_name):
    wb = Workbook()

    # add_sheet is used to create sheet.
    sheet1 = wb.add_sheet('Sheet 1')
    acc=[None]*N
    accd = [None] * N
    accdn = [None] * N
    accks = [None] * N
    accma = [None] * N
    accbut = [None] * N
    for j in range(N):

        acc[j]=np.diff(u[j])/dt
        accd[j] = np.diff(ud[j]) / dt
        accdn[j] = np.diff(udn[j]) / dt
        accks[j] = np.diff(uks[j]) / dt
        accma[j] = np.diff(uma[j]) / dt
        accbut[j] = np.diff(ubut[j]) / dt

        sheet1.write(0, j,       f"x{j+1}")
        sheet1.write(0, j+N,     f"y{j+1}")
        sheet1.write(0, j+(2*N), f"z{j+1}")
        sheet1.write(0, j+(3*N), f"u{j+1}")
        sheet1.write(0, j+(4*N), f"ud{j+1}")
        sheet1.write(0, j + (5 * N), f"udn{j+1}")
        sheet1.write(0, j + (6 * N), f"uks{j + 1}")
        sheet1.write(0, j + (7 * N), f"uma{j + 1}")
        sheet1.write(0, j + (8 * N), f"ubut{j + 1}")
        sheet1.write(0, j + (9 * N), f"acc{j + 1}")
        sheet1.write(0, j + (10 * N), f"accd{j + 1}")
        sheet1.write(0, j + (11 * N), f"accdn{j + 1}")
        sheet1.write(0, j + (12 * N), f"accks{j + 1}")
        sheet1.write(0, j + (13 * N), f"accma{j + 1}")
        sheet1.write(0, j + (14 * N), f"accbut{j + 1}")
        sheet1.write(0, j + (15 * N), f"dist{j + 1}")
        sheet1.write(0, j + (16 * N), f"dud{j + 1}")
        sheet1.write(0, j + (17 * N), f"dudn{j + 1}")
        sheet1.write(0, j + (18 * N), f"duks{j + 1}")
        sheet1.write(0, j + (19 * N), f"duma{j + 1}")
        sheet1.write(0, j + (20 * N), f"dubut{j + 1}")



        for i in range(k):
            sheet1.write(i+1, j, x[j][i])
            sheet1.write(i+1, j + N, y[j][i])
            sheet1.write(i+1, j + (2 * N), z[j][i])
            sheet1.write(i+1, j + (3 * N), u[j][i])
            sheet1.write(i+1, j + (4 * N), ud[j][i])
            sheet1.write(i + 1, j + (5 * N), udn[j][i])
            sheet1.write(i + 1, j + (6 * N), uks[j][i])
            sheet1.write(i + 1, j + (7 * N), uma[j][i])
            sheet1.write(i + 1, j + (8 * N), ubut[j][i])



        for i in range(k-1):
            sheet1.write(i + 1, j + (9 * N),  acc[j][i])
            sheet1.write(i + 1, j + (10 * N), accd[j][i])
            sheet1.write(i + 1, j + (11 * N), accdn[j][i])
            sheet1.write(i + 1, j + (12 * N), accks[j][i])
            sheet1.write(i + 1, j + (13 * N), accma[j][i])
            sheet1.write(i + 1, j + (14 * N), accbut[j][i])

        for i in range(k - 2):
            sheet1.write(i + 1, j + (15 * N), dist[j][i+1])
            sheet1.write(i + 1, j + (16 * N), dud[j][i+1])
            sheet1.write(i + 1, j + (17 * N), dudn[j][i+1])
            sheet1.write(i + 1, j + (18 * N), duks[j][i+1])
            sheet1.write(i + 1, j + (19 * N), duma[j][i+1])
            sheet1.write(i + 1, j + (20 * N), dubut[j][i+1])



    # for j in range(N-1):
    #     sheet1.write(0, j + (7 * N), f"ivs{j + 1}")
    #     for i in range(k):
    #         sheet1.write(i + 1, j + (7 * N), ivs[j][i])

    wb.save(output_file_name)

    return None


def save_metrics(err_rmse,err_min,err_max,err_mean,err_dop,err_total,output_file_name):
    wb = Workbook()
    sheet1 = wb.add_sheet('Sheet 1')

    sheet1.write(0, 1, "doppler")
    sheet1.write(0, 2, "doppler corrected")
    sheet1.write(0, 3, "moving average")
    sheet1.write(0, 4, "butterworth")
    sheet1.write(0, 5, "kalman")

    sheet1.write(1, 0, "ivs_err_min")
    sheet1.write(2, 0, "ivs_err_max")
    sheet1.write(3, 0, "ivs_err_mean")
    sheet1.write(4, 0, "ivs_err_rmse")
    sheet1.write(5, 0, "err_dop")
    sheet1.write(6, 0, "err_total")

    for j in range(5):
        sheet1.write(1, j + 1, err_min[j])
        sheet1.write(2, j + 1, err_max[j])
        sheet1.write(3, j + 1, err_mean[j])
        sheet1.write(4, j + 1, err_rmse[j])

    for j in range(3):
        sheet1.write(5, j + 3, err_dop[j])
        sheet1.write(6, j + 3, err_total[j])

    wb.save(output_file_name)

    return None