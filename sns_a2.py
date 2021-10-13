import numpy as np
import math
import time
import matplotlib
import matplotlib.pyplot as plt
from itertools import product
from hirvonen import hirvonen
from readrnx_studenci3 import nav, indnav, obs, iobs, RECV, week_start, TOW_start, week_stop, TOW_stop, dt, WROC, start_day, choice, date2tow, c, wE, U
from matplotlib.figure import Figure
from datetime import *


def el_az(sat):
    fi, lamb, h = hirvonen(RECV[0], RECV[1], RECV[2])
    neu = np.array([[-math.sin(fi) * math.cos(lamb), -math.sin(lamb), math.cos(fi) * math.cos(lamb)],
                    [-math.sin(fi) * math.sin(lamb), math.cos(lamb), math.cos(fi) * math.sin(lamb)],
                    [math.cos(fi), 0, math.sin(fi)]])
    xyz = np.array([RECV[0], RECV[1], RECV[2]])
    coordSatellite = np.array([sat[0], sat[1], sat[2]])
    sat_vec = np.transpose(np.array(coordSatellite - xyz))
    xyz_neu = np.dot(np.transpose(neu), sat_vec)
    el = np.rad2deg(math.asin(xyz_neu[2] / (math.sqrt(xyz_neu[0] ** 2 + xyz_neu[1] ** 2 + xyz_neu[2] ** 2))))
    if np.rad2deg( math.atan2(xyz_neu[1], xyz_neu[0])) > 0:
        az = np.rad2deg(math.atan2(xyz_neu[1], xyz_neu[0]))
    else:
        az = np.rad2deg(math.atan2(xyz_neu[1], xyz_neu[0])) + 360
    return el, az


def dops(A):
    Q = np.linalg.inv(np.dot(A.T, A))
    GDOP = math.sqrt(Q.diagonal()[0] + Q.diagonal()[1] + Q.diagonal()[2] + Q.diagonal()[3])
    PDOP = math.sqrt(Q.diagonal()[0] + Q.diagonal()[1] + Q.diagonal()[2])
    TDOP = math.sqrt(Q.diagonal()[3])
    fi, lam, h = hirvonen(RECV[0], RECV[1], RECV[2])
    R = np.array([[-math.sin(fi) * math.cos(lam), -math.sin(lam), math.cos(fi) * math.cos(lam)],
                  [-math.sin(fi) * math.sin(lam), math.cos(lam), math.cos(fi) * math.sin(lam)],
                  [math.cos(fi), 0, math.sin(fi)]])
    Qneu = np.dot(np.dot(R.T, Q[0:3, 0:3]), R)
    HDOP = math.sqrt(Qneu.diagonal()[0] + Qneu.diagonal()[1])
    VDOP = math.sqrt(Qneu.diagonal()[2])
    return PDOP,GDOP,TDOP,HDOP,VDOP


def satellite_position(time, data, week):
    toe = data[17]
    gps_week = data[27]
    a = (data[16]) ** 2
    M0 = data[12]
    dn = data[11]
    e = data[14]
    w = data[23]
    Cus = data[15]
    Cuc = data[13]
    Cic = data[18]
    Cis = data[20]
    Crc = data[22]
    Crs = data[10]
    IDOT = data[25]
    i0 = data[21]
    Omega0 = data[19]
    Omega = data[24]
    af0 = data[6]
    af1 = data[7]
    af2 = data[8]
    tow_all = week * 604800 + time
    toe_full = gps_week * 604800 + toe
    t_k = tow_all - toe_full
    u = 3.986005 * 10 ** 14
    wE = 7.2921151467 * 10 ** (-5)
    n0 = math.sqrt(u / a ** 3)
    n = n0 + dn
    Mk = M0 + n * t_k
    Ep = Mk
    En = Mk + e * math.sin(Ep)
    while abs(En - Ep) >= 10 ** (-12):
        Ep = En
        En = Mk + e * math.sin(Ep)
    Vk = math.atan2(math.sqrt(1 - e ** 2) * math.sin(En), math.cos(En) - e)
    FiK = Vk + w
    duk = Cus * math.sin(2 * FiK) + Cuc * math.cos(2 * FiK)
    drk = Crs * math.sin(2 * FiK) + Crc * math.cos(2 * FiK)
    dik = Cis * math.sin(2 * FiK) + Cic * math.cos(2 * FiK)
    uk = FiK + duk
    rk = a * (1 - e * math.cos(En)) + drk
    ik = i0 + IDOT * t_k + dik
    xk = rk * math.cos(uk)
    yk = rk * math.sin(uk)
    OmegaK = Omega0 + (Omega - wE) * t_k - wE * toe
    Xk = xk * math.cos(OmegaK) - yk * math.cos(ik) * math.sin(OmegaK)
    Yk = xk * math.sin(OmegaK) + yk * math.cos(ik) * math.cos(OmegaK)
    Zk = yk * math.sin(ik)
    delta_trel = (-2 * math.sqrt(U) / c ** 2) * e * math.sqrt(a) * math.sin(En)
    dts = af0 + af1 * (t_k) + af2 * (t_k) ** 2 + delta_trel
    return Xk, Yk, Zk, dts


def matrixX(A, y):
    matx = np.dot(np.dot(-np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), y)
    return matx


def matrixA(satellite, dist):
    matA = [(-(satellite[0] - RECV[0]) / dist)[0], (-(satellite[1] - RECV[1]) / dist)[0],
            (-(satellite[2] - RECV[2]) / dist)[0], 1]
    return matA


def get_y(pseudo_dist, observation):
    ind = np.where(np.isnan(observation))
    observation[ind] = pseudo_dist[ind]
    return pseudo_dist - observation


def plot_time():
    tab=[]
    hour_start = time_start[3]
    hour_end = time_end[3]
    if hour_end == 0:
        hour_end = 24
    for hours in range(hour_start, hour_end):
        for minutes in range(0,60):
                for seconds in range(0,59,dt):
                    temp = datetime.datetime(time_start[0],time_start[1],time_start[2],hours,minutes,seconds)
                    tab.append(temp)
    return tab


def iono(time, el, az):
    alpha = [9.3132E-09, 0.0000E+00, -5.9605E-08, 0.0000E+00]
    beta = [9.0112E+04, 0.0000E+00, -1.9661E+05, 0.0000E+00]
    fir, lamr, hr = hirvonen(RECV[0], RECV[1], RECV[2])
    tow = time
    els = el / 180
    # kat geocentryczny
    psi = 0.0137 / (els + 0.11) - 0.022
    # szerokosc_ipp
    fi_ipp = fir / 180 + psi * math.cos(np.deg2rad(az))
    if fi_ipp > 0.416:
        fi_ipp = 0.416
    elif fi_ipp < -0.416:
        fi_ipp = -0.416
    lam_ipp = lamr / 180 + psi * math.sin(np.deg2rad(az)) / math.cos(fi_ipp * np.pi)
    fi_m = fi_ipp + 0.064 * math.cos((lam_ipp - 1.617) * np.pi)
    t = 43200 * lam_ipp + tow
    t = math.fmod(t, 86400)
    if t >= 86400:
        t -= 86400
    elif t < 0:
        t += 86400
    Aion = alpha[0] + alpha[1] * fi_m + alpha[2] * fi_m ** 2 + alpha[3] * fi_m ** 3
    if Aion < 0:
        Aion = 0
    Pion = beta[0] + beta[1] * fi_m + beta[2] * fi_m ** 2 + beta[3] * fi_m ** 3
    if Pion < 72000:
        Pion = 72000
    fi_ion = 2 * np.pi * (t - 50400) / Pion
    mf = 1 + 16 * (0.53 - els) ** 3
    delta_L1 = c * mf * (5 * 10 ** (-9) + Aion * (1 - (fi_ion ** 2) / 2 + (fi_ion ** 4) / 24)) \
        if abs(fi_ion <= math.pi / 2) \
        else c * mf * (5 * 10 ** (-9))
    return delta_L1


def tropo(el):
    hort = 140.857

    p = 1013.25 * ((1 - 0.0000226 * hort) ** 5.225)
    t = 291.15 - 0.0065 * hort
    e = 6.11 * (0.5 * math.exp(-0.0006396 * hort)) * (10 ** ((7.5 * (t - 273.15)) / (t - 35.85)))
    Nd = 77.64 * (p / t)
    Nw = -12.96 * (e / t) + (3.718 * 10 ** 5) * (e / (t ** 2))
    hd = 40136 + 148.72 * (t - 273.15)
    hw = 11000
    delta_Td0 = (10 ** (-6) / 5) * Nd * hd
    delta_Tw0 = (10 ** (-6) / 5) * Nw * hw
    md = 1 / math.sin(np.deg2rad((math.sqrt(el ** 2 + 6.25))))
    mw = 1 / math.sin(np.deg2rad((math.sqrt(el ** 2 + 2.25))))
    delta_Td = md * delta_Td0
    delta_Tw = mw * delta_Tw0

    delta_T = delta_Td + delta_Tw
    return delta_T


def epoch_param(nrSatellite, time):
    frame = nav[np.where(np.any(indnav == nrSatellite, axis=1))[0]]
    return np.append(frame[np.argmin(abs(time - frame[:, 17]))],
                     [nrSatellite])


def correction(satellite_coord, T):
    rot = np.dot([[math.cos(wE * T), math.sin(wE * T), 0],
                  [-math.sin(wE * T), math.cos(wE * T), 0],
                  [0, 0, 1]], [[satellite_coord[0]], [satellite_coord[1]], [satellite_coord[2]]])
    rot = np.transpose(np.append(rot, [satellite_coord[3]]))
    return rot


def geom_dist(XYZ):
    ro = math.sqrt((XYZ[0] - RECV[0]) ** 2 + (XYZ[1] - RECV[1]) ** 2 + (XYZ[2] - RECV[2]) ** 2)
    return ro


def pseudo_dist(satellite, dist, corr_iono, corr_topo):
    if corr_iono == 0:
        pseudoDist = dist - c * satellite[3] + c * RECV[3] + corr_topo
    if corr_topo == 0:
        pseudoDist = dist - c * satellite[3] + c * RECV[3] + corr_iono
    else:
        pseudoDist = dist - c * satellite[3] + c * RECV[3] + corr_topo + corr_iono
    #print(corr_topo, corr_iono)
    return pseudoDist
RECV.append(0)


def differences_dops(t_corr, i_corr):
    recv_coord = np.zeros([0, 4])
    dop = np.zeros([0, 5])
    i = 0
    for time in range(TOW_start, TOW_stop, dt):
        recv = np.zeros([0, 4])
        dist = np.zeros([0, 1])
        T = 0.072
        idx = iobs[:, 2] == time
        iobss = iobs[idx, :]
        obss = obs[idx]
        i += 0
        for j in range(4):
            XYZ, correctedXYZ, geometrical_distance, pseudo_distance, y, matA, elevation, azimuth, = \
                np.zeros([0, 4]), np.zeros([0, 4]), np.zeros([0, 1]), np.zeros([0, 1]),\
                np.zeros([0, 1]), np.zeros([0, 4]), np.zeros([0, 1]), np.zeros([0, 1])
            for i, nrSat in enumerate(iobss):
                if j > 0:
                    T = dist[i, 0] / c
                data = epoch_param(nrSat, time)
                tobs = iobss[i, 2]
                tr = tobs + RECV[3] - T
                el = el_az(satellite_position(tr, data, week_start))[0]
                az = el_az(satellite_position(tr, data, week_start))[1]
                #if el > mask:
                elevation = np.vstack([elevation, el])
                azimuth = np.vstack([azimuth, az])
                XYZ = np.vstack([XYZ, satellite_position(tr, data, week_start)])
                correctedXYZ = np.vstack([correctedXYZ, correction(XYZ[i], T)])
                geometrical_distance = np.vstack([geometrical_distance, geom_dist(correctedXYZ[i])])
                if t_corr:
                    corr_tropo = tropo(el)
                else:
                    corr_tropo = 0
                if i_corr:
                    corr_iono = iono(time, el, az)
                else:
                    corr_iono = 0
                pseudo_distance = np.vstack([pseudo_distance, pseudo_dist(correctedXYZ[i], geometrical_distance[i],corr_iono, corr_tropo)])
                y = np.vstack([y, get_y(pseudo_distance[i], obss[i])])
                matA = np.vstack([matA, matrixA(correctedXYZ[i], geometrical_distance[i])])
            matX = matrixX(matA, y)
            RECV[0] += matX[0, 0]
            RECV[1] += matX[1, 0]
            RECV[2] += matX[2, 0]
            RECV[3] += matX[3, 0] / c
            recv = np.vstack([recv, RECV])
            last = recv[-1]
            dist = geometrical_distance
            A = matA

        PDOP,GDOP,TDOP,HDOP,VDOP = dops(A)
        dop = np.vstack([dop, [GDOP, PDOP, TDOP, HDOP, VDOP]])
        recv_coord = np.vstack([recv_coord, last])
    diff = recv_coord[:, 0:-1] - WROC
    return diff, dop





def plot_xyz_diff(diff,title):
    time = plot_time()
    plt.figure(figsize=(15, 5))
    plt.plot(time,diff)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Difference [m]")
    legend = ['X', 'Y', 'Z']
    plt.legend(legend)
    plt.show()


def plot_diff_mean_std(diff,title):
    xdiff = diff[:,0]
    ydiff = diff[:,1]
    zdiff = diff[:,2]
    time = plot_time()
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.suptitle(title)
    ax1.set_title("Coordinate X")
    ax1.plot(time, xdiff)
    ax1.plot(time, [np.mean(xdiff) for i in range(len(xdiff))], label='mean')
    ax1.plot(time, [np.std(xdiff) for i in range(len(xdiff))],label = 'std')
    ax1.legend()
    ax2.set_title("Coordinate Y")
    ax2.plot(time, ydiff)
    ax2.plot(time, [np.mean(ydiff) for i in range(len(ydiff))], label='mean')
    ax2.plot(time, [np.std(ydiff) for i in range(len(ydiff))],label = 'std')
    ax2.legend()
    ax3.set_title("Coordinate Z")
    ax3.plot(time, zdiff)
    ax3.plot(time, [np.mean(zdiff) for i in range(len(zdiff))], label='mean')
    ax3.plot(time, [np.std(zdiff) for i in range(len(zdiff))],label = 'std')
    ax3.legend()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Difference [m]")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Difference [m]")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Difference [m]")
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

def plot_dops(dop):
    time = plot_time()
    plt.figure(figsize=(15, 5))
    plt.plot(time, dop)
    plt.title("DOP Parameters")
    plt.xlabel("Time")
    plt.ylabel("Values")
    legendDOP = ['PDOP','GDOP',  'TDOP', 'HDOP', 'VDOP']
    plt.legend(legendDOP)
    plt.show()

def plot_diff_atmospheric(none, iono, tropo,title):
    xnone = none[:,0]
    ynone = none[:,1]
    znone = none[:,2]
    xiono = iono[:, 0]
    yiono = iono[:, 1]
    ziono = iono[:, 2]
    xtropo = tropo[:, 0]
    ytropo = tropo[:, 1]
    ztropo = tropo[:, 2]
    time = plot_time()
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig.suptitle(title)
    ax1.set_title("Coordinate X")
    ax1.plot(time, xnone, label='none')
    ax1.plot(time,xiono, label='ionospheric')
    ax1.plot(time,xtropo,label = 'tropospheric')
    ax2.set_title("Coordinate Y")
    ax2.plot(time, ynone, label='none')
    ax2.plot(time,yiono, label='ionospheric')
    ax2.plot(time,ytropo,label = 'tropospheric')
    ax2.legend()
    ax3.set_title("Coordinate Z")
    ax3.plot(time, znone, label='none')
    ax3.plot(time,ziono, label='ionospheric')
    ax3.plot(time,ztropo,label = 'tropospheric')
    ax3.legend()
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Difference [m]")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Difference [m]")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Difference [m]")
    plt.subplots_adjust(hspace=0.5)
    plt.show()


none, none_dopy = differences_dops(False, False)
Iono, iono_dopy = differences_dops(False, True)
Tropo, tropo_dopy = differences_dops(True, False)
both, both_dopy = differences_dops(True, True)

plot_title1 = 'Statistics'
plot_diff_mean_std(both,plot_title1)
plot_dops(both_dopy)
plot_title2 = 'Atmospheric corrections '
plot_diff_atmospheric(none,Iono,Tropo,plot_title2)
plot_title3 = 'XYZ coordinates differences '
plot_xyz_diff(none,plot_title3)
