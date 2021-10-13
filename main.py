import numpy as np
import math
import matplotlib.pyplot as plt
from hirvonen import hirvonen
from readrnx_studenci import readrnxnav, readrnxobs, date2tow, c, wE, U
from datetime import *


def getNEU():
    fi, lamb, h = hirvonen(RECV[0], RECV[1], RECV[2])
    return np.array([[-math.sin(fi) * math.cos(lamb), -math.sin(lamb), math.cos(fi) * math.cos(lamb)],
                     [-math.sin(fi) * math.sin(lamb), math.cos(lamb), math.cos(fi) * math.sin(lamb)],
                     [math.cos(fi), 0, math.sin(fi)]])


def getCoordSatellite(t, data, week):
    toe = data[17]
    gps_week = data[27]
    a = (data[16]) ** 2
    dn = data[11]
    M0 = data[12]
    e = data[14]
    w = data[23]
    Cuc = data[13]
    Cus = data[15]
    Crc = data[22]
    Crs = data[10]
    Cic = data[18]
    Cis = data[20]
    i0 = data[21]
    idot = data[25]
    Omega0 = data[19]
    Omega = data[24]
    aF0 = data[6]
    aF1 = data[7]
    aF2 = data[8]
    tow_all = week * 604800 + t
    toe_full = gps_week * 604800 + toe
    tk = tow_all - toe_full
    n0 = math.sqrt(U / a ** 3)
    n = n0 + dn
    Mk = M0 + n * tk
    previousE = Mk
    nextE = Mk + e * math.sin(previousE)
    while abs(nextE - previousE) >= 10 ** (-12):
        previousE = nextE
        nextE = Mk + e * math.sin(previousE)
    vk = math.atan2(math.sqrt(1 - e ** 2) * math.sin(nextE), math.cos(nextE) - e)
    if vk <= 0:
        vk = vk + 2 * math.pi
    FiK = vk + w
    duk = Cus * math.sin(2 * FiK) + Cuc * math.cos(2 * FiK)
    drk = Crs * math.sin(2 * FiK) + Crc * math.cos(2 * FiK)
    dik = Cis * math.sin(2 * FiK) + Cic * math.cos(2 * FiK)
    uk = FiK + duk
    rk = a * (1 - e * math.cos(nextE)) + drk
    ik = i0 + idot * tk + dik
    xk = rk * math.cos(uk)
    yk = rk * math.sin(uk)
    OmegaK = Omega0 + (Omega - wE) * tk - wE * toe
    Xk = xk * math.cos(OmegaK) - yk * math.cos(ik) * math.sin(OmegaK)
    Yk = xk * math.sin(OmegaK) + yk * math.cos(ik) * math.cos(OmegaK)
    Zk = yk * math.sin(ik)
    delta_trel = (-2 * math.sqrt(U) / c ** 2) * e * math.sqrt(a) * math.sin(nextE)
    dts = aF0 + aF1 * tk + aF2 * tk ** 2 + delta_trel
    return Xk, Yk, Zk, dts


def getMatX(A, y):
    matx = np.dot(np.dot(-np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), y)
    return matx


def getMatA(satellite, dist):
    matA = [(-(satellite[0] - RECV[0]) / dist)[0], (-(satellite[1] - RECV[1]) / dist)[0],
            (-(satellite[2] - RECV[2]) / dist)[0], 1]
    return matA


def getY(pseudo_dist, observation):
    ind = np.where(np.isnan(observation))
    observation[ind] = pseudo_dist[ind]
    return pseudo_dist - observation


def getPseudoDistance(satellite, dist):
    return dist - c * satellite[3] + c * RECV[3]


def getTime():
    timetab = []
    start_time = start_day[3]
    stop_time = day_stop[3]
    if stop_time == 0:
        stop_time = 24
    for hours in range(start_time, stop_time):
        for minutes in range(0, 60):
            for seconds in range(0, 59, dt):
                temp = datetime(start_day[0], start_day[1], start_day[2], hours, minutes, seconds)
                timetab.append(temp)
    return timetab


def getSatelliteFrame(nrSatellite, t):
    frame = nav[np.where(np.any(indnav == nrSatellite, axis=1))[0]]
    return np.append(frame[np.argmin(abs(t - frame[:, 17]))], [nrSatellite])


def getCorrectedCoordEpochSatellite(satellite_coord, T):
    rot = np.dot([[math.cos(wE * T), math.sin(wE * T), 0],
                  [-math.sin(wE * T), math.cos(wE * T), 0],
                  [0, 0, 1]], [[satellite_coord[0]], [satellite_coord[1]], [satellite_coord[2]]])
    rot = np.transpose(np.append(rot, [satellite_coord[3]]))
    return rot


def getGeometricalDistance(satellite):
    ro = math.sqrt((satellite[0] - RECV[0]) ** 2 + (satellite[1] - RECV[1]) ** 2 + (satellite[2] - RECV[2]) ** 2)
    return ro


def getDOP(A):
    Q = np.linalg.inv(np.dot(A.T, A))
    R = getNEU()
    QNEU = np.dot(np.dot(R.T, Q[0:3, 0:3]), R)
    GDOP = math.sqrt(Q.diagonal()[0] + Q.diagonal()[1] + Q.diagonal()[2] + Q.diagonal()[3])
    PDOP = math.sqrt(Q.diagonal()[0] + Q.diagonal()[1] + Q.diagonal()[2])
    TDOP = math.sqrt(Q.diagonal()[3])
    HDOP = math.sqrt(QNEU.diagonal()[0] + QNEU.diagonal()[1])
    VDOP = math.sqrt(QNEU.diagonal()[2])

    return GDOP, PDOP, TDOP, HDOP, VDOP


# TROPO


def getOrthometricHeight(hEl, N):
    return hEl - N


def getPressure(hEl, N):
    return 1013.25 * (1 - 0.0000226 * getOrthometricHeight(hEl, N))**5.225


def getTemperature(hEl, N):
    return 291.15 - 0.0065 * getOrthometricHeight(hEl, N)


def getHumidity(hEl, N):
    return 0.5 * math.exp(-0.0006396 * getOrthometricHeight(hEl, N))


def getSteamPressure(hEl, N):
    temp = getTemperature(hEl, N)
    return 6.11 * getHumidity(hEl, N) * 10 ** (7.5 * (temp - 273.15) / (temp - 35.85))


def getNd0(hEl, N):
    return 77.64 * getPressure(hEl, N) / getTemperature(hEl, N)


def getNw0(hEl, N):
    steamPressure = getSteamPressure(hEl, N)
    temperature = getTemperature(hEl, N)
    return -12.96 * steamPressure / temperature + (3.718 * (10 ** 5)) * steamPressure / (temperature ** 2)


def getN0(hEl, N):
    return getNd0(hEl, N) + getNw0(hEl, N)


def gethd(hEl, N):
    return 40136 + 148.72 * (getTemperature(hEl, N) - 273.15)  # wartosc w metrach


def gethw():
    return 11000


def getDeltaTd0(hEl, N):
    # print("DeltaTd0: ")
    # print(10 ** -6 * getNd0(hEl, N) * gethd(hEl, N) / 5)
    return 10 ** -6 * getNd0(hEl, N) * gethd(hEl, N) / 5


def getDeltaTw0(hEl, N):
    # print("DeltaTw0: ")
    # print(10 ** -6 * getNw0(hEl, N) * gethw() / 5)
    return 10 ** -6 * getNw0(hEl, N) * gethw() / 5


def getmd(El):
    return 1 / math.sin(math.sqrt(np.deg2rad(El ** 2 + 6.25)))


def getmw(El):
    return 1 / math.sin(math.sqrt(np.deg2rad(El ** 2 + 2.25)))


def getDeltaTd(hEl, N, El):
    return getmd(El) * getDeltaTd0(hEl, N)


def getDeltaTw(hEl, N, El):
    return getmw(El) * getDeltaTw0(hEl, N)


def getDeltaTHop(hEl, N, El):
    return getDeltaTw(hEl, N, El) + getDeltaTd(hEl, N, El)


def getDeltaTSas(hEl, N, El):
    return getmd(El) * 0.002277 * getPressure(hEl, N) \
           + getmw(El) * 0.002277 * (1255 / getTemperature(hEl, N) + 0.05) * getSteamPressure(hEl, N)


# JONO


def getsC(angle):
    return angle/180


def getRad(sem):
    return sem * math.pi


def getPsi(El):
    return 0.0137/(getsC(El) + 0.11) - 0.022


def getFiIpp(Fir, El, Az):
    fi_ipp = getsC(Fir) + getPsi(El) * math.cos(np.deg2rad(Az))
    if fi_ipp > 0.416666:
        fi_ipp = 0.416666
    elif fi_ipp < -0.416666:
        fi_ipp = -0.416666
    return fi_ipp


def getLamIpp(Lamr, Fir, El, Az):
    return getsC(Lamr) + getPsi(El) * math.sin(np.deg2rad(Az)) / math.cos(getFiIpp(Fir, El, Az) * np.pi)


def getFim(Lamr, Fir, El, Az):
    return getFiIpp(Fir, El, Az) + 0.064 * math.cos((getLamIpp(Lamr, Fir, El, Az) - 1.617) * np.pi)


def getSecTime(Lamr, Fir, El, Az, Tow):
    t = 43200 * getLamIpp(Lamr, Fir, El, Az) + Tow
    t = math.fmod(t, 86400)
    if t >= 86400:
        t -= 86400
    elif t < 0:
        t += 86400
    return t


def getAIon(Lamr, Fir, El, Az, Alfa):
    Fim = getFim(Lamr, Fir, El, Az)
    Aion = Alfa[0] + Alfa[1] * Fim + Alfa[2] * Fim ** 2 + Alfa[3] * Fim ** 3
    if Aion < 0:
        Aion = 0
    return Aion


def getPIon(Lamr, Fir, El, Az, Beta):
    Fim = getFim(Lamr, Fir, El, Az)
    Pion = Beta[0] + Beta[1] * Fim + Beta[2] * Fim ** 2 + Beta[3] * Fim ** 3
    if Pion < 7200:
        Pion = 7200
    return Pion


def getFIon(Lamr, Fir, El, Az, Tow, Beta):
    return 2 * np.pi * (getSecTime(Lamr, Fir, El, Az, Tow) - 50400) / getPIon(Lamr, Fir, El, Az, Beta)


def getmf(El):
    return 1 + 16 * (0.53 - getsC(El)) ** 3


def getDeltaIon(Lamr, Fir, El, Az, Tow, Beta, Alfa):
    FIon = getFIon(Lamr, Fir, El, Az, Tow, Beta)
    deltaIon = c * getmf(El) * (5 * 10**(-9) + getAIon(Lamr, Fir, El, Az, Alfa) * (1 - FIon ** 2 / 2 + FIon ** 4 / 24))
    if abs(FIon) > np.pi / 2:
        deltaIon = c * getmf(El) * (5 * 10**(-9))
    return deltaIon

#####


def getRecvCoordtest():
    recv_coord = np.zeros([0, 4])
    dop = np.zeros([0, 5])
    for time in range(TOW_start, TOW_stop, dt):
        recv = np.zeros([0, 4])
        dist = np.zeros([0, 1])
        A = np.zeros([0, 4])
        T = 0.072
        idx = iobs[:, 2] == time
        iobss = iobs[idx, :]
        obss = obs[idx]
        for j in range(4):
            XYZ, correctedXYZ, geometricalDist, pseudoDist, y, matA, elevation, azimuth, = \
                np.zeros([0, 4]), np.zeros([0, 4]), np.zeros([0, 1]), np.zeros([0, 1]), np.zeros([0, 1]), \
                np.zeros([0, 4]), np.zeros([0, 1]), np.zeros([0, 1])
            for i, nrSat in enumerate(iobss):
                if j > 0:
                    T = dist[i, 0] / c
                data = getSatelliteFrame(nrSat, time)
                tobs = iobss[i, 2]
                tr = tobs + RECV[3] - T
                XYZ = np.vstack([XYZ, getCoordSatellite(tr, data, week_start)])
                correctedXYZ = np.vstack([correctedXYZ, getCorrectedCoordEpochSatellite(XYZ[i], T)])
                geometricalDist = np.vstack([geometricalDist, getGeometricalDistance(correctedXYZ[i])])
                pseudoDist = np.vstack([pseudoDist, getPseudoDistance(correctedXYZ[i], geometricalDist[i])])
                y = np.vstack([y, getY(pseudoDist[i], obss[i])])
                matA = np.vstack([matA, getMatA(correctedXYZ[i], geometricalDist[i])])
            matX = getMatX(matA, y)
            RECV[0] += matX[0, 0]
            RECV[1] += matX[1, 0]
            RECV[2] += matX[2, 0]
            RECV[3] += matX[3, 0] / c
            recv = np.vstack([recv, RECV])
            A = matA
            last = recv[-1]
            dist = geometricalDist
        dop = np.vstack([dop, getDOP(A)])
        recv_coord = np.vstack([recv_coord, last])
    diff = recv_coord[:, 0:-1] - WROC
    return diff, dop


def plotDiff(diff):
    time = getTime()
    plt.figure(figsize=(15, 5))
    plt.title(f'Różnice współrzędnych dla dnia {dayGPS.strftime("%d-%m-%Y")}')
    plt.plot(time, diff)
    legendDiff = ['X', 'Y', 'Z']
    plt.legend(legendDiff)
    plt.xlabel("Czas [godz.]")
    plt.ylabel("Różnica [m]")
    plt.show()


def plotDiff3D(diff):
    time = getTime()
    plt.figure(figsize=(15, 5))
    diff3d = np.array(np.sqrt(np.sum(np.square(diff), axis=1)))
    plt.title(f'Różnica 3D dla dnia {dayGPS.strftime("%d-%m-%Y")}')
    plt.plot(time, diff3d)
    plt.xlabel("Czas [godz.]")
    plt.ylabel("Różnica [m]")
    plt.show()


def plotDOP(dop):
    time = getTime()
    plt.figure(figsize=(15, 5))
    plt.title(f'Współczynniki DOP dla dnia {dayGPS.strftime("%d-%m-%Y")}')
    plt.plot(time, dop)
    legendDOP = ['GDOP', 'PDOP', 'TDOP', 'HDOP', 'VDOP']
    plt.legend(legendDOP)
    plt.xlabel("Czas [godz.]")
    plt.ylabel("Wartość współczynnika DOP")
    plt.show()


if __name__ == "__main__":
    start_day = [2021, 3, 1, 0, 0, 0]
    day_stop = [2021, 3, 2, 0, 0, 0]
    dt = 30
    week_start, TOW_start, dow_start = date2tow(start_day)
    week_stop, TOW_stop, dow_stop = date2tow(day_stop)
    nav_file = 'WROC00POL_R_20210600000_01D_GN.rnx'
    obs_file = 'WROC00POL_R_20210600000_01D_30S_MO.rnx'
    nav, indnav = readrnxnav(nav_file)
    obs, iobs, RECV = readrnxobs(obs_file, start_day, day_stop, 'G')
    RECV.append(0)
    WROC = [3835751.6257, 1177249.7445, 4941605.0540]
    alfa = [9.3132E-09, 0.0000E+00, -5.9605E-08, 0.0000E+00]  # wartosci z 3 linijki pliku GN.rnx
    beta = [9.0112E+04, 0.0000E+00, -1.9661E+05, 0.0000E+00]  # wartosci z 4 linijki pliku GN.rnx
    fir = 52
    lamr = 21
    el = 60
    az = 180
    hel = 180.818
    n = 40.231
    print('Poprawka ionosferyczna: ' + str(getDeltaIon(lamr, fir, el, az, TOW_start, beta, alfa)))
    print('Poprawka troposferyczna (Hopfield): ' + str(getDeltaTHop(hel, n, el)))
    print('Poprawka troposferyczna (Saastamoinen): ' + str(getDeltaTSas(hel, n, el)))
    dayGPS = date(start_day[0], start_day[1], start_day[2])
    diff, dop = getRecvCoordtest()
    plotDiff(diff)
    plotDiff3D(diff)
    plotDOP(dop)
