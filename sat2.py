import math
import numpy as np

from Constant import Constant
from Parameter import Parameter
from GPS import GPS


class Satellite:
    prop_signal = 0.072
    calc_time = None
    epochSatelliteParameter = None
    parameter = Parameter()
    recv = GPS()
    nav, inav = None, None

    def __init__(self, nrSatellite):
        self.nrSatellite = nrSatellite

    @staticmethod
    def set_def_prop_signal():
        Satellite.prop_signal = 0.072

    @staticmethod
    def set_calc_time(time):
        Satellite.calc_time = time

    def getSatelliteNav(self):
        return self.nav[np.where(np.any(self.inav == self.nrSatellite, axis=1))[0]]

    def getEpochSatelliteParameter(self):
        t = self.calc_time
        satelliteNav = self.getSatelliteNav()
        return np.append(satelliteNav[np.argmin(abs(t - satelliteNav[:, 17]))], [self.nrSatellite])

    def setParameter(self):
        self.parameter.tk = self.get_tk()
        self.parameter.N = self.getN()
        self.parameter.Mk = self.getMk()
        self.parameter.Ek = self.getEk()
        self.parameter.Vk = self.getVk()
        self.parameter.FiK = self.getFiK()
        self.parameter.deltaUk = self.get_deltaUk()
        self.parameter.deltaIk = self.get_deltaIk()
        self.parameter.deltaRk = self.get_deltaRk()
        self.parameter.Uk = self.getUk()
        self.parameter.Ik = self.getIk()
        self.parameter.Rk = self.getRk()
        self.parameter.xk, self.parameter.yk = self.getPositionSatellite()
        self.parameter.omegaK = self.getOmegaK()

    def get_tk(self):
        t = self.calc_time + self.recv.delta_tr - self.prop_signal
        return t - self.epochSatelliteParameter[17]

    def getN(self):
        a = self.epochSatelliteParameter[16] ** 2
        return math.sqrt(Constant.U / a ** 3) + self.epochSatelliteParameter[11]

    def getMk(self):
        M0 = self.epochSatelliteParameter[12]
        tk = self.parameter.tk
        n = self.parameter.N
        Mk = M0 + n * tk
        return Mk

    def getEk(self):
        e = self.epochSatelliteParameter[14]
        Mk = self.parameter.Mk
        previousE = Mk
        nextE = Mk + e * math.sin(previousE)
        while abs((previousE - nextE)) > 10 ** (-15):
            previousE = nextE
            nextE = Mk + e * math.sin(previousE)
        return nextE

    def getVk(self):
        e = self.epochSatelliteParameter[14]
        Ek = self.parameter.Ek
        vk = math.atan2(math.sqrt(1 - e ** 2) * math.sin(Ek), math.cos(Ek) - e)
        return vk if vk > 0 else vk + 2 * math.pi

    def getFiK(self):
        vk = self.parameter.Vk
        w = self.epochSatelliteParameter[23]
        return vk + w

    def get_deltaUk(self):
        FiK = self.parameter.FiK
        Cuc = self.epochSatelliteParameter[13]
        Cus = self.epochSatelliteParameter[15]
        return Cus * math.sin(2 * FiK) + Cuc * math.cos(2 * FiK)

    def get_deltaRk(self):
        FiK = self.parameter.FiK
        Crs = self.epochSatelliteParameter[10]
        Crc = self.epochSatelliteParameter[22]
        return Crs * math.sin(2 * FiK) + Crc * math.cos(2 * FiK)

    def get_deltaIk(self):
        fiK = self.parameter.FiK
        Cic = self.epochSatelliteParameter[18]
        Cis = self.epochSatelliteParameter[20]
        return Cis * math.sin(2 * fiK) + Cic * math.cos(2 * fiK)

    def getUk(self):
        return self.parameter.FiK + self.parameter.deltaUk

    def getRk(self):
        e = self.epochSatelliteParameter[14]
        a = self.epochSatelliteParameter[16] ** 2
        return a * (1 - e * math.cos(self.parameter.Ek)) + self.parameter.deltaRk

    def getPositionSatellite(self):
        rK = self.parameter.Rk
        uK = self.parameter.Uk
        xk = rK * math.cos(uK)
        yk = rK * math.sin(uK)
        if abs(rK - math.sqrt(xk ** 2 + yk ** 2)) > 0.01:
            return False
        return xk, yk

    def getIk(self):
        i0 = self.epochSatelliteParameter[21]
        idot = self.epochSatelliteParameter[25]
        return i0 + idot * self.parameter.tk + self.parameter.deltaIk

    def getOmegaK(self):
        omega0 = self.epochSatelliteParameter[19]
        omegaDot = self.epochSatelliteParameter[24]
        toe = self.epochSatelliteParameter[17]
        return omega0 + (omegaDot - Constant.wE) * self.parameter.tk - Constant.wE * toe

    def getDeltaTime(self):
        aF0 = self.epochSatelliteParameter[6]
        aF1 = self.epochSatelliteParameter[7]
        aF2 = self.epochSatelliteParameter[8]
        e = self.epochSatelliteParameter[14]
        sqrt_a = self.epochSatelliteParameter[16]
        Ek = self.parameter.Ek
        delta_trel = (-2 * math.sqrt(Constant.U) / Constant.c ** 2) * e * sqrt_a * math.sin(Ek)
        return aF0 + aF1 * self.parameter.tk + aF2 * (self.parameter.tk ** 2) + delta_trel

    def getCoordSatellite(self):
        Satellite.epochSatelliteParameter = self.getEpochSatelliteParameter()
        self.setParameter()
        iK = self.parameter.Ik
        xk, yk = self.parameter.xk, self.parameter.yk
        omegaK = self.parameter.omegaK
        Xk = xk * math.cos(omegaK) - yk * math.cos(iK) * math.sin(omegaK)
        Yk = xk * math.sin(omegaK) + yk * math.cos(iK) * math.cos(omegaK)
        Zk = yk * math.sin(iK)
        if abs(self.parameter.Rk - np.linalg.norm([Xk, Yk, Zk])) > 0.01:
            return False
        return np.array([Xk, Yk, Zk, self.getDeltaTime()])

    def getRecvCoordtest():
        recv = np.zeros([0, 4])
        dist = np.zeros([0, 1])
        dop = np.zeros([0, 5])
        for j in range(6):
            XYZ, correctedXYZ, geometricalDist, pseudoDist, y, matA = np.zeros([0, 4]), np.zeros([0, 4]), np.zeros(
                [0, 1]), \
                                                                      np.zeros([0, 1]), np.zeros([0, 1]), np.zeros(
                [0, 4])
            for i, satellite in enumerate(getEpochSatellite()):
                if j > 0:
                    Satellite.prop_signal = dist[i, 0] / c
                XYZ = np.vstack([XYZ, getEpochSatelliteCoord(satellite)])
                correctedXYZ = np.vstack([correctedXYZ, getCorrectedCoordEpochSatellite(XYZ[i])])
                geometricalDist = np.vstack([geometricalDist, getGeometricalDistance(correctedXYZ[i])])
                pseudoDist = np.vstack([pseudoDist, getPseudoDistance(correctedXYZ[i], geometricalDist[i])])
                y = np.vstack([y, getY(pseudoDist[i], obs[i])])
                matA = np.vstack([matA, getMatA(correctedXYZ[i], geometricalDist[i])])
            matX = getMatX(matA, y)
            RECV[0] += matX[0, 0]
            RECV[1] += matX[1, 0]
            RECV[2] += matX[2, 0]
            RECV[3] += matX[3, 0] / c
            recv = np.vstack([recv, RECV])
            dist = geometricalDist
        return recv, dist