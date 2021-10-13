import math
from datetime import *
import numpy as np
from readrnx_studenci import nav, indnav, iobs, start_day, RECV, c, U, wE


class Satellite:
    prop_signal = 0.072
    calcTime = None

    def __init__(self, nrSatellite):
        self.nrSatellite = nrSatellite

    @staticmethod
    def setPropSignal():
        Satellite.propSignal = 0.072

    @staticmethod
    def setCalcTime(Time):
        Satellite.calcTime = Time

    def getSatelliteFrame(self):
        return nav[np.where(np.any(indnav == self.nrSatellite, axis=1))[0]]

    def getEpochSatelliteParameter(self):
        t = calcTimeOfRinex(start_day)
        return np.append(self.getSatelliteFrame()[np.argmin(abs(t - self.getSatelliteFrame()[:, 17]))],
                         [self.nrSatellite])

    def get_tk(self):
        t = iobs[np.where(np.any(iobs[:, 0] == self.nrSatellite))][0][2] + RECV[3] - Satellite.prop_signal
        return t - self.getEpochSatelliteParameter()[17]

    def getN(self):
        a = self.getEpochSatelliteParameter()[16] ** 2
        return math.sqrt(U / a ** 3) + self.getEpochSatelliteParameter()[11]

    def getMk(self):
        M0 = self.getEpochSatelliteParameter()[12]
        tk = self.get_tk()
        n = self.getN()
        Mk = M0 + n * tk
        return Mk

    def getEk(self):
        e = self.getEpochSatelliteParameter()[14]
        Mk = self.getMk()
        previousE = Mk
        nextE = Mk + e * math.sin(previousE)
        while abs((previousE - nextE)) > 10 ** (-15):
            previousE = nextE
            nextE = Mk + e * math.sin(previousE)
        return nextE

    def getVk(self):
        e = self.getEpochSatelliteParameter()[14]
        Ek = self.getEk()
        vk = math.atan2(math.sqrt(1 - e ** 2) * math.sin(Ek), math.cos(Ek) - e)
        return vk if vk > 0 else vk + 2 * math.pi

    def getFiK(self):
        vk = self.getVk()
        w = self.getEpochSatelliteParameter()[23]
        return vk + w

    def get_deltaUk(self):
        fiK = self.getFiK()
        Cuc = self.getEpochSatelliteParameter()[13]
        Cus = self.getEpochSatelliteParameter()[15]
        return Cus * math.sin(2 * fiK) + Cuc * math.cos(2 * fiK)

    def get_deltaRk(self):
        fiK = self.getFiK()
        Crs = self.getEpochSatelliteParameter()[10]
        Crc = self.getEpochSatelliteParameter()[22]
        return Crs * math.sin(2 * fiK) + Crc * math.cos(2 * fiK)

    def get_deltaIk(self):
        fiK = self.getFiK()
        Cic = self.getEpochSatelliteParameter()[18]
        Cis = self.getEpochSatelliteParameter()[20]
        return Cis * math.sin(2 * fiK) + Cic * math.cos(2 * fiK)

    def getUk(self):
        return self.getFiK() + self.get_deltaUk()

    def getRk(self):
        e = self.getEpochSatelliteParameter()[14]
        a = self.getEpochSatelliteParameter()[16] ** 2
        return a * (1 - e * math.cos(self.getEk())) + self.get_deltaRk()

    def getIk(self):
        i0 = self.getEpochSatelliteParameter()[21]
        idot = self.getEpochSatelliteParameter()[25]
        return i0 + idot * self.get_tk() + self.get_deltaIk()

    def getPositionSatellite(self):
        rK = self.getRk()
        uK = self.getUk()
        xk = rK * math.cos(uK)
        yk = rK * math.sin(uK)
        if abs(rK - math.sqrt(xk ** 2 + yk ** 2)) > 0.01:
            return False
        return xk, yk

    def getOmegaK(self):
        wE = 7.2921151467 * 10 ** (-5)
        omega0 = self.getEpochSatelliteParameter()[19]
        omegaDot = self.getEpochSatelliteParameter()[24]
        toe = self.getEpochSatelliteParameter()[17]
        return omega0 + (omegaDot - wE) * self.get_tk() - wE * toe

    def getCoordSatellite(self):
        iK = self.getIk()
        xk, yk = self.getPositionSatellite()
        omegaK = self.getOmegaK()
        Xk = xk * math.cos(omegaK) - yk * math.cos(iK) * math.sin(omegaK)
        Yk = xk * math.sin(omegaK) + yk * math.cos(iK) * math.cos(omegaK)
        Zk = yk * math.sin(iK)
        if abs(self.getRk() - np.linalg.norm([Xk, Yk, Zk])) > 0.01:
            return False
        return np.array([Xk, Yk, Zk, self.getDeltaTime()])

    def getDeltaTime(self):
        aF0 = self.getEpochSatelliteParameter()[6]
        aF1 = self.getEpochSatelliteParameter()[7]
        aF2 = self.getEpochSatelliteParameter()[8]
        e = self.getEpochSatelliteParameter()[14]
        sqrt_a = self.getEpochSatelliteParameter()[16]
        Ek = self.getEk()
        delta_trel = (-2 * math.sqrt(U) / c ** 2) * e * sqrt_a * math.sin(Ek)
        return aF0 + aF1 * (self.get_tk()) + aF2 * ((self.get_tk()) ** 2) + delta_trel


def setVariableFromRNX(SatelliteRow):
    return int(SatelliteRow[0]), int(SatelliteRow[1]), \
           int(SatelliteRow[2]), int(SatelliteRow[3]), \
           int(SatelliteRow[4]), int(SatelliteRow[5])


def calcTimeOfRinex(epoch):
    year, month, day, hour, minute, second = setVariableFromRNX(epoch)
    daysFromStartGPStoDate = date.toordinal(date(year, month, day)) - date.toordinal(date(1980, 1, 6))
    numberOfDayInWeek = daysFromStartGPStoDate % 7
    towParameter = numberOfDayInWeek * 86400 + hour * 3600 + minute * 60 + second
    return towParameter