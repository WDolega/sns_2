import numpy as np
import math

from CalcTime import CalcTime, calcTimeOfRinex
from Constant import Constant
from Satellite import Satellite
from GPS import GPS
from hirvonen import hirvonen
from klobsns import iono
from sastatopo import topo


class CalcRecvCoord:
    RECV = GPS()
    obs, iobs = None, None
    mask = 0

    def getRecvCoord(self, iono_corr=True, tropo_corr=True, mask=10):
        self.mask = mask
        recvCoord_perTime = np.zeros([0, 4])
        dop = np.zeros([0, 5])
        start = calcTimeOfRinex(CalcTime.start_day)
        stop = calcTimeOfRinex(CalcTime.stop_day)
        for next_time in range(start, stop, CalcTime.interval):
            recv = np.zeros([0, 4])
            distance = np.zeros([0, 1])
            A = np.zeros([0, 4])
            Satellite.set_def_prop_signal()
            Satellite.set_calc_time(next_time)
            epoch, elevation, azimuth = self.getEpochSatellite(next_time)
            if iono_corr:
                corrIono = [iono(next_time, elevation[i], azimuth[i]) for i in range(len(elevation))]
            else:
                corrIono = [0 for _i in range(len(elevation))]
            if tropo_corr:
                corrTropo = [topo(elevation[i]) for i in range(len(elevation))]
            else:
                corrTropo = [0 for _i in range(len(elevation))]
            observation = self.obs[(np.isin(self.iobs[:, 0], epoch[:, -1])) & (self.iobs[:, 2] == next_time)]
            for iteration in range(5):
                geometricalDist, y, matA = np.zeros([0, 1]), np.zeros([0, 1]), np.zeros([0, 4])
                for i, satellite in enumerate(epoch):
                    if iteration > 0:
                        Satellite.prop_signal = distance[i, 0] / Constant.c
                    XYZ = getEpochSatelliteCoord(satellite)
                    correctedXYZ = getCorrectedCoordEpochSatellite(XYZ)
                    geometricalDist = np.vstack([geometricalDist, self.getGeometricalDistance(correctedXYZ)])
                    pseudoDist = self.getPseudoDistance(correctedXYZ, geometricalDist[i], corrIono[i], corrTropo[i])
                    y = np.vstack([y, get_matY(pseudoDist, observation[i])])
                    matA = np.vstack([matA, self.get_matA(correctedXYZ, geometricalDist[i])])
                matX = get_matX(matA, y)
                self.RECV.setCoordGPS(matX)
                recv = np.vstack([recv, self.RECV.getArrayParameter()])
                distance = geometricalDist
                A = matA
            Q = get_Q(A)
            dop = np.vstack([dop, [get_GDOP(Q), get_PDOP(Q), get_TDOP(Q), get_HDOP(Q), get_VDOP(Q)]])
            recvCoord_perTime = np.vstack([recvCoord_perTime, recv[-1]])
        return recvCoord_perTime, dop

    def getEpochSatellite(self, start_time):
        epoch = np.zeros([0, 38])
        elevation = []
        azimuth = []
        for nrSat in self.getTimeInterval(start_time)[:, 0]:
            sat = Satellite(nrSat)
            elev = self.getElevation(sat.getCoordSatellite())
            if elev > self.mask:
                elevation.append(elev)
                azimuth.append(self.getAzimuth(sat.getCoordSatellite()))
                epoch = np.vstack([epoch, sat.getEpochSatelliteParameter()])
        return epoch, np.array(elevation), np.array(azimuth)

    def getTimeInterval(self, start_time):
        return self.iobs[self.iobs[:, 2] == start_time]

    def getGeometricalDistance(self, satellite):
        ro = math.sqrt(
            (satellite[0] - self.RECV.get_x()) ** 2 + (satellite[1] - self.RECV.get_y()) ** 2 + (
                    satellite[2] - self.RECV.get_z()) ** 2)
        return ro

    def getPseudoDistance(self, satellite, dist, corr_iono, corr_topo):
        pseudoDist = dist - Constant.c * satellite[3] + Constant.c * self.RECV.delta_tr + corr_iono + corr_topo
        return pseudoDist

    def get_matA(self, satellite, dist):
        matA = [(-(satellite[0] - self.RECV.get_x()) / dist)[0], (-(satellite[1] - self.RECV.get_y()) / dist)[0],
                (-(satellite[2] - self.RECV.get_z()) / dist)[0], 1]
        return matA

    def getElevation(self, satellite):
        NEU = self.getVectorNEU(satellite)
        return np.rad2deg(math.asin(NEU[2] / (math.sqrt(NEU[0] ** 2 + NEU[1] ** 2 + NEU[2] ** 2))))

    def getAzimuth(self, satellite):
        NEU = self.getVectorNEU(satellite)
        return np.rad2deg(math.atan2(NEU[1], NEU[0])) if np.rad2deg(
            math.atan2(NEU[1], NEU[0])) > 0 else np.rad2deg(
            math.atan2(NEU[1], NEU[0])) + 360

    def getVectorSatelliteGPS(self, satellite):
        geocentric_XYZ = self.RECV.getArrayCoord()
        coordSatellite = np.array([satellite[0], satellite[1], satellite[2]])
        return np.transpose(np.array(coordSatellite - geocentric_XYZ))

    def getVectorNEU(self, satellite):
        return np.dot(np.transpose(self.getNEU()), self.getVectorSatelliteGPS(satellite))

    def getNEU(self):
        fi, lamb, h = hirvonen(self.RECV.get_x(), self.RECV.get_y(), self.RECV.get_z())
        return np.array([[-math.sin(fi) * math.cos(lamb), -math.sin(lamb), math.cos(fi) * math.cos(lamb)],
                         [-math.sin(fi) * math.sin(lamb), math.cos(lamb), math.cos(fi) * math.sin(lamb)],
                         [math.cos(fi), 0, math.sin(fi)]])


def getEpochSatelliteCoord(satellite):
    nr_satellite = satellite[len(satellite) - 1]
    return Satellite(nr_satellite).getCoordSatellite()


def getCorrectedCoordEpochSatellite(satellite_coord):
    wE = Constant.wE
    rot = np.dot([[math.cos(wE * Satellite.prop_signal), math.sin(wE * Satellite.prop_signal), 0],
                  [-math.sin(wE * Satellite.prop_signal), math.cos(wE * Satellite.prop_signal), 0],
                  [0, 0, 1]], [[satellite_coord[0]], [satellite_coord[1]], [satellite_coord[2]]])
    rot = np.transpose(np.append(rot, [satellite_coord[3]]))
    return rot


def get_matY(pseudo_dist, observation):
    ind = np.where(np.isnan(observation))
    observation[ind] = pseudo_dist[ind]
    return pseudo_dist - observation


def get_matX(A, y):
    return np.dot(np.dot(-np.linalg.inv(np.dot(np.transpose(A), A)), np.transpose(A)), y)


def get_Q(A):
    return np.linalg.inv(np.dot(A.T, A))


def get_GDOP(Q):
    return math.sqrt(Q.diagonal()[0] + Q.diagonal()[1] + Q.diagonal()[2] + Q.diagonal()[3])


def get_PDOP(Q):
    return math.sqrt(Q.diagonal()[0] + Q.diagonal()[1] + Q.diagonal()[2])


def get_TDOP(Q):
    return math.sqrt(Q.diagonal()[3])


def get_Qneu(Q):
    fi, lam, h = hirvonen(GPS.x, GPS.y, GPS.z)
    R = np.array([[-math.sin(fi) * math.cos(lam), -math.sin(lam), math.cos(fi) * math.cos(lam)],
                         [-math.sin(fi) * math.sin(lam), math.cos(lam), math.cos(fi) * math.sin(lam)],
                         [math.cos(fi), 0, math.sin(fi)]])
    return np.dot(np.dot(R.T, Q[0:3, 0:3]), R)


def get_HDOP(Q):
    Qneu = get_Qneu(Q)
    return math.sqrt(Qneu.diagonal()[0] + Qneu.diagonal()[1])


def get_VDOP(Q):
    Qneu = get_Qneu(Q)
    return math.sqrt(Qneu.diagonal()[2])