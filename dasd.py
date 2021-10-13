import math
import numpy
fi = 0
lamb = 0
h = 0
a = 6378137
e2 = 0.00669438002290

N = a / math.sqrt(1 - e2 * math.sin(fi) ** 2)

Xo = (N + h) * math.cos(fi) * math.cos(lamb)
Yo = (N + h) * math.cos(fi) * math.sin(lamb)
Zo = (N * (1 - e2) + h) * math.sin(fi)

XS = numpy.array([[42172306 - Xo], [0 - Yo], [0 - Zo]])
neu = numpy.array([[-math.sin(fi) * math.cos(lamb), -math.sin(lamb), math.cos(fi) * math.cos(lamb)],
                   [-math.sin(fi) * math.sin(lamb), math.cos(lamb), math.cos(fi) * math.sin(lamb)],
                   [math.cos(fi), 0, math.sin(fi)]])
XS_NEU = numpy.dot(neu.T, XS)
Az = numpy.rad2deg(math.atan2(XS_NEU[1], XS_NEU[0]))

if Az < 0:
    Az = Az + 360
el = numpy.rad2deg(math.asin((XS_NEU[2]) / math.sqrt(XS_NEU[0] * 2 + XS_NEU[1] * 2 + XS_NEU[2] ** 2)))
print(el)