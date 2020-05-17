import numpy as np
import math
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# Datos para el modelo: h cant horas, w = resultados

#hVec = [1, 2, 4, 3.5, 10, 6, 5, 11, 3, 2]
#wVec = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0]

hVec = [1, 10, 50, 50, 40, 30, 90, 70, 80, 100]
wVec = [0, 0, 0, 1, 0, 1, 1, 1, 1, 0]

numDataPoints = len(wVec)

betaVecInit = (0, 0)

def func(betaVec):
    beta0 = betaVec[0]
    beta1 = betaVec[1]
    res = 1
    for i in range(numDataPoints):
        if(wVec[i]==1):
            res = res * (1/(1 + math.exp(-(beta0 + hVec[i]*beta1))))
        else:
            res = res * (1 - (1/(1 + math.exp(-(beta0 + hVec[i]*beta1)))))
    return -res

def funcOpt(betaOpt0, betaOpt1, hval):
    return 1/(1 + math.exp(-(betaOpt0 + hval*betaOpt1)))
    

min = minimize(func,betaVecInit)

betaOpt0 = min.x[0]
betaOpt1 = min.x[1]
print(min.x)
xvals = np.arange(-50, 150, 0.1)
yvals = np.zeros(len(xvals))
for i in range(len(xvals)):
    yvals[i] = funcOpt(betaOpt0, betaOpt1, xvals[i])

plt.plot(xvals,yvals)
plt.show()