import numpy as np
import math
from matplotlib import pyplot as plt

# Datos para el modelo: h cant horas, w = resultados

#hVals = [1, 2, 4, 3.5, 10, 6, 5, 11, 3, 2]
#yVals = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0]

hVals = np.array([1, 10, 50, 50, 40, 30, 90, 70, 80, 100])
yVals = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 0])
d = 2 # dimensionalidad del problema

n = len(yVals) #cant de datos

wVecInit = np.array((-10, 0.2))

def alfaVal(wVec, xVec):
    return 1/(1 + math.exp(-np.dot(wVec,xVec)))
    
def getxVec(hVal):
    xList = [1]
    xList.append(hVal)
    return np.array(xList)

def getAlfaVec(wVecInit,hVals):
    alfaVec = []
    for i in range(n):
        alfaVec.append(alfaVal(wVecInit,getxVec(hVals[i])))
    return np.array(alfaVec)

def getBMatrix(alfaVec):
    vecDiag = []
    for alfa in alfaVec:
        auxDiag = alfa * (1 - alfa)
        vecDiag.append(auxDiag)
    return np.diag(vecDiag)

B = getBMatrix(getAlfaVec(wVecInit,hVals))
print(B[1,1])
        
