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

wVecInit = np.matrix(np.array([0, 0])).reshape(-1,1)


def alfaVal(wVec, xVec):
    return 1/(1 + math.exp(-(wVec.T*xVec).item()))

def getxVec(hVal):
    xList = [1]
    xList.append(hVal)
    return np.matrix(xList).reshape(-1,1)

def getAlfaVec(wVecInit,hVals):
    alfaVec = []
    for i in range(n):
        alfaVec.append(alfaVal(wVecInit,getxVec(hVals[i])))
    return np.matrix(alfaVec).reshape(-1,1)

def getBMatrix(alfaVec):
    vecDiag = []
    for alfa in alfaVec:
        auxDiag = alfa * (1 - alfa)
        vecDiag.append(auxDiag.item())
    return np.matrix(np.diag(vecDiag))

def getAMatrix(hVals):
    listX = []
    for h in hVals:
        xVec = getxVec(h).T
        listX.append(xVec.tolist()[0])
    return np.matrix(listX)

def getNextwVec(wVec,A,B,alfaVec,yVec):
    return wVec - ( np.linalg.inv(A.T * B * A) * (A.T) * (alfaVec - yVec))

A = getAMatrix(hVals)
yVec = np.matrix(yVals).reshape(-1, 1)

def getOptwVec(A, yVec, hVals, wVecInit):
    wVec = wVecInit
    for i in range(10):        
        alfaVec = getAlfaVec(wVec,hVals)
        B = getBMatrix(alfaVec)
        wVec = getNextwVec(wVec,A,B,alfaVec,yVec)
    return (i,wVec)


wVecOpt = getOptwVec(A,yVec,hVals,wVecInit)

print(wVecOpt[1])





