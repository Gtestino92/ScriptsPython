import numpy as np
import math
from matplotlib import pyplot as plt

# Datos para el modelo: h cant horas, w = resultados

hVals = np.array([1, 10, 20, 30, 40, 50, 50, 40, 30, 90, 70, 80, 100])
yVals = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
d = 2 # dimensionalidad del problema

n = len(yVals) #cant de datos

wVecInit = np.matrix(np.array([0, 0])).reshape(-1,1)


def alfaVal(wVec, xVec):
    return 1/(1 + math.exp(-(wVec.T*xVec).item()))

def sigmoid(betaOpt0, betaOpt1, hval):
    return 1/(1 + math.exp(-(betaOpt0 + hval*betaOpt1)))
 
def getxVec(hVal):
    xList = [1]
    xList.append(hVal)
    return np.matrix(xList).reshape(-1,1)

def getAlfaVec(wVec,hVals):
    alfaVec = []
    for i in range(n):
        alfaVec.append(alfaVal(wVec,getxVec(hVals[i])))
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
    alfaVec = getAlfaVec(wVec,hVals)
    B = getBMatrix(alfaVec)
    for i in range(10):     
        likelihoodOld = calcLikelihood(alfaVec, yVec)
        wVec = getNextwVec(wVec,A,B,alfaVec,yVec)
        alfaVec = getAlfaVec(wVec,hVals)
        B = getBMatrix(alfaVec)
        likelihood = calcLikelihood(alfaVec,yVec)
        dist = np.absolute((likelihood-likelihoodOld)*100/likelihoodOld)
        if(dist<0.1): 
            break
    return wVec

def calcLikelihood(alfaVec, yVec):
    likelihood = 1
    for i in range(len(alfaVec)):
        alfa = alfaVec[i].item()
        y = yVec[i].item()
        likelihood *= math.pow(alfa,y) * math.pow((1-alfa), (1-y))
    return likelihood


wVecOpt = getOptwVec(A,yVec,hVals,wVecInit)

wVecOpt0 = wVecOpt[0].item()
wVecOpt1 = wVecOpt[1].item()

xvals = np.arange(-100, 200, 0.1)
yvals = np.zeros(len(xvals))
for i in range(len(xvals)):
    yvals[i] = sigmoid(wVecOpt0, wVecOpt1, xvals[i])

plt.plot(xvals,yvals)
plt.show()