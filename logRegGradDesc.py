import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import cm

MAX_ITER = 100
MIN_DIST = 0.1
GRAF_STEP = 1
d = 3 # dimensionalidad del problema

hVals = np.transpose(np.matrix([[1, 100, 20, 30, 40, 50, 50, 40, 30, 90, 40, 30, 100],
                [10, 40, 30, 50, 30, 40, 60, 70, 60, 90, 30, 20, 100]]))

yVec = np.matrix(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1])).reshape(-1,1)

n = len(yVec) #cant de datos

wVecInit = np.matrix(np.zeros(d)).reshape(-1,1)

def alfaVal(wVec, xVec):
    return 1/(1 + math.exp(-(wVec.T*xVec).item()))

def sigmoid(betaOpt0, betaOpt1, hval):
    return 1/(1 + math.exp(-(betaOpt0 + hval*betaOpt1)))
 
def getxVec(hVals):
    xList = [1]
    for hVal in hVals:
        xList.append(hVal)
    return np.matrix(xList).reshape(-1,1)

def getAlfaVec(wVec,hVals):
    alfaVec = []
    for i in range(n):
        alfaVec.append(alfaVal(wVec,getxVec(np.ravel(hVals[i][:]))))
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
        xVec = getxVec(np.ravel(h)).T
        listX.append(xVec.tolist()[0])
    return np.matrix(listX)

def getNextwVec(wVec,A,B,alfaVec,yVec):
    return wVec - ( np.linalg.inv(A.T * B * A) * (A.T) * (alfaVec - yVec))

A = getAMatrix(hVals)
alfaVec = getAlfaVec(wVecInit,hVals)
B = getBMatrix(alfaVec)

def getOptwVec(A, yVec, hVals, wVecInit):
    wVec = wVecInit
    alfaVec = getAlfaVec(wVec,hVals)
    B = getBMatrix(alfaVec)
    i = 0
    while i < MAX_ITER:    
        likelihoodOld = calcLikelihood(alfaVec, yVec)
        wVec = getNextwVec(wVec,A,B,alfaVec,yVec)
        alfaVec = getAlfaVec(wVec,hVals)
        B = getBMatrix(alfaVec)
        likelihood = calcLikelihood(alfaVec,yVec)
        dist = np.absolute((likelihood-likelihoodOld)*100/likelihoodOld)
        if(dist<MIN_DIST): 
            break
        i+=1
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
wVecOpt2 = wVecOpt[2].item()

x1vals = np.arange(-100, 200, GRAF_STEP)
x2vals = np.arange(-100, 200, GRAF_STEP)
zvals = np.zeros(len(x1vals))



for i in range(len(x1vals)):
    for j in range(len(x2vals)):
        hValsGraf = np.array([x1vals[i],x2vals[j]])
        zvals[i] = alfaVal(wVecOpt,getxVec(hValsGraf))    
    
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(-100, 200, GRAF_STEP)
X, Y = np.meshgrid(x, y)
zs = np.array([alfaVal(wVecOpt,getxVec(np.array([x,y]))) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
surf = ax.plot_surface(X, Y, Z, cmap='jet')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()