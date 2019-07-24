from __future__ import division
import numpy as np
import numpy.matlib as npMat



def constructEnvironment(environment, Period):
    vect = np.array([])
    for periode in range(environment.size):
        vect = np.append(vect, environment[periode]*np.ones((Period)))
    return vect


def PlayBernoulli(arm):
    return int ((np.random.uniform() < arm) == True)


def GLRConfidenceLevel(t0,s, t, delta, sigma):
    a = 1/(s-t0+1) + 1/(t-s)
    b = 1 + 1/(t-t0+1)
    c = 2*(t-t0)*np.sqrt(t-t0+1)/delta
    return sigma*np.sqrt(2*a*b*np.log(c))


def ChangePointIndicator(t0,x, delta, sigma):
    CPEstimation = t0
    t = np.size(x)
    for s in range(t-t0):
        diff = np.mean(x[t0:s+t0+1]) - np.mean(x[s+1+t0:t+1])
        if (np.abs(diff) >= GLRConfidenceLevel(t0,s+t0, t, delta, sigma)):
            CPEstimation = t
            break
    return CPEstimation




def GLR(environment, sigma, delta):
    T = np.size(environment)
    CPEstimations = np.array([])
    Restart = 0
    x = np.array([])
    for t in range(T):
        x = np.append(x, PlayBernoulli(environment[t]))
        Restart = ChangePointIndicator(Restart, x, delta, sigma)
        CPEstimations = np.append(CPEstimations, Restart)
    return CPEstimations        

        
