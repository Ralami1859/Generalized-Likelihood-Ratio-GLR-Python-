from __future__ import division
import numpy as np
from ImprovedGLR import *
import matplotlib.pyplot as plt



"""
---------------------------------------------------------------------------------------------------------------------------
                                         Define the environment (Piece-wise stationary Bernoulli Process)
---------------------------------------------------------------------------------------------------------------------------
"""

def Bernoulli_Environment(means, period):
    env = np.array([])
    for p in range(means.size):
        env = np.append(env, means[p]*np.ones((period[p])))
    return env

"""
environment = constructEnvironment(environment, Period) # Building the piece-wise stationary Bernoulli distributions
"""
env = Bernoulli_Environment(np.array([0.9,0.1,0.8,0.2,0.6,0.2]),np.array([600,700,200,300,400,800]))

seq_obs = ((np.random.uniform(0,1,np.size(env)) < env) == True)*1 # sequence of observations


"""
------------------------------------------------------------------------------------------------------------------------------
                                Launch the change-point detection using Improved GLR strategy
------------------------------------------------------------------------------------------------------------------------------
"""
sigma = 0.5
delta = 0..01


glr = ImprovedGLR(sigma) # instantiation of a GLR object

#bocpd = BOCPD(1/seq_obs.size)
vect_restart = np.array([]) # for plotting....

# Launching the interaction with the environment
for t in range (env.size):
    obs = np.random.uniform() < env[t]
    restart = glr.process(obs)
    vect_restart = np.append(vect_restart, restart)




"""
----------------------------------------------------------------------------------------------------------------------------
                    Plotting the results
----------------------------------------------------------------------------------------------------------------------------
"""

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax3.plot(range(env.size), vect_restart.tolist(), color='red', marker = '.', label = "Restart done by GLR")
ax1.plot(range(env.size), env)
ax2.plot(range(env.size),seq_obs , marker='.')
plt.show()
