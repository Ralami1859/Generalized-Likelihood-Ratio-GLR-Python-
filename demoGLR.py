from __future__ import division
import numpy as np
import numpy.matlib as npMat
from GLR_Modules import * 
import matplotlib.pyplot as plt


"""
---------------------------------------------------------------------------------------------------------------------------
                                         Define the environment
---------------------------------------------------------------------------------------------------------------------------
"""

environment = np.array([0.9,0.1,0.8,0.2]) # Bernoulli distributions
Period = 500 # Length of each stationary period
environment = constructEnvironment(environment, Period) # Building the piece-wise stationary Bernoulli distributions


"""
----------------------------------------------------------------------------------------------------
                                Launch the change-point detection
----------------------------------------------------------------------------------------------------
"""



sigma = 0.5
delta= 0.05

CPEstimations = GLR(environment, sigma, delta) 



"""
----------------------------------------------------------------------------------------------------------------------------
                                       Plotting the results
----------------------------------------------------------------------------------------------------------------------------
"""

plt.plot(range(environment.size), CPEstimations.tolist(), color='red', marker='o', label = "GLR")
plt.grid(True)
plt.legend(loc='upper left')
plt.xlabel('Time step')
plt.ylabel('Change-point Estimation')
plt.show()
