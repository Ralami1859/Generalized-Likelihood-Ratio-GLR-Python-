import numpy as np


class ImprovedGLR(object):
    

    
    def __init__(self, sigma, delta = 0.01):
        self.sigma = sigma
        self.delta = delta
        self.t0 = 0
        self.means_left = np.array([])
        self.means_right = np.array([])
        self.seq_obs = np.array([])
        self.t = 0
    
    def ConfidenceLevel(self, s):
        t0 = self.t0
        t = self.t
        a = 1/(s-t0+1) + 1/(t-s)
        b = 1 + 1/(t-t0+1)
        c = 2*(t-t0)*np.sqrt(t-t0+1)/self.delta
        return self.sigma*np.sqrt(2*a*b*np.log(c))
    
    def ChangePointIndicator(self):
        restart = 0
        t = self.t
        for s in range(t-self.t0):
            diff = self.means_left[s] - self.means_right[s]
            if (np.abs(diff) >= self.ConfidenceLevel(s+self.t0)):
                restart = 1
                break
        return restart
    
    def restarting(self):
        self.t0 = self.t+1
        self.means_left = np.array([])
        self.means_right = np.array([])
        self.seq_obs = np.array([])
        
        
    def process(self, x):
        self.seq_obs = np.append(self.seq_obs, x)
        restart = self.ChangePointIndicator()
        if restart == 1:
            self.restarting()
        else:
            N = np.array(range(0, self.t-self.t0)[::-1])
            self.means_left = np.append(self.means_left, np.mean(self.seq_obs))
            self.means_right = np.append((self.means_right*N + self.seq_obs[-1])/(N+1) , self.seq_obs[-1])
        self.t += 1
        return restart        
