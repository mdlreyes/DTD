"""
ccfrac.py

Computes the conversion from N_stars to N_CC.
"""

import numpy as np
from scipy import integrate

# Normalization from Kroupa IMF
class kroupaIMF(object):
    def __init__(self, A1=1.):
        self.A1 = A1
        self.A2 = self.A1*0.08**(-0.3)/(0.08**(-1.3))
        self.A3 = self.A2*0.5**(-1.3)/(0.5**(-2.3))
    def __call__(self, x):
        if np.size(x) == 1:
            x = [x]
        gamma = np.zeros(len(x))
        A = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] <= 0.08:
                gamma[i] = -0.3
                A[i] = self.A1
            elif x[i] <= 0.5:
                gamma[i] = -1.3
                A[i] = self.A2
            else:
                gamma[i] = -2.3
                A[i] = self.A3
            
        return np.power(x,gamma) * A
    
# Renormalize Kroupa IMF across reasonable range of stellar masses
kroupa = kroupaIMF()
total, _ = integrate.quad(kroupa, a=0.01, b=100.)
kroupa.A1 = kroupa.A1/total
kroupa.A2 = kroupa.A2/total
kroupa.A3 = kroupa.A3/total

# Compute (# stars formed with M > 8 Msun)/(total # stars)
highmass, _ = integrate.quad(kroupa, a=10., b=40.)
allmass, _ = integrate.quad(kroupa, a=0.01, b=100.)
CCfrac = highmass/allmass