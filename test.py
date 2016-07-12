from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from utils import *

lx = 500000.
dx=1000.

#x=np.linspace(-lx,lx-dx,nx)
x = np.arange(-lx,lx,dx)
nx = x.size
s =0.5 
r = 1.#50000
phi = gen_srf_fft(x,s,r,"gaussian")
print np.var(phi)
plt.plot(phi)
plt.show()
