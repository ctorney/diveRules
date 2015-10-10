import numpy as np
from matplotlib import pylab as plt
import dive_model
#import noba_model
import pymc
from pymc import MCMC, MAP
from pymc.Matplot import plot as mcplot


M = MCMC(dive_model)

M.sample(iter=2000000, burn=0, thin=10,verbose=0)
mcplot(M)

plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
plt.hist([M.trace('social_rate')[:]],label='social')
plt.legend(loc='upper left')
plt.xlim(0,0.2)
plt.show()

d1=M.trace('blind_angle')[:]
bc = d1*180/3.142
plt.hist(bc)
plt.xlim(0,380)
plt.show()

plt.hist([M.trace('lag')[:]])
plt.legend(loc='upper left')
plt.xlim(0,5)
plt.show()

plt.hist([M.trace('dist')[:]],100)
plt.legend(loc='upper left')
plt.xlim(0,200)
plt.show()



np.savetxt('dist.txt', M.trace('dist')[:]) 
np.savetxt('lag.txt', M.trace('lag')[:]) 
np.savetxt('blind_angle.txt', M.trace('blind_angle')[:]) 
np.savetxt('social_rate.txt', M.trace('social_rate')[:]) 
np.savetxt('intrinsic_rate.txt', M.trace('intrinsic_rate')[:]) 


