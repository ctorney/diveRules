import numpy as np
from matplotlib import pylab as plt

import noba_model
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot


M = MCMC(noba_model)

M.sample(iter=2000000, burn=0, thin=10,verbose=0)
mcplot(M)

plt.hist([M.trace('intrinsic_rate')[:]],500,label='intrinsic')
plt.hist([M.trace('social_rate')[:]],500,label='social')
plt.legend(loc='upper left')
plt.xlim(0,0.2)
plt.show()

plt.hist([M.trace('lag')[:]])
plt.legend(loc='upper left')
plt.xlim(0,5)
plt.show()

plt.hist([M.trace('dist')[:]],100)
plt.legend(loc='upper left')
plt.xlim(0,200)
plt.show()



np.savetxt('distNOBA.txt', M.trace('dist')[:]) 
np.savetxt('lagNOBA.txt', M.trace('lag')[:]) 
np.savetxt('social_rateNOBA.txt', M.trace('social_rate')[:]) 
np.savetxt('intrinsic_rateNOBA.txt', M.trace('intrinsic_rate')[:]) 


