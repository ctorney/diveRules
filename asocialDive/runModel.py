


from matplotlib import pylab as plt


import dive_model
import pymc
from pymc import MCMC,MAP
from pymc.Matplot import plot as mcplot
M = MCMC(dive_model)


M.sample(iter=12000, burn=500, thin=10,verbose=0)
mcplot(M)
#from pylab import hist, show

#hist(M.trace('late_mean')[:])
#show()






plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
plt.legend(loc='upper left')
plt.xlim(0,0.2)
plt.show()


M2 = MAP(dive_model)
M2.fit()
print(M2.AIC)
print(M2.BIC)
