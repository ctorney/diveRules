


from matplotlib import pylab as plt


import dive_model
import pymc
from pymc import MCMC,MAP
from pymc.Matplot import plot as mcplot
#M = MCMC(dive_model)
#
#
#M.sample(iter=12000, burn=500, thin=10,verbose=0)
#mcplot(M)
##from pylab import hist, show
#
##hist(M.trace('late_mean')[:])
##show()
#
#
#
#
#
#
#plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
#plt.hist([M.trace('social_rate')[:]],label='social')
#plt.legend(loc='upper left')
#plt.xlim(0,0.2)
#plt.show()

#
#
#plt.hist([M.trace('lag')[:]])
#plt.legend(loc='upper left')
#plt.xlim(0,5)
#plt.show()
#
#plt.hist([M.trace('dist')[:]],100)
#plt.legend(loc='upper left')
#plt.xlim(0,2000)
#plt.show()
#
#na = dive_model.na_rate.get_value()
#allRates = dive_model.rates.get_value()
#allDives = dive_model.dvector
#goodRates = allRates[allRates!=na]
#goodDives = allDives[allRates!=na]
#
#pymc.bernoulli_like(goodDives,goodRates)

M2 = pymc.MAP(dive_model)
M2.fit()
print(M2.AIC)
print(M2.BIC)