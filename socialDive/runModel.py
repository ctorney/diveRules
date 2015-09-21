
from matplotlib import pylab as plt
import dive_model
import pymc
from pymc import MCMC, MAP
from pymc.Matplot import plot as mcplot


M = MCMC(dive_model)


M.sample(iter=12000, burn=500, thin=10,verbose=0)
##mcplot(M)
#
#plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
#plt.hist([M.trace('social_rate')[:]],label='social')
#plt.legend(loc='upper left')
#plt.xlim(0,0.2)
#plt.show()
#
#
#d1=M.trace('blind_angle')[:]
#
#
#
#bc = d1*180/3.142
#plt.hist(bc)
#plt.xlim(0,180)
#plt.show()
#
#plt.hist([M.trace('lag')[:]])
#plt.legend(loc='upper left')
#plt.xlim(0,5)
#plt.show()
#
#plt.hist([M.trace('dist')[:]],100)
#plt.legend(loc='upper left')
#plt.xlim(0,200)
#plt.show()


M2 = MAP(dive_model)
M2.fit()
print(M2.AIC)
print(M2.BIC)


plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
plt.hist([M.trace('social_rate')[:]],label='social')
plt.legend(loc='upper left')
plt.xlabel('dive rate')
plt.ylabel('frequency')
plt.xlim(0,0.2)
plt.savefig('rates.png')



plt.show()


plt.hist([2*bc])
#plt.legend(loc='upper left')
plt.xlabel('visual angle')
plt.ylabel('frequency')
plt.xlim(0,180)
plt.savefig('blind_angle.png')
