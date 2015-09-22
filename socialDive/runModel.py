
from matplotlib import pylab as plt
import dive_model
import pymc
from pymc import MCMC, MAP
from pymc.Matplot import plot as mcplot


M = MCMC(dive_model)


M.sample(iter=1200, burn=500, thin=10,verbose=0)
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
#
#
##M.sample(iter=120000, burn=500, thin=100,verbose=0)
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
#plt.xlim(0,400)
#plt.show()
#
#
#
#
#plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
#plt.hist([M.trace('social_rate')[:]],label='social')
#plt.legend(loc='upper left')
#plt.xlabel('dive rate')
#plt.ylabel('frequency')
#plt.xlim(0,0.2)
#plt.savefig('rates.png')
#
#
#
#plt.show()
#
#
#plt.hist([2*bc])
##plt.legend(loc='upper left')
#plt.xlabel('visual angle')
#plt.ylabel('frequency')
#plt.xlim(0,180)
#plt.savefig('blind_angle.png')

ma = np.mean(aa,1)
mb = np.mean(bb,1)

plt.hist(ma[ma>0],100, weights=np.zeros_like(ma[ma>0]) + 1. / ma[ma>0].size)
plt.xlim(0,100)
plt.show()

plt.hist(mb[mb>0],100, weights=np.zeros_like(mb[mb>0]) + 1. / mb[mb>0].size)
plt.xlim(0,100)
plt.show()


ma = np.max(aa,1)
mb = np.max(bb,1)

plt.hist(ma[ma>0],100, weights=np.zeros_like(ma[ma>0]) + 1. / ma[ma>0].size)
plt.xlim(0,200)
plt.show()

plt.hist(mb[mb>0],100, weights=np.zeros_like(mb[mb>0]) + 1. / mb[mb>0].size)
plt.xlim(0,200)
plt.show()
