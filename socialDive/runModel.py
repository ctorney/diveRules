#tester


from matplotlib import pylab as plt


import dive_model
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot
M = MCMC(dive_model)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.left_angle, M.right_angle, M.lag, M.dist],  delay=1000)
M.sample(iter=200000, burn=10000, thin=100,verbose=0)
#mcplot(M)
#from pylab import hist, show

#hist(M.trace('late_mean')[:])
#show()






plt.hist([M.trace('intrinsic_rate')[:]],label='intrinsic')
plt.hist([M.trace('social_rate')[:]],label='social')
plt.legend(loc='upper left')
plt.xlim(0,0.2)
plt.show()


d1=M.trace('left_angle')[:]

d2=M.trace('right_angle')[:]

#plt.hist(d2)
#plt.xlim(0,6.28)
#plt.show()

bc = d1+0.5*d2
plt.hist(d1)
plt.xlim(0,6.28)
plt.show()

plt.hist([M.trace('lag')[:]])
plt.legend(loc='upper left')
plt.xlim(0,5)
plt.show()

plt.hist([M.trace('dist')[:]],100)
plt.legend(loc='upper left')
plt.xlim(0,2000)
plt.show()

