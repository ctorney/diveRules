#tester


from matplotlib import pylab as plt


import move_model
import pymc
from pymc import MCMC
from pymc.Matplot import plot as mcplot
M = MCMC(move_model)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.left_angle, M.right_angle, M.lag, M.dist],  delay=1000)
M.sample(iter=20000, burn=100, thin=10,verbose=0)
#mcplot(M)
#from pylab import hist, show

#hist(M.trace('late_mean')[:])
#show()






plt.hist([M.trace('blind_angle')[:]],label='ba')
plt.legend(loc='upper left')
plt.xlim(0,4)
plt.show()

plt.hist([M.trace('replen')[:]],label='alpha')
plt.hist([M.trace('attlen')[:]],label='beta')
plt.legend(loc='upper left')
plt.xlim(0,50)
plt.show()

plt.hist([M.trace('maxrho')[:]],label='beta')
plt.legend(loc='upper left')
plt.xlim(0,1)
plt.show()
