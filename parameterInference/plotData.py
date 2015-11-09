import numpy as np
from scipy.stats import norm
from matplotlib import pylab as plt


dist = np.loadtxt('dist.txt')
lag = np.loadtxt('lag.txt')
ba = np.loadtxt('blind_angle.txt')
sr = np.loadtxt('social_rate.txt')
ir = np.loadtxt('intrinsic_rate.txt')
ba = ba[5000:25000]
dist = dist[5000:25000]
lag = lag[5000:25000]
sr = sr[5000:25000]
ir = ir[5000:25000]
#maximum likelihood estimates
mldist = 69.523326698013
mllag = 1.659979663079878
mlva = 1.910270458127053*180/3.142
mlir = 0.06304935362561966

mlsr = 0.14855277873751646






plt.axvline(mlsr,linewidth=2, color='gray',linestyle='-')
plotArray = sr
range = np.arange(np.mean(sr)-4*np.std(sr), np.mean(plotArray)+4*np.std(sr), 0.0001)
plt.plot(range, norm.pdf(range,np.mean(sr),np.std(sr)),linewidth=2, color='k',linestyle='--', dashes=(10,2))
plt.hist(sr,40,normed=1,range=[0, 0.2],  color='red')

plt.axvline(mlir,linewidth=2, color='gray',linestyle='-')
range = np.arange(np.mean(ir)-4*np.std(ir), np.mean(ir)+4*np.std(ir), 0.0001)
plt.plot(range, norm.pdf(range,np.mean(ir),np.std(ir)),linewidth=2, color='k',linestyle='--', dashes=(10,2))
plt.hist(ir,40,normed=1,range=[0, 0.2],   color='blue')
plt.xlim([0,0.2])
plt.show()

plt.axvline(mldist,linewidth=2, color='gray',linestyle='-')


range = np.arange(np.mean(dist)-4*np.std(dist), np.mean(dist)+4*np.std(dist), 0.0001)
plt.plot(range, norm.pdf(range,np.mean(dist),np.std(dist)),linewidth=2, color='k',linestyle='--', dashes=(10,2))
plt.hist(dist,50,normed=1,range=[00,100], color='blue')
plt.xlim([0,100])
plt.show()



plt.hist(lag,50,normed=1,range=[0,5], color='blue')
plt.axvline(mllag,linewidth=2, color='gray',linestyle='-')
range = np.arange(0,5, 0.0001)
plt.plot(range, norm.pdf(range,np.mean(lag),np.std(lag)),linewidth=2, color='k',linestyle='--', dashes=(10,2))
plt.xlim([0,5])
plt.show()




plt.axvline(mlva,linewidth=2, color='gray',linestyle='-')
fullVA = ba*180/3.142
range = np.arange(0,180, 0.0001)
plt.plot(range, norm.pdf(range,np.mean(fullVA),np.std(fullVA)),linewidth=2, color='k',linestyle='--', dashes=(10,2))
plt.xlim([0,180])
plt.hist(fullVA,50,normed=1,range=[0,180] ,color='blue')

plt.show()


print('social rate')
print('max likelihood estimae ' + str(mlsr) + ' mean ' + str(np.mean(sr)) + ', sd ' + str(np.std(sr)) + ', 95% Credible Interval '+ str(np.percentile(sr,2.5)) + '-' + str(np.percentile(sr,97.5)))

#effect size
print('intrinsic rate')
print('max likelihood estimae ' + str(mlir) + ' mean ' + str(np.mean(ir)) + ', sd ' + str(np.std(ir)) + ', 95% Credible Interval '+ str(np.percentile(ir,2.5)) + '-' + str(np.percentile(ir,97.5)))
print('Cohens d = ' + str((np.mean(sr)-np.mean(ir))/(((np.std(sr)**2+np.std(ir)**2)*0.5)**0.5)))


print('time lag')
print('max likelihood estimae ' + str(mllag) + ' mean ' + str(np.mean(lag)) + ', sd ' + str(np.std(lag)) + ', 95% Credible Interval '+ str(np.percentile(lag,2.5)) + '-' + str(np.percentile(lag,97.5)))
print('visual angle')
print('max likelihood estimae ' + str(mlva) + ' mean ' + str(np.mean(fullVA)) + ', sd ' + str(np.std(fullVA)) + ', 95% Credible Interval '+ str(np.percentile(fullVA,2.5)) + '-' + str(np.percentile(fullVA,97.5)))

print('distance')
print('max likelihood estimae ' + str(mldist) + ' mean ' + str(np.mean(dist)) + ', sd ' + str(np.std(dist)) + ', 95% Credible Interval '+ str(np.percentile(dist,2.5)) + '-' + str(np.percentile(dist,97.5)))

