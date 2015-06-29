


import os
import csv
import math
import numpy as np
from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['rates','dvector', 'intrinsic_rate', 'social_rate', 'na_rate', 'dist', 'lag', 'left_angle', 'right_angle']



# Define data and stochastics

maxLag = 5

lag = Uniform('lag', lower=0, upper=maxLag)
dist = Uniform('dist', lower=0, upper=2000)
intrinsic_rate = Uniform('intrinsic_rate',lower=0, upper=1)
social_rate = Uniform('social_rate', lower=0, upper=1)
na_rate = Uniform('na_rate', lower=0, upper=1)
left_angle = Uniform('left_angle', lower=0, upper=pi)
right_angle = Uniform('right_angle', lower=0, upper=pi)


#os.chdir("D:\\shags\\")
allData=np.zeros([0,7]) #dive vector
for trial in np.arange(0,45):
   folderpath=os.path.join (os.getcwd()+'/data1/',str(trial) )
   fileimportname=os.path.join(folderpath,('tdata'+str(trial)+'.csv'))
           
   if os.path.isfile(fileimportname):
       with open (fileimportname) as tdata:
           reader=csv.reader(tdata)
           for row in reader:
               data=list(reader)
               result=np.array(data).astype('float')
        
       trialData=result[((result[:,5]==0)|((result[:,6]>(result[:,0]-0.5)))),:] 
       trialData[:,6]=trial
       allData=np.vstack((allData,trialData))


dvector = np.copy(allData[:,5])
dsize = len(dvector)
# first find the maximum number of dives that any individual could observe
maxDives=0
for thisRow in range(dsize):
        thisTime = allData[thisRow,0]        
        thisTrial = allData[thisRow,6]
        window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]<thisTime)&(allData[:,6]==thisTrial)&(allData[:,5]==1),:]
        if len(window)>maxDives:
            maxDives=len(window)

dparams = np.zeros((dsize,maxDives,3)).astype(np.float32) # time, dist, angle
dparams[:,:,0]=maxLag + 1.0
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = math.radians(allData[thisRow,4])
    thisTrial = allData[thisRow,6]
    window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]<thisTime)&(allData[:,6]==thisTrial)&(allData[:,5]==1),:]
    ncount = 0
    for w in window:
        xj = w[2]
        yj = w[3]
        tj = w[6]        
        dparams[thisRow,ncount,0] = thisTime - tj
        dparams[thisRow,ncount,1] = (((thisX-xj)**2+(thisY-yj)**2))**0.5
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        dparams[thisRow,ncount,2] = -1.0*math.atan2(math.sin(angle), math.cos(angle))
        #if dparams[thisRow,ncount,2]<0:
        #    dparams[thisRow,ncount,2]+=2.0*pi
        ncount+=1



print(maxDives)

    
@deterministic(plot=False)
def rates(T=lag,D=dist,d1=left_angle,d2=right_angle,i=intrinsic_rate,s=social_rate, na=na_rate):
    #os.chdir("D:\\shags\\")
    # 0 TIME # 1 DK # 2 XPOS # 3 YPOS # 4 ANGLE # 5 DIVE # 6 TRIAL
    #dstop = d1 + d2
    svector=np.zeros_like(dvector) #social vector
    svector[allData[:,0]<T] = -1
    svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&(dparams[:,:,2]>-d1)&(dparams[:,:,2]<d1),1)]=1
    #svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&((dparams[:,:,2]<d1)|(dparams[:,:,2]>dstop)),1)]=1

    out = np.ones_like(dvector).astype(float)*i 
    out[svector<0]=na
    out[svector>0]=s
    
    return out

dives = Bernoulli('dives', p=rates, value=dvector, observed=True)

