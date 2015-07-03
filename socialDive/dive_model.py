


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
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['rates','dvector', 'intrinsic_rate', 'social_rate', 'na_rate', 'dist', 'lag', 'left_angle', 'right_angle']



workDir = '/home/ctorney/workspace/diveRules/'
# Define data and stochastics

maxLag = 5

lag = Uniform('lag', lower=0, upper=maxLag)
dist = Uniform('dist', lower=0, upper=2000)
intrinsic_rate = Uniform('intrinsic_rate',lower=0, upper=1)
social_rate = Uniform('social_rate', lower=0, upper=1)
na_rate = Uniform('na_rate', lower=0, upper=1)
#left_angle = Beta('left_angle', alpha=4, beta=10)
left_angle = Uniform('left_angle', lower=0, upper=0.5*pi)
right_angle = Uniform('right_angle', lower=0, upper=pi)

allDF = pd.DataFrame()
for trial in np.arange(0,45):
    fileimportname= workDir + '/data/tdata'+str(trial)+'.csv'
    if os.path.isfile(fileimportname):
        df = pd.read_csv(fileimportname)
        df['trial']=trial
        allDF = allDF.append(df[(df['dive']==0)|(df['time']==df['time_dive'])])

           
allData = allDF.values


dvector = np.copy(allData[:,5])
dsize = len(dvector)
# first find the maximum number of dives that any individual could observe
maxDives=0
for thisRow in range(dsize):
        thisTime = allData[thisRow,0]        
        thisTrial = allData[thisRow,8]
        window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]<thisTime)&(allData[:,8]==thisTrial)&(allData[:,5]==1),:]
        if len(window)>maxDives:
            maxDives=len(window)

dparams = np.zeros((dsize,maxDives,3)).astype(np.float32) # time, dist, angle
dparams[:,:,0]=maxLag + 1.0
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = math.radians(allData[thisRow,4])
    thisTrial = allData[thisRow,8]
    window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]<thisTime)&(allData[:,8]==thisTrial)&(allData[:,5]==1),:]
    ncount = 0
    for w in window:
        xj = w[2]
        yj = w[3]
        tj = w[7]        
        dparams[thisRow,ncount,0] = thisTime - tj
        dparams[thisRow,ncount,1] = (((thisX-xj)**2+(thisY-yj)**2))**0.5
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        dparams[thisRow,ncount,2] = math.atan2(math.sin(angle), math.cos(angle))
        ncount+=1




    
@deterministic(plot=False)
def rates(T=lag,D=dist,d1=left_angle,d2=right_angle,i=intrinsic_rate,s=social_rate, na=na_rate):
    #os.chdir("D:\\shags\\")
    # 0 TIME # 1 DK # 2 XPOS # 3 YPOS # 4 ANGLE # 5 DIVE # 6 TRIAL
    #dstop = d1 + d2
    d3 = 3.142
    svector=np.zeros_like(dvector) #social vector
    svector[allData[:,0]<T] = -1
    svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&(dparams[:,:,2]>-d3)&(dparams[:,:,2]<d3),1)]=1
    #svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&((dparams[:,:,2]<d1)|(dparams[:,:,2]>dstop)),1)]=1

    out = np.ones_like(dvector).astype(float)*i 
    out[svector<0]=na
    out[svector>0]=s
    
    return out

dives = Bernoulli('dives', p=rates, value=dvector, observed=True)

