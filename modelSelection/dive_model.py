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

__all__ = ['dives','dvector','intrinsic_rate','social_rate','dist','lag','blind_angle']


workDir = '/home/ctorney/workspace/diveRules/'
# Define data and stochastics

maxLag = 5

lag = Uniform('lag', lower=0.5, upper=maxLag,value=2.04)
dist = Uniform('dist', lower=0, upper=100,value= 54.719)
#dist=TruncatedNormal('dist',mu=25,tau=1.0/(12.5**2),a=0,b=200)

intrinsic_rate = Uniform('intrinsic_rate',lower=0, upper=1,value=0.08)
social_rate = Uniform('social_rate', lower=0, upper=1,value=0.1646)
blind_angle = Uniform('blind_angle', lower=0, upper=pi,value=0.9547)


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
        thisTrial = allData[thisRow,9]
        window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]<thisTime)&(allData[:,9]==thisTrial)&(allData[:,5]==1),:]
        if len(window)>maxDives:
            maxDives=len(window)

# build an array to store all observed dives and their time, distance and angle from the focal individual
dparams = np.zeros((dsize,maxDives,3)).astype(np.float32) # time, dist, angle
dparams[:,:,0]=maxLag + 1.0
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisIndex = allData[thisRow,1]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = math.radians(allData[thisRow,4])
    thisTrial = allData[thisRow,9]
    thisTrack = allData[(allData[:,9]==thisTrial)&(allData[:,1]==thisIndex),:]
    window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]<thisTime)&(allData[:,9]==thisTrial)&(allData[:,5]==1),:]
    ncount = 0
    
    for w in window:
        xj = w[2]
        yj = w[3]
        tj = w[0]
        texj = w[7]
        thatTime = thisTrack[(thisTrack[:,0]==tj),:]
        if len(thatTime)!=1:
            continue
        oldX = thatTime[0,2]
        oldY = thatTime[0,3]
        oldAngle = math.radians(thatTime[0,4])
                
        dparams[thisRow,ncount,0] = thisTime - texj
        dparams[thisRow,ncount,1] = (((oldX-xj)**2+(oldY-yj)**2))**0.5
        dx = xj - oldX
        dy = yj - oldY
        angle = math.atan2(dy,dx)
        angle = angle - oldAngle
        dparams[thisRow,ncount,2] = math.atan2(math.sin(angle), math.cos(angle))
        ncount+=1





@stochastic(observed=True)
def dives(T=lag,D=dist,d1=blind_angle,i=intrinsic_rate,s=social_rate, value=dvector):
     def logp(value, T, D, d1, i, s):
        
        svector=np.zeros_like(value) #social vector
        svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&(dparams[:,:,2]>-d1)&(dparams[:,:,2]<d1),1)]=1

        asocdiv = value[svector==0]
        socdiv = value[svector==1]
        return (np.sum(np.log((1.0-i)**(1-asocdiv)*(i**asocdiv))) + np.sum(np.log((1.0-s)**(1-socdiv)*(s**socdiv))))

        

    
