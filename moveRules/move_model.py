
import os
import csv
import math
import numpy as np
from datetime import datetime
from pymc import *
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
from pymc.Matplot import plot as mcplot
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['rates','dvector', 'intrinsic_rate', 'social_rate', 'na_rate', 'dist', 'lag', 'left_angle', 'right_angle']


# rho is tanh(a dx) * exp(-b dx)
# the inflexion point is located at (1/2a)ln(2a/b + sqrt((2a/b)^2+1)



workDir = '/home/ctorney/workspace/diveRules/'
# Define data and stochastics

maxLag = 5

lag = Uniform('lag', lower=0, upper=maxLag)
dist = Uniform('dist', lower=0, upper=2000)
intrinsic_rate = Uniform('intrinsic_rate',lower=0, upper=1)
social_rate = Uniform('social_rate', lower=0, upper=1)
na_rate = Uniform('na_rate', lower=0, upper=1)
left_angle = Uniform('left_angle', lower=0, upper=pi)
#left_angle = Beta('left_angle', alpha=2, beta=2)
right_angle = Uniform('right_angle', lower=0, upper=pi)


#os.chdir("D:\\shags\\")
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
# first find the maximum number of neighbours that any individual could observe
maxN=0
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisID = allData[thisRow,1]
    thisTrial = allData[thisRow,8]
    window = allData[(allData[:,0]==thisTime)&(allData[:,8]==thisTrial)&(allData[:,1]!=thisID),:]
    if len(window)>maxN:
        maxN=len(window)#

dparams = np.zeros((dsize,maxN,2)).astype(np.float32) # dist, angle
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = math.radians(allData[thisRow,4])
    thisTrial = allData[thisRow,6]
    window = allData[(allData[:,0]==thisTime)&(allData[:,8]==thisTrial)&(allData[:,1]!=thisID),:]
    ncount = 0
    for w in window:
        xj = w[2]
        yj = w[3]
        dparams[thisRow,ncount,0] = (((thisX-xj)**2+(thisY-yj)**2))**0.5
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        dparams[thisRow,ncount,1] = math.atan2(math.sin(angle), math.cos(angle))
        ncount+=1



#print(maxDives)

    
#@deterministic(plot=False)
#def rates(T=lag,D=dist,d1=left_angle,d2=right_angle,i=intrinsic_rate,s=social_rate, na=na_rate):
#    #os.chdir("D:\\shags\\")
#    # 0 TIME # 1 DK # 2 XPOS # 3 YPOS # 4 ANGLE # 5 DIVE # 6 TRIAL
#    #dstop = d1 + d2
#    svector=np.zeros_like(dvector) #social vector
#    svector[allData[:,0]<T] = -1
##svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&(dparams[:,:,2]>-d1)&(dparams[:,:,2]<d1),1)]=1
#    #svector[np.any((dparams[:,:,0]<T)&(dparams[:,:,1]<D)&((dparams[:,:,2]<d1)|(dparams[:,:,2]>dstop)),1)]=1##
#
#    out = np.ones_like(dvector).astype(float)*i 
#    out[svector<0]=na
#    out[svector>0]=s
#    
#    return out#
#
#dives = Bernoulli('dives', p=rates, value=dvector, observed=True)

