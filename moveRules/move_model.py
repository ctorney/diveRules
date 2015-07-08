
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

__all__ = ['replen','attlen','eta','maxrho','mvector','blind_angle']


# rho is tanh(a dx) * exp(-b dx)
# the inflexion point is located at (1/2a)ln(2a/b + sqrt((2a/b)^2+1)



workDir = '/home/ctorney/workspace/diveRules/'
# Define data and stochastics

replen = Uniform('replen',lower=1,upper=50)
attlen = Uniform('attlen',lower=1,upper=50)
maxrho = Uniform('maxrho',lower=0,upper=1)
eta = Uniform('eta',lower=0,upper=1) # persistence
blind_angle = Uniform('blind_angle', lower=0, upper=pi)

dt = 1


#os.chdir("D:\\shags\\")
allDF = pd.DataFrame()
for trial in np.arange(0,45):
    print(trial)
    fileimportname= workDir + '/data/tdata'+str(trial)+'.csv'
    if os.path.isfile(fileimportname):
        df = pd.read_csv(fileimportname)
        df['trial']=trial
        df['dtheta']=np.NaN
        for index, row in df.iterrows():
            thisTime =  row['time']
            thisID = row['id']
            thisTheta = row['angle']
            nextTime = df[(np.abs(df['time']-(thisTime+dt))<1e-6)&(df['id']==thisID)]
            if len(nextTime)==1:
                df.ix[index,'dtheta'] = nextTime.iloc[0]['angle'] -  thisTheta 
        allDF = allDF.append(df[(df['dive']==0)|(df['time']==df['time_dive'])])



allDF = allDF[pd.notnull(allDF['dtheta'])]

allData = allDF.values

mvector = np.copy(allData[:,9])
mvector = np.radians(mvector)
mvector[mvector<-pi]=mvector[mvector<-pi]+2*pi
mvector[mvector>pi]=mvector[mvector>pi]-2*pi
dsize = len(mvector)
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
rhos = np.zeros((dsize,maxN)).astype(np.float32) # dist, angle
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisID = allData[thisRow,1]
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = math.radians(allData[thisRow,4])
    thisTrial = allData[thisRow,8]
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


avDir=np.zeros_like(mvector)
ind=0
for f in dparams:
    ff=f[:,1]
    avDir[ind]= np.mean(ff[(f[:,0]!=0)&(f[:,0]<500)&(f[:,0]>20)])
    ind+=1
print(np.nanmean(avDir*mvector))

@stochastic(observed=True)
def moves(dr=replen,da=attlen,mr=maxrho, d1=angle,ep=eta,value=mvector):
    a=1/dr
    b=1/(da*dr)
    prefact = mr/((1-b/(a+b))*(b/(a+b))**(b/a))
    lambdas = np.zeros_like(mvector)
    #lambdas[np.abs(mvector)>pi]=pi
    lambdas[np.abs(mvector)>(1-ep)*pi]=pi
    lambdas[np.abs(mvector)<(1-ep)*pi]=mvector[np.abs(mvector)<(1-ep)*pi]/(1-ep)
    
    # first calculate all the rhos
    rhos = prefact*(1.0-np.exp(-a*dparams[:,:,0]))*np.exp(-b*dparams[:,:,0])
    rhos[((dparams[:,:,1]>-d1)&(dparams[:,:,1]<d1))]=0
    nc = np.sum(rhos,1) # normalizing constant

    wwc = (rhos)*(1/(2*pi)) * (1-np.power(rhos,2))/(1+np.power(rhos,2)-2*rhos*np.cos((lambdas-dparams[:,:,1].transpose()).transpose())) # weighted wrapped cauchy
    wwc = np.sum(wwc,1)/nc
    wwc[np.isinf(wwc)]=1/(2*pi)
    return np.sum(np.log(wwc))
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

