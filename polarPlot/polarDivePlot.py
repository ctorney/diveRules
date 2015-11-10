import os
import csv
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
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import pandas as pd
import math

def convert_to_polar(x, y):
    theta = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    return theta, r 
    
def convert_from_polar(theta, r):
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    return x, y     



workDir = '/home/ctorney/workspace/diveRules/'


# open all the data files and import them
allDF = pd.DataFrame()
for trial in np.arange(0,45):
    #trial = 0
    fileimportname= workDir + '/data/tdata'+str(trial)+'.csv'
    if os.path.isfile(fileimportname):
        df = pd.read_csv(fileimportname)
        df['trial']=trial
        allDF = allDF.append(df[(df['dive']==0)|(df['time']==df['time_dive'])])
        
# convert to a numpy array
allData = allDF.values

# calculate the headings based on the difference between the positions at successive time steps
for thisTrial in np.unique(allData[:,9]):
    for thisIndex in np.unique(allData[:,1]):
        window = allData[(allData[:,1]==thisIndex)&(allData[:,9]==thisTrial),:]
        
        x = window[:,2]
        y = window[:,3]
        angs = np.radians(window[:,4])
        dx = x[1:]-x[0:-1]
        dy = y[1:]-y[0:-1]
        angs[0:-1] = np.arctan2(dy,dx)
        allData[(allData[:,1]==thisIndex)&(allData[:,9]==thisTrial),4]=angs
        
        
dvector = np.copy(allData[:,5])
dsize = len(dvector)

maxLag = 1.659979663079878


# build an array to store the relative angles and distances to all neighbours
socialDiveLocations = np.zeros((0,2)).astype(np.float32) 
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisIndex = allData[thisRow,1]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = (allData[thisRow,4])
    thisDive = (allData[thisRow,5])
    thisTrial = allData[thisRow,9]
    thisTrack = allData[(allData[:,9]==thisTrial)&(allData[:,1]==thisIndex),:]
    socRowLoc = np.zeros((0,2)).astype(np.float32) 
    if thisDive:
        window = allData[(allData[:,0]>=thisTime-maxLag)&(allData[:,0]!=thisIndex)&(allData[:,0]<thisTime)&(allData[:,9]==thisTrial)&(allData[:,5]==1),:]
        for w in window:
            xj = w[2]
            yj = w[3]
            tj = w[0]
            jAngle = (w[4])
            texj = w[7]
            thatTime = thisTrack[(thisTrack[:,0]==tj),:]
            if len(thatTime)!=1:
                continue
            oldX = thatTime[0,2]
            oldY = thatTime[0,3]
            oldAngle = (thatTime[0,4])
            
            r = (((oldX-xj)**2+(oldY-yj)**2))**0.5
            dx = xj - oldX
            dy = yj - oldY
            angle = math.atan2(dy,dx)
            angle = angle - oldAngle
            theta = math.atan2(math.sin(angle), math.cos(angle))
            
    
        
            socRowLoc = np.vstack((socRowLoc,[r, theta]))
    socialDiveLocations = np.vstack((socialDiveLocations,socRowLoc))


# build an array to store the relative angles and distances to all neighbours
allDiveLocations = np.zeros((0,2)).astype(np.float32) 
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisIndex = allData[thisRow,1]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = (allData[thisRow,4])
    thisDive = (allData[thisRow,5])
    thisTrial = allData[thisRow,9]
    thisTrack = allData[(allData[:,9]==thisTrial)&(allData[:,1]==thisIndex),:]
    window = allData[(allData[:,0]!=thisIndex)&(allData[:,0]==thisTime)&(allData[:,9]==thisTrial)&(allData[:,5]==1),:]
    ncount = 0
    allRowLoc = np.zeros((0,2)).astype(np.float32) 
    for w in window:
        xj = w[2]
        yj = w[3]
        tj = w[0]
        jAngle = (w[4])
        texj = w[7]
        thatTime = thisTrack[(thisTrack[:,0]==tj),:]
        if len(thatTime)!=1:
            continue
        oldX = thatTime[0,2]
        oldY = thatTime[0,3]
        oldAngle = (thatTime[0,4])
        
        r = (((oldX-xj)**2+(oldY-yj)**2))**0.5
        dx = xj - oldX
        dy = yj - oldY
        angle = math.atan2(dy,dx)
        angle = angle - oldAngle
        theta = math.atan2(math.sin(angle), math.cos(angle))
        
        allRowLoc = np.vstack((allRowLoc,[r, theta]))
    allDiveLocations = np.vstack((allDiveLocations,allRowLoc))
    

## POLAR PLOT OF ALL DIVES
binn2=7
binn1=24
maxr=75

theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
r2 = np.linspace(5, maxr, binn2+1)

# wrap to [0, 2pi]
allDiveLocations[allDiveLocations[:,1]<0,1] = allDiveLocations[allDiveLocations[:,1]<0,1] + 2 *pi

hista1=np.histogram2d(x=allDiveLocations[:,0],y=allDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  

#hista1[1:] =hista1[1:]/np.max(hista1[1:])
#hista1 =hista1/np.max(hista1)

size = 8
# make a square figure
#fig,axes = plt.subplots(1,1,figsize=(size, size),subplot_kw=dict(polar=True))
fig1=plt.figure(figsize=(size, size))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2[0:],hista1[0:],lw=0.0,vmin=np.min(hista1),vmax=np.max(hista1),cmap='OrRd')
ax2.yaxis.set_visible(False)

#ax2.set_xticklabels(['0°(front)', '45°', '90°', '135°', '180°(back)', '225°', 
                     #'270°', '315°'])

ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', 
               '135°', '', '225°','270°', '315°'],frac=1.1)

ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', 
                         label='twin', frame_on=False,
                         theta_direction=ax2.get_theta_direction(),
                         theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', 
               '', '180°(back)', '','', ''],frac=1.15)

xbin=np.linspace(-10,10,20)


## POLAR PLOT OF SOCIAL DIVES


# wrap to [0, 2pi]
socialDiveLocations[socialDiveLocations[:,1]<0,1] = socialDiveLocations[socialDiveLocations[:,1]<0,1] + 2 *pi

hista2=np.histogram2d(x=socialDiveLocations[:,0],y=socialDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  

#hista2= hista2/np.max(hista2)
#hista2[1:] =hista2[1:]/np.max(hista2[1:])

#for hrow in range(hista2.shape[0]):
#    hista2[hrow,:] = hista2[hrow,:]/np.max(hista2[hrow,:])
    
size = 8
# make a square figure
#fig,axes = plt.subplots(1,1,figsize=(size, size),subplot_kw=dict(polar=True))
fig1=plt.figure(figsize=(size, size))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2[0:],hista2[0:],lw=0.0,vmin=np.min(hista2),vmax=np.max(hista2),cmap='OrRd')
ax2.yaxis.set_visible(False)

#ax2.set_xticklabels(['0°(front)', '45°', '90°', '135°', '180°(back)', '225°', 
                     #'270°', '315°'])

ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', 
               '135°', '', '225°','270°', '315°'],frac=1.1)

ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', 
                         label='twin', frame_on=False,
                         theta_direction=ax2.get_theta_direction(),
                         theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', 
               '', '180°(back)', '','', ''],frac=1.15)

xbin=np.linspace(-10,10,20)





hista1=np.histogram2d(x=allDiveLocations[:,0],y=allDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  


hista2=np.histogram2d(x=socialDiveLocations[:,0],y=socialDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  
diff = hista2/hista1
#for hrow in range(diff.shape[0]):
#    diff[hrow,:] = diff[hrow,:]/np.max(diff[hrow,:])
    
size = 8
# make a square figure
#fig,axes = plt.subplots(1,1,figsize=(size, size),subplot_kw=dict(polar=True))
fig1=plt.figure(figsize=(size, size))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2[0:],diff,lw=0.5,vmin=0.3,vmax=np.max(diff),cmap='OrRd')
ax2.yaxis.set_visible(False)

#ax2.set_xticklabels(['0°(front)', '45°', '90°', '135°', '180°(back)', '225°', 
                     #'270°', '315°'])

ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', 
               '135°', '', '225°','270°', '315°'],frac=1.1)

ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', 
                         label='twin', frame_on=False,
                         theta_direction=ax2.get_theta_direction(),
                         theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', 
               '', '180°(back)', '','', ''],frac=1.15)

xbin=np.linspace(-10,10,20)