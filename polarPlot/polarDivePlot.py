# -*- coding: utf-8 -*-

import os
import csv
import numpy as np
from datetime import datetime
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
from pymc.Matplot import plot as mcplot
import matplotlib
from matplotlib import transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.stats import binned_statistic_2d
import pandas as pd
import math


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

# time taken from Bayesian inference
maxLag = 1.659979663079878

# build an array to store the dives that resulted in another dive
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


# build an array to store all dives
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


################################
################################
#### END OF DATA PROCESSING ####
################################
################################

################################
################################
########### FIGURES ############
################################
################################

## POLAR PLOT OF ALL DIVES
binn2=7
binn1=24
maxr=75
size = 8

theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
r2 = np.linspace(5, maxr, binn2+1)

# wrap to [0, 2pi]
allDiveLocations[allDiveLocations[:,1]<0,1] = allDiveLocations[allDiveLocations[:,1]<0,1] + 2 *pi
hista1=np.histogram2d(x=allDiveLocations[:,0],y=allDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  



fig1=plt.figure(figsize=(size, size))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2,hista1,lw=0.0,cmap='OrRd')
ax2.yaxis.set_visible(False)


ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)

ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', label='twin', frame_on=False,
                         theta_direction=ax2.get_theta_direction(),
                         theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', '', '180°(back)', '','', ''],frac=1.15)
               
m = plt.cm.ScalarMappable(cmap='OrRd')
m.set_array(hista1)
position=fig1.add_axes([1.1,0.12,0.04,0.8])
cbar = plt.colorbar(m,cax=position)

#cbar=plt.colorbar(im,ticks=[0,0.5, 1])#,cax=position) 
cbar.set_label('Number of dives', rotation=90,fontsize='xx-large',labelpad=15)    


axes=ax2
factor = 1.0
d = axes.get_yticks()[-1] * factor
r_tick_labels = [0] + axes.get_yticks()
r_tick_labels = r_tick_labels[:-1]
r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
#r_ticks = r_ticks[:-1]
theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2

# fixed offsets in x
offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)

# apply these to the data coordinates of the line/ticks
trans_spine = axes.transData + offset_spine
trans_ticklabels = trans_spine + offset_ticklabels
trans_axlabel = trans_spine + offset_axlabel

# plot the 'spine'
axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)

# plot the 'tick labels'
for ii in xrange(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii],
                 ha="right", va="center", clip_on=False,
                 transform=trans_ticklabels)

# plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'bodylengths',rotation='vertical', fontsize='xx-large',
             ha='right', va='center', clip_on=False, transform=trans_axlabel,
             family='Trebuchet MS')
            

fig1.savefig("alldives.png",bbox_inches='tight',dpi=100)



## POLAR PLOT OF SOCIAL DIVES FRACTION


hista1=np.histogram2d(x=allDiveLocations[:,0],y=allDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  
hista2=np.histogram2d(x=socialDiveLocations[:,0],y=socialDiveLocations[:,1],bins=[r2,theta2],normed=0)[0]  

diff = hista2/hista1

fig1=plt.figure(figsize=(size, size))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2[0:],diff,lw=0.5,vmin=0.3,vmax=0.4,cmap='OrRd')
ax2.yaxis.set_visible(False)

ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', '135°', '', '225°','270°', '315°'],frac=1.1)

ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', label='twin', frame_on=False,
                         theta_direction=ax2.get_theta_direction(),
                         theta_offset=ax2.get_theta_offset())
ax1.yaxis.set_visible(False)
ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', '', '180°(back)', '','', ''],frac=1.15)
              
m = plt.cm.ScalarMappable(cmap='OrRd')
m.set_array(diff[:-1,:])
position=fig1.add_axes([1.1,0.12,0.04,0.8])
cbar = plt.colorbar(m,cax=position,ticks=[0.3,0.32,0.34,0.36,0.38,0.4])
cbar.set_clim(0.3,0.4)
#cbar=plt.colorbar(im,ticks=[0,0.5, 1])#,cax=position) 
cbar.set_label('Fraction followed by dive', rotation=90,fontsize='xx-large',labelpad=15)    


axes=ax2
factor = 1.0
d = axes.get_yticks()[-1] * factor
r_tick_labels = [0] + axes.get_yticks()
r_tick_labels = r_tick_labels[:-1]
r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
#r_ticks = r_ticks[:-1]
theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2

# fixed offsets in x
offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)

# apply these to the data coordinates of the line/ticks
trans_spine = axes.transData + offset_spine
trans_ticklabels = trans_spine + offset_ticklabels
trans_axlabel = trans_spine + offset_axlabel

# plot the 'spine'
axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine, clip_on=False)

# plot the 'tick labels'
for ii in xrange(len(r_ticks)):
    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii],
                 ha="right", va="center", clip_on=False,
                 transform=trans_ticklabels)

# plot the 'axis label'
axes.text(theta_axlabel, r_axlabel, 'bodylengths',rotation='vertical', fontsize='xx-large',
             ha='right', va='center', clip_on=False, transform=trans_axlabel,
             family='Trebuchet MS')

fig1.savefig("socialdives.png",bbox_inches='tight',dpi=100)
            
