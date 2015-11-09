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
# Define data and stochastics



allDF = pd.DataFrame()
for trial in np.arange(0,45):
    #trial = 0
    fileimportname= workDir + '/data/tdata'+str(trial)+'.csv'
    if os.path.isfile(fileimportname):
        df = pd.read_csv(fileimportname)
        df['trial']=trial
        allDF = allDF.append(df[(df['dive']==0)|(df['time']==df['time_dive'])])

           
allData = allDF.values

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


# build an array to store all observed dives and their time, distance and angle from the focal individual
locations = np.zeros((0,3)).astype(np.float32) # time, dist, angle
for thisRow in range(dsize):
    thisTime = allData[thisRow,0]        
    thisIndex = allData[thisRow,1]        
    thisX = allData[thisRow,2]
    thisY = allData[thisRow,3]
    thisAngle = (allData[thisRow,4])
    thisTrial = allData[thisRow,9]
    thisTrack = allData[(allData[:,9]==thisTrial)&(allData[:,1]==thisIndex),:]
    window = allData[(allData[:,0]==thisTime)&(allData[:,9]==thisTrial)&(allData[:,1]!=thisIndex),:]
    ncount = 0
    rowLoc = np.zeros((0,3)).astype(np.float32) 
    for w in window:
        xj = w[2]
        yj = w[3]
        tj = w[0]
        jAngle = (w[4])
        texj = w[7]
        r = (((thisX-xj)**2+(thisY-yj)**2))**0.5
        dx = xj - thisX
        dy = yj - thisY
        angle = math.atan2(dy,dx)
        angle = angle - thisAngle
        jAngle = jAngle - thisAngle
        theta = math.atan2(math.sin(angle), math.cos(angle))
        jHeading  = math.atan2(math.sin(jAngle), math.cos(jAngle))
        rowLoc = np.vstack((rowLoc,[r, theta, jHeading]))
    locations = np.vstack((locations,rowLoc))

        
binn2=10
binn1=16
maxr=30

theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
r2 = np.linspace(0, maxr, binn2+1)
Ar = np.linspace(2**2, maxr**2, binn2+1)
#r2 = Ar**0.5
#
locations[locations[:,1]<0,1] = locations[locations[:,1]<0,1] + 2 *pi
hista2=np.histogram2d(x=locations[:,0],y=locations[:,1],bins=[r2,theta2],normed=0)[0]  
areas=np.zeros((0,binn2))
for i in range(binn1):
    areas = np.vstack((areas,pi*(r2[1:]**2-r2[0:-1]**2)/(binn1)))

areas = areas.T
        
#hista2[0,:]=0
#hista2 = hista2/areas # hista2/np.max(hista2)
hista2 =hista2/np.max(hista2)

#for i in range(binn2):
#    imin = np.min(hista2[i,:])
#    imax = np.max(hista2[i,:])
#    hista2[i,:]= hista2[i,:]- imin
#    hista2[i,:] = hista2[i,:] /(imax-imin)
size = 8
# make a square figure
#fig,axes = plt.subplots(1,1,figsize=(size, size),subplot_kw=dict(polar=True))
fig1=plt.figure(figsize=(size, size))
ax2=plt.subplot(projection="polar",frameon=False)
im=ax2.pcolormesh(theta2,r2,hista2,lw=0.0,vmin=0,vmax=1.0)
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

locations3 =  locations[locations[:,0]>0.0,:]
locations2=np.zeros_like(locations3)
locations2[:,0]=np.cos(locations3[:,1])*locations3[:,0]
locations2[:,1]=np.sin(locations3[:,1])*locations3[:,0]
fig1=plt.figure(figsize=(size, size))
hista2=np.histogram2d(x=locations2[:,0],y=locations2[:,1],bins=[xbin,xbin],normed=0)[0]
hista2[9,9]=0
hista2 =hista2/np.median(hista2)
ax2=plt.subplot(frameon=False)
im=ax2.pcolormesh(xbin,xbin,hista2,lw=0.0,vmin=0,vmax=2.0)
plt.figure()
plt.hist(locations[:,1],100)


#
#workDir = '/home/ctorney/workspace/diveRules/'
#histaall=np.zeros([binn2,binn1])
#angdataalls=np.zeros([binn2,binn1])
#angdataallc=np.zeros([binn2,binn1])
#angvarall=np.zeros([binn2,binn1])
#meanalldist=np.zeros([45,1])
#sdalldist=np.zeros([45,1])
#theta2 = np.linspace(0.0,2.0 * np.pi, binn1+1)
#r2 = np.linspace(0, maxr, binn2+1)
#grid_r, grid_theta = np.meshgrid(r2, theta2)
#nozero=(grid_theta!=0)&(grid_r!=0)
#grid_r=np.reshape(grid_r[nozero],[binn1,binn2]).T
#grid_theta=np.reshape(grid_theta[nozero],[binn1,binn2]).T
#goodtrial=0
#
#finalexporthist=os.path.join(os.getcwd(),('reldata.csv'))
#finalexportang=os.path.join(os.getcwd(),('relangdata.csv'))
#finalexportmeandist=os.path.join(os.getcwd(),('meandist.csv'))
#
#for trial in np.arange(0,45,1):
#        
#        
#        fileimportname= workDir + '/data/data/'+str(trial) + '/tdata'+str(trial)+'.csv'
#        
#        filexportname=workDir + 'reldata'+str(trial)+'.csv'
#        filexportname2=workDir + 'relangdata'+str(trial)+'.csv'
#       # print (trial)
#        if os.path.isfile(fileimportname):
#            with open (fileimportname) as tdata:
#                reader=csv.reader(tdata)
#                for row in reader:
#                    data=list(reader)
#                    result=np.array(data).astype('float')
#
#            inc=np.unique(result[:,0])[1]-np.unique(result[:,0])[0]
#            hista=np.zeros([binn2,binn1])
#            angdatas=np.zeros([binn2,binn1])
#            angdatac=np.zeros([binn2,binn1])
#            angvar=np.zeros([binn2,binn1])
#            
#            meandist=np.zeros([1,1])
#            sddist=np.zeros([1,1])
#            goodtrial=goodtrial+1
#            for time in np.arange(min(result[:,0]),max(result[:,0]),inc):
#                dists3=np.zeros([1,1])
#                sddists3=np.zeros([1,1])
#                relmatrix=np.empty([0,4])
#
#                window=(result[:,0]==time)&np.isnan(result[:,6]) 
#                index=np.linspace(1,len(result[window,0]),len(result[window,0]))
#                dists2=np.zeros(index.shape)    
#                dists2=np.zeros(index.shape) 
#                sddists2=np.zeros(index.shape)
#                did=result[window,1]
#                x=result[window,2]
#                y=result[window,3]
#                a=result[window,4]
#
#                if (index.size>1):
#                    
#                    for i in index:
#                        
#                        didi=did[index==i]
#                        xi=x[index==i]
#                        yi=y[index==i]
#                        ai=a[index==i]
#                        didj=did[(index!=i)]
#                        xj=x[(index!=i)]
#                        yj=y[(index!=i)]
#                        aj=a[(index!=i)]
#                        
#                        dists2[index==i]=(((xj-xi)**2+(yj-yi)**2)**0.5).min()
#                        sddists2[index==i]=(((xj-xi)**2+(yj-yi)**2)**0.5).min()
#                        
#                        x1 = -1000*np.cos(np.radians(ai))+xi
#                        x2 = 1000*np.cos(np.radians(ai))+xi
#                        y1 = -1000*np.sin(np.radians(ai))+yi
#                        y2 = 1000*np.sin(np.radians(ai))+yi
#                        
#                        u=(((xj - x1) * (x2 - x1)) + ((yj - y1) * (y2 - y1)))
#                        u = u / ((((x2-x1)**2+(y2-y1)**2)**0.5)*(((x2-x1)**2+(y2-y1)**2)**0.5))
#                        ix = x1 + (u * (x2 - x1))
#                        iy = y1 + u * ((y2 - y1))
#                        #swapped these around (might fuck it up)
#                        relx=(((xj-ix)**2+(yj-iy)**2)**0.5)
#                        rely=(((xi-ix)**2+(yi-iy)**2)**0.5)
#                        
#                        if (ai>-90)&(ai<90):
#                            negx=yj>iy;    
#                            negy=ix<xi;
#                        elif (ai>90)|(ai<-90):
#                            negx=yj<iy;    
#                            negy=ix>xi;
#                        elif ai==90:
#                            negx=xj<ix;
#                            negy=iy<yi;
#                        elif ai==-90:
#                            negx=xj>ix;
#                            negy=iy>yi;
#                        
#                        relx[negx]=-(relx[negx])   
#                        rely[negy]=-(rely[negy])
#                        reldistfw=((relx[-negy])**2+(rely[-negy])**2)**0.5
#                        reldistbk=((relx[negy])**2+(rely[negy])**2)**0.5
#                        
#                        #if reldistbk.size==0:
#                            #dists2[index==i]=np.nan
#                            #sddists2[index==i]==np.nan
#                        #else:    
#                            #dists2[index==i]=reldistbk.min()
#                            #sddists2[index==i]=reldistbk.min()
#                        
#                        #get relative angles and convert to 360 degrees where direction of movement
#                        #is 0
#    
#                        aj[aj<0]=aj[aj<0]+360
#                        ai[ai<0]=ai[ai<0]+360
#                        
#                        rela=aj-ai
#                        rela[rela<0]=rela[rela<0]+360
#                        
#                        relmatrix2=np.transpose(np.vstack((np.repeat(i,len(relx)),relx,rely,rela)))
#                        relmatrix=np.concatenate((relmatrix,relmatrix2),axis=0)
#                #2d hisogram
#                if relmatrix.shape[0]>0:
#                    theta,r=convert_to_polar(relmatrix[:,1], relmatrix[:,2])
#                    theta=theta+np.pi
#                    points=np.transpose(np.vstack([r,theta]))
#                    
#
#                    
#                    
#
#                    hista2=np.histogram2d(x=points[:,0],y=points[:,1],bins=[r2,theta2],normed=0)[0]               
#                    angdata2s=np.zeros([binn2,binn1])
#                    angdata2c=np.zeros([binn2,binn1])
#                    angvar2=np.zeros([binn2,binn1])
#                    
#                    for i in np.arange(0,points.shape[0]-1,1):
#                        cellselect=((points[i,0]<grid_r)&(grid_r-(maxr/r2.size)<points[i,0]))&((points[i,1]<grid_theta)&(grid_theta-((2.0 * np.pi)/theta2.size)<points[i,1]))
#                        angdata2s[cellselect]=angdata2s[cellselect]+np.sin(np.radians(relmatrix[i,3]))
#                        angdata2c[cellselect]=angdata2c[cellselect]+np.cos(np.radians(relmatrix[i,3]))
#                    
#
#                    
#                    if np.sum(hista2)>0:
#       #                 angvar2[hista2>0]=1-(((angdata2s[hista2>0]**2+angdata2c[hista2>0]**2)**0.5)/hista2[hista2>0])                      
#
#                        angdata2s[hista2>0]=angdata2s[hista2>0]/hista2[hista2>0]         
#                        angdata2c[hista2>0]=angdata2c[hista2>0]/hista2[hista2>0]         
#                        angdata2=np.degrees(np.arctan2(angdata2s,angdata2c))
#                        angdata2[angdata2<0]=angdata2[angdata2<0]+360
#                        angdatas=angdatas+np.sin(np.radians(angdata2))
#                        angdatac=angdatac+np.cos(np.radians(angdata2))                        
#       #                 angvar=angvar+angvar2
#                        hista2[hista2>0]=hista2[hista2>0]/np.amax(hista2[hista2>0])                   
#                        hista=hista+hista2
#                if sum(dists2)>0:
#                    dists=np.mean(dists2)
#                    dists3=dists+dists3
#                    sddists=np.std(sddists2)
#                    sddists3=sddists+sddists3
#                if dists3>0:
#                    meandist=meandist+dists3
#                    sddist=sddist+sddists3
#                    
#
#            #NORMALISE!
#            if np.sum(hista)>0:
#                hista[hista>0]=hista[hista>0]/np.amax(hista[hista>0])
#                np.savetxt(filexportname,hista,delimiter=',')
#            
#            if np.sum(angdatas)!=0:    
#                np.savetxt(filexportname2,angdatas,delimiter=',')
#                angvar[angdatas!=0]=1-((((angdatas[angdatas!=0]**2)+(angdatac[angdatas!=0]**2))**0.5)/np.arange(min(result[:,0]),max(result[:,0]),inc).size)
##                angvar[angvar!=0]=angvar[angvar!=0]/np.arange(min(result[:,0]),max(result[:,0]),inc).size
#                angdatas[angdatas!=0]=angdatas[angdatas!=0]/np.arange(min(result[:,0]),max(result[:,0]),inc).size
#                angdatac[angdatac!=0]=angdatac[angdatac!=0]/np.arange(min(result[:,0]),max(result[:,0]),inc).size
#                angdata=np.degrees(np.arctan2(angdatas,angdatac))
#                angdata[angdata<0]=angdata[angdata<0]+360
#                
#                
#                
#                angvarall=angvarall+angvar
#                angdataalls=angdataalls+np.sin(np.radians(angdata))
#                angdataallc=angdataallc+np.cos(np.radians(angdata))                
#            
#            histaall=histaall+hista
#            
#            if meandist>0:
#                meandist=meandist/np.arange(min(result[:,0]),max(result[:,0]),inc).size
#                meanalldist[trial]=meandist
#
#                sddist=sddist/np.arange(min(result[:,0]),max(result[:,0]),inc).size
#                sdalldist[trial]=sddist
#
#
##angvarall[angdataalls!=0]=1-((((angdataalls[angdataalls!=0]**2)+(angdataallc[angdataalls!=0]**2))**0.5)/goodtrial)
#angvarall[angvarall!=0]=angvarall[angvarall!=0]/goodtrial
#angdataalls[angdataalls!=0]=angdataalls[angdataalls!=0]/goodtrial
#angdataallc[angdataallc!=0]=angdataallc[angdataallc!=0]/goodtrial
#angdataall=np.degrees(np.arctan2(angdataalls,angdataallc))
#angdataall[angdataall<0]=angdataall[angdataall<0]+360
#histaall[histaall>0]=histaall[histaall>0]/np.amax(histaall[histaall>0])
#meanalldist=np.mean(meanalldist[meanalldist>0])
#sdalldist=np.mean(sdalldist[sdalldist>0])
#np.savetxt(finalexporthist,histaall,delimiter=',')
#np.savetxt(finalexportang,angdataall,delimiter=',')
#np.savetxt(finalexportmeandist,[meanalldist,sdalldist],delimiter=',')
#                    
#                    
#
#
##                    plt.figure(figsize=(5,5))
##                    plt.plot([x1,x2],[y1,y2])
##                    plt.scatter(ix,iy,color='green')
##                    plt.scatter(xi,yi,color='red')
##                    plt.scatter(xj,yj,color='blue')
##
###                    #for j in np.arange(0,len(index[index!=i]),1):
###                        #plt.annotate(str(index[index!=i][j]),xy=(ix[j],iy[j]))
##                    plt.xlim(-100,100)
##                    plt.ylim(-100,100)
##                    plt.show()
###                    
##                    plt.figure(figsize=(5,5))
##                    plt.scatter(relx[0],rely[0])
##                    plt.scatter(0,0,color='red')
##                    plt.xlim(-100,100)
##                    plt.ylim(-100,100)
##                    plt.show() 
#
#histaall = np.loadtxt('reldata.csv',delimiter=',')
#angdataall = np.loadtxt('relangdata.csv',delimiter=',')
#
#r3=r2[0:r2.size-1]+((r2[1]-r2[0])/2)
#theta3=theta2[0:theta2.size-1]+((theta2[1]-theta2[0])/2)
#grid_r3, grid_theta3 = np.meshgrid(r3, theta3)
#
#arrowbasex, arrowbasey=convert_from_polar(grid_theta3.T,grid_r3.T)
#arrowbasex=arrowbasex-2
#arrowendx=(3.5*np.cos(np.radians(angdataall)))+arrowbasex
#arrowendy=(3.5*np.sin(np.radians(angdataall)))+arrowbasey
#
#arrowbasetheta,arrowbaser=convert_to_polar(arrowbasex,arrowbasey)
#arrowendtheta,arrowendr=convert_to_polar(arrowendx,arrowendy)
#
#arrowbasetheta=np.reshape(arrowbasetheta,arrowbasetheta.size)
#arrowendtheta=np.reshape(arrowendtheta,arrowbasetheta.size)  
#
##arrowendx=np.reshape(arrowendx,arrowbasex.size)
##arrowendy=np.reshape(arrowendy,arrowbasex.size)
#
##arrowbasex=np.reshape(arrowbasex,arrowbasex.size)
##arrowbasey=np.reshape(arrowbasey,arrowbasex.size)
#
#arrowbaser=np.reshape(arrowbaser,arrowbasetheta.size)
#arrowendr=np.reshape(arrowendr,arrowbasetheta.size)
#
#noplot=grid_r<=r2[1]
#histaall=np.ma.masked_where((noplot==1),histaall)
#angvarall=np.ma.masked_where((noplot==1),angvarall)
#
#edgeseq=['black']*histaall.size                
#edgeseq[0:noplot[noplot].size-1]=['none']*noplot[noplot].size
##i=28
#
#size = 8
## make a square figure
##fig,axes = plt.subplots(1,1,figsize=(size, size),subplot_kw=dict(polar=True))
#fig1=plt.figure(figsize=(size, size))
#ax2=plt.subplot(projection="polar",frameon=False)
#im=ax2.pcolormesh(theta2,r2,histaall,edgecolors=edgeseq,lw=0.05,vmin=0,vmax=1)
#ax2.yaxis.set_visible(False)
#
##ax2.set_xticklabels(['0°(front)', '45°', '90°', '135°', '180°(back)', '225°', 
#                     #'270°', '315°'])
#
#ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', 
#               '135°', '', '225°','270°', '315°'],frac=1.1)
#
#ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', 
#                         label='twin', frame_on=False,
#                         theta_direction=ax2.get_theta_direction(),
#                         theta_offset=ax2.get_theta_offset())
#ax1.yaxis.set_visible(False)
#ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', 
#               '', '180°(back)', '','', ''],frac=1.15)
#
##plt.scatter(0,0,color='white')
##plt.scatter(arrowbasetheta,arrowbaser)
##plt.scatter(arrowendtheta,arrowendr,color='red')
##plottheta=np.zeros(arrowendr.size)
##neg1=arrowbasetheta>arrowendtheta
##plottheta[neg1]=arrowendtheta[neg1]-arrowbasetheta[neg1]
##neg2=arrowendtheta>arrowbasetheta
##plottheta[neg2]=arrowendtheta[neg2]-arrowbasetheta[neg2]
#
##plotr=arrowendr-arrowbaser
##for i in np.linspace(0,arrowendtheta.size-1,arrowendtheta.size)[-np.reshape(noplot,arrowbasetheta.size)]:  
###for i in [192]:
##   #arr1=plt.arrow(arrowbasetheta[i], arrowbaser[i],plottheta[i],plotr[i],length_includes_head=True)
##    ax2.annotate("",
##            xy=(arrowendtheta[i], arrowendr[i]), xycoords='data',
##            xytext=(arrowbasetheta[i], arrowbaser[i]), textcoords='data',
##            arrowprops=dict(arrowstyle="->",
##                            connectionstyle="arc3"),
##            )
#axes=ax2            
#factor = 0.98
#d = axes.get_yticks()[-1] * factor
#r_tick_labels = [0] + axes.get_yticks()
#r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
#r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#
## fixed offsets in x
#offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
#offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
#offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
#trans_spine = axes.transData + offset_spine
#trans_ticklabels = trans_spine + offset_ticklabels
#trans_axlabel = trans_spine + offset_axlabel
#
## plot the 'spine'
#axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine,
#             clip_on=False)
#
## plot the 'tick labels'
#for ii in xrange(len(r_ticks)):
#    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii],
#                 ha="right", va="center", clip_on=False,
#                 transform=trans_ticklabels)
#
## plot the 'axis label'
#axes.text(theta_axlabel, r_axlabel, 'bodylengths',rotation='vertical', fontsize='xx-large',
#             ha='right', va='center', clip_on=False, transform=trans_axlabel,
#             family='Trebuchet MS')
#             
#position=fig1.add_axes([1.1,0.12,0.04,0.8])
#cbar=plt.colorbar(im,ticks=[0,0.5, 1],cax=position) 
#cbar.set_label('Density of neighbours', rotation=90,fontsize='xx-large',labelpad=15)            
#fig1.savefig("relpos.png",bbox_inches='tight',dpi=100)
#plt.show()  
#
####### CIRCULAR VARIANCE PLOT
#
#fig1=plt.figure(figsize=(size, size))
#ax2=plt.subplot(projection="polar",frameon=False)
#im=ax2.pcolormesh(theta2,r2,angvarall,edgecolors=edgeseq,lw=0.05,vmin=0,vmax=1)
#ax2.yaxis.set_visible(False)
#
##ax2.set_xticklabels(['0°(front)', '45°', '90°', '135°', '180°(back)', '225°', 
#                     #'270°', '315°'])
#
#ax2.set_thetagrids(angles=np.arange(0,360,45),labels=['', '45°', '90°', 
#               '135°', '', '225°','270°', '315°'],frac=1.1)
#
#ax1 = ax2.figure.add_axes(ax2.get_position(), projection='polar', 
#                         label='twin', frameon=False,
#                         theta_direction=ax2.get_theta_direction(),
#                         theta_offset=ax2.get_theta_offset())
#ax1.yaxis.set_visible(False)
#ax1.set_thetagrids(angles=np.arange(0,360,45),labels=['0°(front)', '', '', 
#               '', '180°(back)', '','', ''],frac=1.15)
#
##plt.scatter(0,0,color='white')
##plt.scatter(arrowbasetheta,arrowbaser)
##plt.scatter(arrowendtheta,arrowendr,color='red')
##plottheta=np.zeros(arrowendr.size)
##neg1=arrowbasetheta>arrowendtheta
##plottheta[neg1]=arrowendtheta[neg1]-arrowbasetheta[neg1]
##neg2=arrowendtheta>arrowbasetheta
##plottheta[neg2]=arrowendtheta[neg2]-arrowbasetheta[neg2]
#
##plotr=arrowendr-arrowbaser
#for i in np.linspace(0,arrowendtheta.size-1,arrowendtheta.size)[-np.reshape(noplot,arrowbasetheta.size)]:  
##for i in [192]:
#   #arr1=plt.arrow(arrowbasetheta[i], arrowbaser[i],plottheta[i],plotr[i],length_includes_head=True)
#    ax2.annotate("",
#            xy=(arrowendtheta[i], arrowendr[i]), xycoords='data',
#            xytext=(arrowbasetheta[i], arrowbaser[i]), textcoords='data',
#            arrowprops=dict(arrowstyle="->",
#                            connectionstyle="arc3"),
#            )
#axes=ax2            
#factor = 0.98
#d = axes.get_yticks()[-1] * factor
#r_tick_labels = [0] + axes.get_yticks()
#r_ticks = (np.array(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_ticks = np.arcsin(d / r_ticks) + np.pi / 2
#r_axlabel = (np.mean(r_tick_labels) ** 2 + d ** 2) ** 0.5
#theta_axlabel = np.arcsin(d / r_axlabel) + np.pi / 2
#
## fixed offsets in x
#offset_spine = transforms.ScaledTranslation(-100, 0, axes.transScale)
#offset_ticklabels = transforms.ScaledTranslation(-10, 0, axes.transScale)
#offset_axlabel = transforms.ScaledTranslation(-40, 0, axes.transScale)
#
## apply these to the data coordinates of the line/ticks
#trans_spine = axes.transData + offset_spine
#trans_ticklabels = trans_spine + offset_ticklabels
#trans_axlabel = trans_spine + offset_axlabel
#
## plot the 'spine'
#axes.plot(theta_ticks, r_ticks, '-_k', transform=trans_spine,
#             clip_on=False)
#
## plot the 'tick labels'
#for ii in xrange(len(r_ticks)):
#    axes.text(theta_ticks[ii], r_ticks[ii], "%.0f" % r_tick_labels[ii],
#                 ha="right", va="center", clip_on=False,
#                 transform=trans_ticklabels)
#
## plot the 'axis label'
#axes.text(theta_axlabel, r_axlabel, 'bodylengths',rotation='vertical', fontsize='xx-large',
#             ha='right', va='center', clip_on=False, transform=trans_axlabel,
#             family='Trebuchet MS')
#             
#position=fig1.add_axes([1.1,0.12,0.04,0.8])
#cbar=plt.colorbar(im,ticks=[0,0.5, 1],cax=position) 
#cbar.set_label('Circular variance', rotation=90,fontsize='xx-large',labelpad=15)            
#fig1.savefig("relpos.png",bbox_inches='tight',dpi=100)
#plt.show()  
#
#
#
##ax1.arrow(arrowbasetheta[i], arrowbaser[i], arrowendx[i], arrowendy[i], head_width=0.05, head_length=0.1, fc='k', ec='k')
#                
#
##plt.scatter(arrowbasex,arrowbasey)
##plt.scatter(arrowendx,arrowendy,color='red')
##plt.show()