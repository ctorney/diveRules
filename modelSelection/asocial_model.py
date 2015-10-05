


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


__all__ = ['dives','dvector','intrinsic_rate']

workDir = '/home/ctorney/workspace/diveRules/'
# Define data and stochastics

maxLag = 5



intrinsic_rate = Uniform('intrinsic_rate',lower=0, upper=1)


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



@stochastic(observed=True)
def dives(i=intrinsic_rate,value=dvector):
     def logp(value, i):
        

        svector=np.zeros_like(value) #social vector

        asocdiv = value[svector==0]
        return (np.sum(np.log((1.0-i)**(1-asocdiv)*(i**asocdiv))) )

        

    
