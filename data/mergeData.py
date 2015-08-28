


import os
import csv
import math
import numpy as np
from datetime import datetime
from numpy import array, empty
from numpy.random import randint, rand
import numpy as np
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt






#os.chdir("D:\\shags\\")
allData=np.zeros([0,7]) #dive vector
for trial in np.arange(0,45):
    folderpath=os.path.join (os.getcwd()+'/data3/',str(trial) )
    fileimportname =  folderpath + '/tdata'+str(trial)+'.csv'
    if os.path.isfile(fileimportname):
        df3 = pd.read_csv(fileimportname, header=None)
        #folderpath=os.path.join (os.getcwd()+'/data/',str(trial) )
        #fileimportname =  folderpath + '/tdata'+str(trial)+'.csv'
        #df2 = pd.read_csv(fileimportname, header=None)
        #df3 = pd.concat([df,df2[6]],axis=1)
        df3 = df3.fillna(0)
        df3.columns = ['time','id','x','y','angle','dive','time_dive','exact_time','speed']
        df3.to_csv('tdata' + str(trial) + '.csv', index=False)
           


