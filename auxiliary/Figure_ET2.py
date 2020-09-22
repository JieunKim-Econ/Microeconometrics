#!/usr/bin/env python
# coding: utf-8

# In[129]:


import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm
import seaborn as sns
import warnings; warnings.filterwarnings(action='once')

import econtools
import econtools.metrics as mt

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[130]:


##============1. Fertility
##====================1-1 Conceptions========================
#Variables
#mesp: Month of birth
#year: Year of birth
#prem: Prematurity indicator
#semanas: Weeks of gestation at birth

etdf=pd.read_stata("data/data_births_20110196.dta")


# In[131]:


#1. Create month of birth variable: (0 = July 2007, 1 = August 2007, etc)

etdf.loc[(etdf['year']==2010), 'm'] = etdf['mesp']+29
etdf.loc[(etdf['year']==2009), 'm'] = etdf['mesp'] + 17 #replace m = mesp + 17 if year==2009
etdf.loc[(etdf['year']==2008), 'm'] = etdf['mesp'] + 5  #replace m = mesp + 5 if year==2008
etdf.loc[(etdf['year']==2007), 'm'] = etdf['mesp'] - 7  #replace m = mesp - 7 if year==2007
etdf.loc[(etdf['year']==2006), 'm'] = etdf['mesp'] - 19 #replace m = mesp - 19 if year==2006
etdf.loc[(etdf['year']==2005), 'm'] = etdf['mesp'] - 31 #replace m = mesp - 31 if year==2005
etdf.loc[(etdf['year']==2004), 'm'] = etdf['mesp'] - 43 #replace m = mesp - 43 if year==2004
etdf.loc[(etdf['year']==2003), 'm'] = etdf['mesp'] - 55 #replace m = mesp - 55 if year==2003
etdf.loc[(etdf['year']==2002), 'm'] = etdf['mesp'] - 67
etdf.loc[(etdf['year']==2001), 'm'] = etdf['mesp'] - 79 #replace m = mesp - 79 if year==2001
etdf.loc[(etdf['year']==2000), 'm'] = etdf['mesp'] - 91 #replace m = mesp - 91 if year==2000


# In[132]:


getdf=etdf.groupby('m',as_index=False)['m'].agg({'n':'count'})


# In[133]:


#getdf


# In[134]:


def figure_ET2(data):
    
    fig = plt.figure(figsize = (10, 5)) 
    plt.grid(True)
    #plt.xlim(-90, 40)
    plt.ylim(0, 52000) 
    birth = list(getdf.n) 
    month = list(getdf.m) 

    # creating the bar plot 
    plt.bar(month, birth, color ='orange',  width = 1) 
    plt.axvline(x=0, color='salmon')

    
    plt.fill_betweenx(y=range(44000), x1=-30,x2=30, alpha=0.2, facecolor='c', label = '60 months around the cutoff')
    plt.fill_betweenx(y=range(44000), x1=-20,x2=20, alpha=0.4, facecolor='c', label = '40 months')
    plt.fill_betweenx(y=range(44000), x1=-9,x2=9, alpha=0.6, facecolor='c', label = '18 motnhs ')
    plt.legend(loc='best')
    
    plt.xlabel("m=month of birth (0=July 2007)") 
    plt.ylabel("No. of Births") 
    plt.title("Figure ET2. Discontinuity check in births") 
    plt.show() 
    
    return
    

