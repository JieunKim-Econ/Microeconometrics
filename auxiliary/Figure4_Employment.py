#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import localreg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import econtools
import econtools.metrics as mt


# In[2]:


#*****************Female Labor Supply*********************************
ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[3]:


#1.Control variables
#gen m2 = m*m
ls['m2']=ls['m']*ls['m']

#No father present.*/
#gen nodad=0
#replace nodad=1 if dadid==0
ls['nodad']=0
ls.loc[ls['dadid']==0, 'nodad']=1


# Mother not married 
#gen smom=0
#replace smom=1 if eciv!=2
ls['smom']=0
ls.loc[ls['eciv']!=2, 'smom']=1

#Mother single.*/
#gen single=0
#replace single=1 if eciv==1
ls['single']=0
ls.loc[ls['eciv']==1, 'single']=1

#Mother separated or divorced.*/
#gen sepdiv=0
#replace sepdiv=1 if eciv==4
ls['sepdiv']=0
ls.loc[ls['eciv']==4, 'sepdiv']=1

# No partner in the household.*/
#gen nopart=0
#replace nopart=1 if partner==0
ls['nopart']=0
ls.loc[ls['partner']==0, 'nopart']=1


# In[4]:


#***************Probability of the mother being in the maternity leave period at the time of the interview
#gen pleave=0
#replace pleave=0.17 if (q==1 & m==2) | (q==2 & m==5) | (q==3 & m==8) | (q==4 & m==11)
#replace pleave=0.5 if (q==1 & m==3) | (q==2 & m==6) | (q==3 & m==9) | (q==4 & m==12)
#replace pleave=0.83 if (q==1 & m==4) | (q==2 & m==7) | (q==3 & m==10) | (q==4 & m==13)
#replace pleave=1 if (q==1 & m>4 & m<9) | (q==2 & m>7 & m<12) | (q==3 & m>10 & m<15) | (q==4 & m>13)

ls['pleave']=0

ls.loc[(ls['q']==1) & (ls['m']==2)|(ls['q']==2) & (ls['m']==5)|(ls['q']==3) & (ls['m']==8)|(ls['q']==4) & (ls['m']==11) ,'pleave']=0.17
ls.loc[((ls['q']==1) & (ls['m']==3)) | ((ls['q']==2) & (ls['m']==6))  | ((ls['q']==3) & (ls['m']==9)) |((ls['q']==4) & (ls['m']==12)), 'pleave'] = 0.5
ls.loc[((ls['q']==1) & (ls['m']==4)) | ((ls['q']==2) & (ls['m']==7)) | ((ls['q']==3) & (ls['m']==10))  | ((ls['q']==4) & (ls['m']==13)), 'pleave'] = 0.83
ls.loc[((ls['q']==1) & (ls['m']>4) & (ls['m']<9)) | ((ls['q']==2) & (ls['m']>7) & (ls['m']<12)) | ((ls['q']==3) & (ls['m']>10) & (ls['m']<15))| ((ls['q']==4) & (ls['m']>13)), 'pleave'] = 1


# In[5]:


#create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2 and so on
for j in range(1,5):
    ls['iq_'+str(j)] = 0
    for i in range(len(ls)):
        if ls.loc[i,'q'] == j:
            ls.loc[i, 'iq_'+str(j)] = 1


# In[6]:


#Create interaction dummies
ls['ipost_1']=ls['post']*ls['m']
ls['ipost_2'] =ls['post']*ls['m2']


# In[7]:


#'work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'
gbm_ls=ls.groupby(['m'], as_index=False).agg({'work':'mean', 'work2':'mean','post':'mean','m2':'mean','ipost_1':'mean','ipost_2':'mean','age':'mean','age2':'mean',
                                              'age3':'mean', 'immig':'mean','primary':'mean','hsgrad':'mean','univ':'mean','sib':'mean',
                                              'pleave':'mean','iq_2':'mean','iq_3':'mean','iq_4':'mean'})

#gbm_ls


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


#**************************RRD: Linear Regression******************************************************************************/
##*******************************Working last week

#Create interaction dummies
#ls['ipost_1']=ls['post']*ls['m']
#ls['ipost_2'] =ls['post']*ls['m2']

#create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2 and so on
#for j in range(1,5):
    #df_ls['iq_'+str(j)] = 0
    #for i in range(len(df_ls)):
        #if df_ls.loc[i,'q'] == j:
            #df_ls.loc[i, 'iq_'+str(j)] = 1

#xi: reg work post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
pY_ls= ['work','work2']
pX_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
p_ls_index = gbm_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

result_ls=mt.reg(p_ls_index[(gbm_ls['m']> -30) & (gbm_ls['m']< 12)], 'work', ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
predict_work = result_ls.yhat


# In[9]:


predict_work


# In[10]:


predict_plot_ls = pd.DataFrame(predict_work)
predict_plot_ls.rename(columns={0:'predict_work'}, inplace=True)
#predict_plot_gast


# In[11]:


gbm_ls.loc[gbm_ls['m']==-29, 'predict_work']= predict_plot_ls.iloc[0,0]
gbm_ls.loc[gbm_ls['m']==-28, 'predict_work']= predict_plot_ls.iloc[1,0]
gbm_ls.loc[gbm_ls['m']==-27, 'predict_work']= predict_plot_ls.iloc[2,0]
gbm_ls.loc[gbm_ls['m']==-26, 'predict_work']= predict_plot_ls.iloc[3,0]
gbm_ls.loc[gbm_ls['m']==-25, 'predict_work']= predict_plot_ls.iloc[4,0]
gbm_ls.loc[gbm_ls['m']==-24, 'predict_work']= predict_plot_ls.iloc[5,0]
gbm_ls.loc[gbm_ls['m']==-23, 'predict_work']= predict_plot_ls.iloc[6,0]
gbm_ls.loc[gbm_ls['m']==-22, 'predict_work']= predict_plot_ls.iloc[7,0]
gbm_ls.loc[gbm_ls['m']==-21, 'predict_work']= predict_plot_ls.iloc[8,0]
gbm_ls.loc[gbm_ls['m']==-20, 'predict_work']= predict_plot_ls.iloc[9,0]
gbm_ls.loc[gbm_ls['m']==-19, 'predict_work']= predict_plot_ls.iloc[10,0]
gbm_ls.loc[gbm_ls['m']==-18, 'predict_work']= predict_plot_ls.iloc[11,0]
gbm_ls.loc[gbm_ls['m']==-17, 'predict_work']= predict_plot_ls.iloc[12,0]
gbm_ls.loc[gbm_ls['m']==-16, 'predict_work']= predict_plot_ls.iloc[13,0]
gbm_ls.loc[gbm_ls['m']==-15, 'predict_work']= predict_plot_ls.iloc[14,0]
gbm_ls.loc[gbm_ls['m']==-14, 'predict_work']= predict_plot_ls.iloc[15,0]
gbm_ls.loc[gbm_ls['m']==-13, 'predict_work']= predict_plot_ls.iloc[16,0]
gbm_ls.loc[gbm_ls['m']==-12, 'predict_work']= predict_plot_ls.iloc[17,0]
gbm_ls.loc[gbm_ls['m']==-11, 'predict_work']= predict_plot_ls.iloc[18,0]
gbm_ls.loc[gbm_ls['m']==-10, 'predict_work']= predict_plot_ls.iloc[19,0]
gbm_ls.loc[gbm_ls['m']==-9, 'predict_work']= predict_plot_ls.iloc[20,0]
gbm_ls.loc[gbm_ls['m']==-8, 'predict_work']= predict_plot_ls.iloc[21,0]
gbm_ls.loc[gbm_ls['m']==-7, 'predict_work']= predict_plot_ls.iloc[22,0]
gbm_ls.loc[gbm_ls['m']==-6, 'predict_work']= predict_plot_ls.iloc[23,0]
gbm_ls.loc[gbm_ls['m']==-5, 'predict_work']= predict_plot_ls.iloc[24,0]
gbm_ls.loc[gbm_ls['m']==-4, 'predict_work']= predict_plot_ls.iloc[25,0]
gbm_ls.loc[gbm_ls['m']==-3, 'predict_work']= predict_plot_ls.iloc[26,0]
gbm_ls.loc[gbm_ls['m']==-2, 'predict_work']= predict_plot_ls.iloc[27,0]
gbm_ls.loc[gbm_ls['m']==-1, 'predict_work']= predict_plot_ls.iloc[28,0]

gbm_ls.loc[gbm_ls['m']==0, 'predict_work']= predict_plot_ls.iloc[29,0]
gbm_ls.loc[gbm_ls['m']==1, 'predict_work']= predict_plot_ls.iloc[30,0]
gbm_ls.loc[gbm_ls['m']==2, 'predict_work']= predict_plot_ls.iloc[31,0]
gbm_ls.loc[gbm_ls['m']==3, 'predict_work']= predict_plot_ls.iloc[32,0]
gbm_ls.loc[gbm_ls['m']==4, 'predict_work']= predict_plot_ls.iloc[33,0]
gbm_ls.loc[gbm_ls['m']==5, 'predict_work']= predict_plot_ls.iloc[34,0]
gbm_ls.loc[gbm_ls['m']==6, 'predict_work']= predict_plot_ls.iloc[35,0]
gbm_ls.loc[gbm_ls['m']==7, 'predict_work']= predict_plot_ls.iloc[36,0]
gbm_ls.loc[gbm_ls['m']==8, 'predict_work']= predict_plot_ls.iloc[37,0]
gbm_ls.loc[gbm_ls['m']==9, 'predict_work']= predict_plot_ls.iloc[38,0]
gbm_ls.loc[gbm_ls['m']==10, 'predict_work']= predict_plot_ls.iloc[39,0]
gbm_ls.loc[gbm_ls['m']==11, 'predict_work']= predict_plot_ls.iloc[40,0]
#gbm_ls.loc[gbm_ls['m']==12, 'predict_work']= predict_plot_ls.iloc[41,0]
#gbm_ls.loc[gbm_ls['m']==13, 'predict_work']= predict_plot_ls.iloc[42,0]
#gbm_ls.loc[gbm_ls['m']==14, 'predict_work']= predict_plot_ls.iloc[43,0]
#gbm_ls.loc[gbm_ls['m']==15, 'predict_work']= predict_plot_ls.iloc[44,0]
#gbm_ls.loc[gbm_ls['m']==16, 'predict_work']= predict_plot_ls.iloc[45,0]
#gbm_ls.loc[gbm_ls['m']==17, 'predict_work']= predict_plot_ls.iloc[46,0]


# In[12]:


gbm_ls.loc[gbm_ls['m']>=12, 'predict_work'] = np.nan


# In[13]:


def plot_RRD_curve_work(data):
    plt.grid(True)
    
    ls_p1 = gbm_ls[['predict_work','m']]
    ls_plot = ls_p1.dropna()
    ls_untreat = ls_plot[ls_plot['m'] < 0]
    ls_treat = ls_plot[ls_plot['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),ls_untreat['predict_work'], 2))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,12,1),ls_treat['predict_work'], 2))
    plt.plot(np.arange(0,12,1), poly_fit(np.arange(0,12,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 10, 10)
    plt.ylim(0.2, 0.6, 0.1)
    
    return


# In[14]:


#plot_RRD_curve_work(gbm_ls)


# In[15]:


def plot_figure4(data):
    work_plot = gbm_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
    plt.xlim(-30, 10, 10)
    plt.ylim(0.2, 0.6, 0.1)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Proportion working by month of birth, LFS 2008')
    plt.plot(gbm_ls.m, gbm_ls.work, 'o')
    plt.grid(True)
    plot_RRD_curve_work(data)

    plt.title("Figure 4. Proportion working by month of birth, LFS 2008")
    return


# In[18]:


#plot_figure4(gbm_ls)


# In[ ]:




