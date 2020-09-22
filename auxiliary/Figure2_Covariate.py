#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import localreg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import econtools
import econtools.metrics as mt


# In[4]:


#*****************Female Labor Supply*********************************
cls=pd.read_stata("data/data_lfs_20110196.dta")


# In[5]:


#1.Control variables
#gen m2 = m*m
cls['m2']=cls['m']*cls['m']

#No father present.*/
#gen nodad=0
#replace nodad=1 if dadid==0
cls['nodad']=0
cls.loc[cls['dadid']==0, 'nodad']=1


# Mother not married 
#gen smom=0
#replace smom=1 if eciv!=2
cls['smom']=0
cls.loc[cls['eciv']!=2, 'smom']=1

#Mother single.*/
#gen single=0
#replace single=1 if eciv==1
cls['single']=0
cls.loc[cls['eciv']==1, 'single']=1

#Mother separated or divorced.*/
#gen sepdiv=0
#replace sepdiv=1 if eciv==4
cls['sepdiv']=0
cls.loc[cls['eciv']==4, 'sepdiv']=1

# No partner in the household.*/
#gen nopart=0
#replace nopart=1 if partner==0
cls['nopart']=0
cls.loc[cls['partner']==0, 'nopart']=1


# In[6]:


#Married mom
cls['married']=0
cls.loc[cls['eciv']==2, 'married']=1


# In[7]:


#Create interaction dummies
cls['ipost_1']=cls['post']*cls['m']


# In[8]:


#'work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'
gbm_cls=cls.groupby(['m'], as_index=False).agg({'age':'mean','immig':'mean','married':'mean','univ':'mean','post':'mean','ipost_1':'mean'})
#gbm_cls
gbm_cls.loc[gbm_cls['immig']==0, 'immig']= np.NaN


# In[9]:


#**************************Regression to check Balance in Covariate******************************************************************************/
##*******************************Figure 2-1: Age of Mother

cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_age=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'age', ['post','m','ipost_1'], addcons=True)
predict_age = result_age.yhat.round(0)


# In[10]:


#predict_age


# In[11]:


predict_plot1 = pd.DataFrame(predict_age)
predict_plot1.rename(columns={0:'predict_age'}, inplace=True)


# In[12]:


gbm_cls.loc[gbm_cls['m']==-29, 'predict_age']= predict_plot1.iloc[0,0]
gbm_cls.loc[gbm_cls['m']==-28, 'predict_age']= predict_plot1.iloc[1,0]
gbm_cls.loc[gbm_cls['m']==-27, 'predict_age']= predict_plot1.iloc[2,0]
gbm_cls.loc[gbm_cls['m']==-26, 'predict_age']= predict_plot1.iloc[3,0]
gbm_cls.loc[gbm_cls['m']==-25, 'predict_age']= predict_plot1.iloc[4,0]
gbm_cls.loc[gbm_cls['m']==-24, 'predict_age']= predict_plot1.iloc[5,0]
gbm_cls.loc[gbm_cls['m']==-23, 'predict_age']= predict_plot1.iloc[6,0]
gbm_cls.loc[gbm_cls['m']==-22, 'predict_age']= predict_plot1.iloc[7,0]
gbm_cls.loc[gbm_cls['m']==-21, 'predict_age']= predict_plot1.iloc[8,0]
gbm_cls.loc[gbm_cls['m']==-20, 'predict_age']= predict_plot1.iloc[9,0]
gbm_cls.loc[gbm_cls['m']==-19, 'predict_age']= predict_plot1.iloc[10,0]
gbm_cls.loc[gbm_cls['m']==-18, 'predict_age']= predict_plot1.iloc[11,0]
gbm_cls.loc[gbm_cls['m']==-17, 'predict_age']= predict_plot1.iloc[12,0]
gbm_cls.loc[gbm_cls['m']==-16, 'predict_age']= predict_plot1.iloc[13,0]
gbm_cls.loc[gbm_cls['m']==-15, 'predict_age']= predict_plot1.iloc[14,0]
gbm_cls.loc[gbm_cls['m']==-14, 'predict_age']= predict_plot1.iloc[15,0]
gbm_cls.loc[gbm_cls['m']==-13, 'predict_age']= predict_plot1.iloc[16,0]
gbm_cls.loc[gbm_cls['m']==-12, 'predict_age']= predict_plot1.iloc[17,0]
gbm_cls.loc[gbm_cls['m']==-11, 'predict_age']= predict_plot1.iloc[18,0]
gbm_cls.loc[gbm_cls['m']==-10, 'predict_age']= predict_plot1.iloc[19,0]
gbm_cls.loc[gbm_cls['m']==-9, 'predict_age']= predict_plot1.iloc[20,0]
gbm_cls.loc[gbm_cls['m']==-8, 'predict_age']= predict_plot1.iloc[21,0]
gbm_cls.loc[gbm_cls['m']==-7, 'predict_age']= predict_plot1.iloc[22,0]
gbm_cls.loc[gbm_cls['m']==-6, 'predict_age']= predict_plot1.iloc[23,0]
gbm_cls.loc[gbm_cls['m']==-5, 'predict_age']= predict_plot1.iloc[24,0]
gbm_cls.loc[gbm_cls['m']==-4, 'predict_age']= predict_plot1.iloc[25,0]
gbm_cls.loc[gbm_cls['m']==-3, 'predict_age']= predict_plot1.iloc[26,0]
gbm_cls.loc[gbm_cls['m']==-2, 'predict_age']= predict_plot1.iloc[27,0]
gbm_cls.loc[gbm_cls['m']==-1, 'predict_age']= predict_plot1.iloc[28,0]

gbm_cls.loc[gbm_cls['m']==0, 'predict_age']= predict_plot1.iloc[29,0]
gbm_cls.loc[gbm_cls['m']==1, 'predict_age']= predict_plot1.iloc[30,0]
gbm_cls.loc[gbm_cls['m']==2, 'predict_age']= predict_plot1.iloc[31,0]
gbm_cls.loc[gbm_cls['m']==3, 'predict_age']= predict_plot1.iloc[32,0]
gbm_cls.loc[gbm_cls['m']==4, 'predict_age']= predict_plot1.iloc[33,0]
gbm_cls.loc[gbm_cls['m']==5, 'predict_age']= predict_plot1.iloc[34,0]
gbm_cls.loc[gbm_cls['m']==6, 'predict_age']= predict_plot1.iloc[35,0]
gbm_cls.loc[gbm_cls['m']==7, 'predict_age']= predict_plot1.iloc[36,0]
gbm_cls.loc[gbm_cls['m']==8, 'predict_age']= predict_plot1.iloc[37,0]
gbm_cls.loc[gbm_cls['m']==9, 'predict_age']= predict_plot1.iloc[38,0]
gbm_cls.loc[gbm_cls['m']==10, 'predict_age']= predict_plot1.iloc[39,0]
gbm_cls.loc[gbm_cls['m']==11, 'predict_age']= predict_plot1.iloc[40,0]
gbm_cls.loc[gbm_cls['m']==12, 'predict_age']= predict_plot1.iloc[41,0]
gbm_cls.loc[gbm_cls['m']==13, 'predict_age']= predict_plot1.iloc[42,0]
gbm_cls.loc[gbm_cls['m']==14, 'predict_age']= predict_plot1.iloc[43,0]
gbm_cls.loc[gbm_cls['m']==15, 'predict_age']= predict_plot1.iloc[44,0]
gbm_cls.loc[gbm_cls['m']==16, 'predict_age']= predict_plot1.iloc[45,0]
gbm_cls.loc[gbm_cls['m']==17, 'predict_age']= predict_plot1.iloc[46,0]


# In[13]:


def plot_cov_age(data):
    plt.grid(True)
    
    cls1 = gbm_cls[['predict_age','m']]
    age_plot = cls1.dropna()
    age_untreat = age_plot[age_plot['m'] < 0]
    age_treat = age_plot[age_plot['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),age_untreat['predict_age'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),age_treat['predict_age'], 1))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 10, 10)
    plt.ylim(30, 36, 2)
    
    return


# In[14]:


#plot_cov_age(gbm_cls)


# In[15]:


def plot_figure1_age(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 10, 10)
    plt.ylim(30, 36, 2)
    plt.axvline(x=0, color='r')
    #plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Average age of mother')
    plt.plot(gbm_cls.m, gbm_cls.age, 'o')
    plt.grid(True)
    plot_cov_age(data)

    #plt.title("Figure 2-1. Average age of the mother by month of birth")
    return


# In[16]:


#plot_figure1_age(gbm_cls)


# In[ ]:





# In[17]:


##*******************************Figure 2-2. Fraction foreign mothers by month of birth
cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_imm=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'immig', ['post','m'], addcons=True)
predict_immig = result_imm.yhat.round(4)


# In[18]:


predict_plot2 = pd.DataFrame(predict_immig)
predict_plot2.rename(columns={0:'predict_immig'}, inplace=True)


# In[19]:


gbm_cls.loc[gbm_cls['m']==-29, 'predict_immig']= predict_plot2.iloc[0,0]
gbm_cls.loc[gbm_cls['m']==-28, 'predict_immig']= predict_plot2.iloc[1,0]
gbm_cls.loc[gbm_cls['m']==-27, 'predict_immig']= predict_plot2.iloc[2,0]
gbm_cls.loc[gbm_cls['m']==-26, 'predict_immig']= predict_plot2.iloc[3,0]
gbm_cls.loc[gbm_cls['m']==-25, 'predict_immig']= predict_plot2.iloc[4,0]
gbm_cls.loc[gbm_cls['m']==-24, 'predict_immig']= predict_plot2.iloc[5,0]
gbm_cls.loc[gbm_cls['m']==-23, 'predict_immig']= predict_plot2.iloc[6,0]
gbm_cls.loc[gbm_cls['m']==-22, 'predict_immig']= predict_plot2.iloc[7,0]
gbm_cls.loc[gbm_cls['m']==-21, 'predict_immig']= predict_plot2.iloc[8,0]
gbm_cls.loc[gbm_cls['m']==-20, 'predict_immig']= predict_plot2.iloc[9,0]
gbm_cls.loc[gbm_cls['m']==-19, 'predict_immig']= predict_plot2.iloc[10,0]
gbm_cls.loc[gbm_cls['m']==-18, 'predict_immig']= predict_plot2.iloc[11,0]
gbm_cls.loc[gbm_cls['m']==-17, 'predict_immig']= predict_plot2.iloc[12,0]
gbm_cls.loc[gbm_cls['m']==-16, 'predict_immig']= predict_plot2.iloc[13,0]
gbm_cls.loc[gbm_cls['m']==-15, 'predict_immig']= predict_plot2.iloc[14,0]
gbm_cls.loc[gbm_cls['m']==-14, 'predict_immig']= predict_plot2.iloc[15,0]
gbm_cls.loc[gbm_cls['m']==-13, 'predict_immig']= predict_plot2.iloc[16,0]
gbm_cls.loc[gbm_cls['m']==-12, 'predict_immig']= predict_plot2.iloc[17,0]
gbm_cls.loc[gbm_cls['m']==-11, 'predict_immig']= predict_plot2.iloc[18,0]
gbm_cls.loc[gbm_cls['m']==-10, 'predict_immig']= predict_plot2.iloc[19,0]
gbm_cls.loc[gbm_cls['m']==-9, 'predict_immig']= predict_plot2.iloc[20,0]
gbm_cls.loc[gbm_cls['m']==-8, 'predict_immig']= predict_plot2.iloc[21,0]
gbm_cls.loc[gbm_cls['m']==-7, 'predict_immig']= predict_plot2.iloc[22,0]
gbm_cls.loc[gbm_cls['m']==-6, 'predict_immig']= predict_plot2.iloc[23,0]
gbm_cls.loc[gbm_cls['m']==-5, 'predict_immig']= predict_plot2.iloc[24,0]
gbm_cls.loc[gbm_cls['m']==-4, 'predict_immig']= predict_plot2.iloc[25,0]
gbm_cls.loc[gbm_cls['m']==-3, 'predict_immig']= predict_plot2.iloc[26,0]
gbm_cls.loc[gbm_cls['m']==-2, 'predict_immig']= predict_plot2.iloc[27,0]
gbm_cls.loc[gbm_cls['m']==-1, 'predict_immig']= predict_plot2.iloc[28,0]

gbm_cls.loc[gbm_cls['m']==0, 'predict_immig']= predict_plot2.iloc[29,0]
gbm_cls.loc[gbm_cls['m']==1, 'predict_immig']= predict_plot2.iloc[30,0]
gbm_cls.loc[gbm_cls['m']==2, 'predict_immig']= predict_plot2.iloc[31,0]
gbm_cls.loc[gbm_cls['m']==3, 'predict_immig']= predict_plot2.iloc[32,0]
gbm_cls.loc[gbm_cls['m']==4, 'predict_immig']= predict_plot2.iloc[33,0]
gbm_cls.loc[gbm_cls['m']==5, 'predict_immig']= predict_plot2.iloc[34,0]
gbm_cls.loc[gbm_cls['m']==6, 'predict_immig']= predict_plot2.iloc[35,0]
gbm_cls.loc[gbm_cls['m']==7, 'predict_immig']= predict_plot2.iloc[36,0]
gbm_cls.loc[gbm_cls['m']==8, 'predict_immig']= predict_plot2.iloc[37,0]
gbm_cls.loc[gbm_cls['m']==9, 'predict_immig']= predict_plot2.iloc[38,0]
gbm_cls.loc[gbm_cls['m']==10, 'predict_immig']= predict_plot2.iloc[39,0]
gbm_cls.loc[gbm_cls['m']==11, 'predict_immig']= predict_plot2.iloc[40,0]
gbm_cls.loc[gbm_cls['m']==12, 'predict_immig']= predict_plot2.iloc[41,0]
gbm_cls.loc[gbm_cls['m']==13, 'predict_immig']= predict_plot2.iloc[42,0]
gbm_cls.loc[gbm_cls['m']==14, 'predict_immig']= predict_plot2.iloc[43,0]
gbm_cls.loc[gbm_cls['m']==15, 'predict_immig']= predict_plot2.iloc[44,0]
gbm_cls.loc[gbm_cls['m']==16, 'predict_immig']= predict_plot2.iloc[45,0]
#gbm_cls.loc[gbm_cls['m']==17, 'predict_immig']= predict_plot2.iloc[46,0]


# In[20]:


def plot_cov_immig(data):
    plt.grid(True)
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 0.4, 0.1)
    
    cls_imm = gbm_cls[['predict_immig','m']]
    imm_plot = cls_imm.dropna()
    imm_untreat = imm_plot[imm_plot['m'] < 0]
    imm_treat = imm_plot[imm_plot['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),imm_untreat['predict_immig'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,17,1),imm_treat['predict_immig'], 1))
    plt.plot(np.arange(0,17,1), poly_fit(np.arange(0,17,1)), c='green',linestyle='-')    
    
    
    return


# In[21]:


#plot_cov_immig(gbm_cls)


# In[22]:


def plot_figure2_immig(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 20,10)
    plt.ylim(0, 0.4, 0.1)
    
    plt.axvline(x=0, color='r')
    #plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Fraction Foreign Mothers')
    plt.plot(gbm_cls.m, gbm_cls.immig, 'o')
    plt.grid(True)
    plot_cov_immig(gbm_cls)

    #plt.title("Figure 2-2. Fraction foreign mothers by month of birth")
    return


# In[23]:


#plot_figure2_immig(gbm_cls)


# In[24]:


##*******************************Figure 2-3. Fraction married mothers by month of birth
cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_mar=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'married', ['post','m'], addcons=True)
predict_married = result_mar.yhat.round(4)


# In[25]:


predict_plot3 = pd.DataFrame(predict_married)
predict_plot3.rename(columns={0:'predict_married'}, inplace=True)


# In[26]:


gbm_cls.loc[gbm_cls['m']==-29, 'predict_married']= predict_plot3.iloc[0,0]
gbm_cls.loc[gbm_cls['m']==-28, 'predict_married']= predict_plot3.iloc[1,0]
gbm_cls.loc[gbm_cls['m']==-27, 'predict_married']= predict_plot3.iloc[2,0]
gbm_cls.loc[gbm_cls['m']==-26, 'predict_married']= predict_plot3.iloc[3,0]
gbm_cls.loc[gbm_cls['m']==-25, 'predict_married']= predict_plot3.iloc[4,0]
gbm_cls.loc[gbm_cls['m']==-24, 'predict_married']= predict_plot3.iloc[5,0]
gbm_cls.loc[gbm_cls['m']==-23, 'predict_married']= predict_plot3.iloc[6,0]
gbm_cls.loc[gbm_cls['m']==-22, 'predict_married']= predict_plot3.iloc[7,0]
gbm_cls.loc[gbm_cls['m']==-21, 'predict_married']= predict_plot3.iloc[8,0]
gbm_cls.loc[gbm_cls['m']==-20, 'predict_married']= predict_plot3.iloc[9,0]
gbm_cls.loc[gbm_cls['m']==-19, 'predict_married']= predict_plot3.iloc[10,0]
gbm_cls.loc[gbm_cls['m']==-18, 'predict_married']= predict_plot3.iloc[11,0]
gbm_cls.loc[gbm_cls['m']==-17, 'predict_married']= predict_plot3.iloc[12,0]
gbm_cls.loc[gbm_cls['m']==-16, 'predict_married']= predict_plot3.iloc[13,0]
gbm_cls.loc[gbm_cls['m']==-15, 'predict_married']= predict_plot3.iloc[14,0]
gbm_cls.loc[gbm_cls['m']==-14, 'predict_married']= predict_plot3.iloc[15,0]
gbm_cls.loc[gbm_cls['m']==-13, 'predict_married']= predict_plot3.iloc[16,0]
gbm_cls.loc[gbm_cls['m']==-12, 'predict_married']= predict_plot3.iloc[17,0]
gbm_cls.loc[gbm_cls['m']==-11, 'predict_married']= predict_plot3.iloc[18,0]
gbm_cls.loc[gbm_cls['m']==-10, 'predict_married']= predict_plot3.iloc[19,0]
gbm_cls.loc[gbm_cls['m']==-9, 'predict_married']= predict_plot3.iloc[20,0]
gbm_cls.loc[gbm_cls['m']==-8, 'predict_married']= predict_plot3.iloc[21,0]
gbm_cls.loc[gbm_cls['m']==-7, 'predict_married']= predict_plot3.iloc[22,0]
gbm_cls.loc[gbm_cls['m']==-6, 'predict_married']= predict_plot3.iloc[23,0]
gbm_cls.loc[gbm_cls['m']==-5, 'predict_married']= predict_plot3.iloc[24,0]
gbm_cls.loc[gbm_cls['m']==-4, 'predict_married']= predict_plot3.iloc[25,0]
gbm_cls.loc[gbm_cls['m']==-3, 'predict_married']= predict_plot3.iloc[26,0]
gbm_cls.loc[gbm_cls['m']==-2, 'predict_married']= predict_plot3.iloc[27,0]
gbm_cls.loc[gbm_cls['m']==-1, 'predict_married']= predict_plot3.iloc[28,0]

gbm_cls.loc[gbm_cls['m']==0, 'predict_married']= predict_plot3.iloc[29,0]
gbm_cls.loc[gbm_cls['m']==1, 'predict_married']= predict_plot3.iloc[30,0]
gbm_cls.loc[gbm_cls['m']==2, 'predict_married']= predict_plot3.iloc[31,0]
gbm_cls.loc[gbm_cls['m']==3, 'predict_married']= predict_plot3.iloc[32,0]
gbm_cls.loc[gbm_cls['m']==4, 'predict_married']= predict_plot3.iloc[33,0]
gbm_cls.loc[gbm_cls['m']==5, 'predict_married']= predict_plot3.iloc[34,0]
gbm_cls.loc[gbm_cls['m']==6, 'predict_married']= predict_plot3.iloc[35,0]
gbm_cls.loc[gbm_cls['m']==7, 'predict_married']= predict_plot3.iloc[36,0]
gbm_cls.loc[gbm_cls['m']==8, 'predict_married']= predict_plot3.iloc[37,0]
gbm_cls.loc[gbm_cls['m']==9, 'predict_married']= predict_plot3.iloc[38,0]
gbm_cls.loc[gbm_cls['m']==10, 'predict_married']= predict_plot3.iloc[39,0]
gbm_cls.loc[gbm_cls['m']==11, 'predict_married']= predict_plot3.iloc[40,0]
gbm_cls.loc[gbm_cls['m']==12, 'predict_married']= predict_plot3.iloc[41,0]
gbm_cls.loc[gbm_cls['m']==13, 'predict_married']= predict_plot3.iloc[42,0]
gbm_cls.loc[gbm_cls['m']==14, 'predict_married']= predict_plot3.iloc[43,0]
gbm_cls.loc[gbm_cls['m']==15, 'predict_married']= predict_plot3.iloc[44,0]
gbm_cls.loc[gbm_cls['m']==16, 'predict_married']= predict_plot3.iloc[45,0]
gbm_cls.loc[gbm_cls['m']==17, 'predict_married']= predict_plot3.iloc[46,0]


# In[27]:


def plot_cov_married(data):
    plt.grid(True)
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1, 0.2)
    
    cls_mar = gbm_cls[['predict_married','m']]
    ls_plot3 = cls_mar.dropna()
    mar_untreat = ls_plot3[ls_plot3['m'] < 0]
    mar_treat = ls_plot3[ls_plot3['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),mar_untreat['predict_married'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),mar_treat['predict_married'], 1))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
      
    return


# In[28]:


#plot_cov_married(gbm_cls)


# In[29]:


def plot_figure3_married(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1, 0.2)
     
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Fraction married mothers')
    plt.plot(gbm_cls.m, gbm_cls.married, 'o')
    plt.grid(True)
    plot_cov_married(data)

    #plt.title("Figure 2-3. Fraction married mothers by month of birth")
    return


# In[30]:


#plot_figure3_married(gbm_cls)


# In[ ]:





# In[31]:


##*******************************Figure 2-4: Fraction mothers with university degree by month of birth
cov_X = ['age','immig','married','univ','post','m','ipost_1']
cov_index = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]

result_univ=mt.reg(cov_index[(gbm_cls['m']> -30) & (gbm_cls['m']< 20)], 'univ', ['post','m','ipost_1'], addcons=True)
predict_univ = result_univ.yhat.round(2)


# In[32]:


predict_plot4 = pd.DataFrame(predict_univ)
predict_plot4.rename(columns={0:'predict_univ'}, inplace=True)


# In[33]:


gbm_cls.loc[gbm_cls['m']==-29, 'predict_univ']= predict_plot4.iloc[0,0]
gbm_cls.loc[gbm_cls['m']==-28, 'predict_univ']= predict_plot4.iloc[1,0]
gbm_cls.loc[gbm_cls['m']==-27, 'predict_univ']= predict_plot4.iloc[2,0]
gbm_cls.loc[gbm_cls['m']==-26, 'predict_univ']= predict_plot4.iloc[3,0]
gbm_cls.loc[gbm_cls['m']==-25, 'predict_univ']= predict_plot4.iloc[4,0]
gbm_cls.loc[gbm_cls['m']==-24, 'predict_univ']= predict_plot4.iloc[5,0]
gbm_cls.loc[gbm_cls['m']==-23, 'predict_univ']= predict_plot4.iloc[6,0]
gbm_cls.loc[gbm_cls['m']==-22, 'predict_univ']= predict_plot4.iloc[7,0]
gbm_cls.loc[gbm_cls['m']==-21, 'predict_univ']= predict_plot4.iloc[8,0]
gbm_cls.loc[gbm_cls['m']==-20, 'predict_univ']= predict_plot4.iloc[9,0]
gbm_cls.loc[gbm_cls['m']==-19, 'predict_univ']= predict_plot4.iloc[10,0]
gbm_cls.loc[gbm_cls['m']==-18, 'predict_univ']= predict_plot4.iloc[11,0]
gbm_cls.loc[gbm_cls['m']==-17, 'predict_univ']= predict_plot4.iloc[12,0]
gbm_cls.loc[gbm_cls['m']==-16, 'predict_univ']= predict_plot4.iloc[13,0]
gbm_cls.loc[gbm_cls['m']==-15, 'predict_univ']= predict_plot4.iloc[14,0]
gbm_cls.loc[gbm_cls['m']==-14, 'predict_univ']= predict_plot4.iloc[15,0]
gbm_cls.loc[gbm_cls['m']==-13, 'predict_univ']= predict_plot4.iloc[16,0]
gbm_cls.loc[gbm_cls['m']==-12, 'predict_univ']= predict_plot4.iloc[17,0]
gbm_cls.loc[gbm_cls['m']==-11, 'predict_univ']= predict_plot4.iloc[18,0]
gbm_cls.loc[gbm_cls['m']==-10, 'predict_univ']= predict_plot4.iloc[19,0]
gbm_cls.loc[gbm_cls['m']==-9, 'predict_univ']= predict_plot4.iloc[20,0]
gbm_cls.loc[gbm_cls['m']==-8, 'predict_univ']= predict_plot4.iloc[21,0]
gbm_cls.loc[gbm_cls['m']==-7, 'predict_univ']= predict_plot4.iloc[22,0]
gbm_cls.loc[gbm_cls['m']==-6, 'predict_univ']= predict_plot4.iloc[23,0]
gbm_cls.loc[gbm_cls['m']==-5, 'predict_univ']= predict_plot4.iloc[24,0]
gbm_cls.loc[gbm_cls['m']==-4, 'predict_univ']= predict_plot4.iloc[25,0]
gbm_cls.loc[gbm_cls['m']==-3, 'predict_univ']= predict_plot4.iloc[26,0]
gbm_cls.loc[gbm_cls['m']==-2, 'predict_univ']= predict_plot4.iloc[27,0]
gbm_cls.loc[gbm_cls['m']==-1, 'predict_univ']= predict_plot4.iloc[28,0]

gbm_cls.loc[gbm_cls['m']==0, 'predict_univ']= predict_plot4.iloc[29,0]
gbm_cls.loc[gbm_cls['m']==1, 'predict_univ']= predict_plot4.iloc[30,0]
gbm_cls.loc[gbm_cls['m']==2, 'predict_univ']= predict_plot4.iloc[31,0]
gbm_cls.loc[gbm_cls['m']==3, 'predict_univ']= predict_plot4.iloc[32,0]
gbm_cls.loc[gbm_cls['m']==4, 'predict_univ']= predict_plot4.iloc[33,0]
gbm_cls.loc[gbm_cls['m']==5, 'predict_univ']= predict_plot4.iloc[34,0]
gbm_cls.loc[gbm_cls['m']==6, 'predict_univ']= predict_plot4.iloc[35,0]
gbm_cls.loc[gbm_cls['m']==7, 'predict_univ']= predict_plot4.iloc[36,0]
gbm_cls.loc[gbm_cls['m']==8, 'predict_univ']= predict_plot4.iloc[37,0]
gbm_cls.loc[gbm_cls['m']==9, 'predict_univ']= predict_plot4.iloc[38,0]
gbm_cls.loc[gbm_cls['m']==10, 'predict_univ']= predict_plot4.iloc[39,0]
gbm_cls.loc[gbm_cls['m']==11, 'predict_univ']= predict_plot4.iloc[40,0]
gbm_cls.loc[gbm_cls['m']==12, 'predict_univ']= predict_plot4.iloc[41,0]
gbm_cls.loc[gbm_cls['m']==13, 'predict_univ']= predict_plot4.iloc[42,0]
gbm_cls.loc[gbm_cls['m']==14, 'predict_univ']= predict_plot4.iloc[43,0]
gbm_cls.loc[gbm_cls['m']==15, 'predict_univ']= predict_plot4.iloc[44,0]
gbm_cls.loc[gbm_cls['m']==16, 'predict_univ']= predict_plot4.iloc[45,0]
gbm_cls.loc[gbm_cls['m']==17, 'predict_univ']= predict_plot4.iloc[46,0]


# In[34]:


def plot_cov_univ(data):
    plt.grid(True)
    plt.xlim(-30, 20, 10)
    plt.ylim(0.1, 0.5, 0.1)
    
    cls_uni = gbm_cls[['predict_univ','m']]
    ls_plot4 = cls_uni.dropna()
    uni_untreat = ls_plot4[ls_plot4['m'] < 0]
    uni_treat = ls_plot4[ls_plot4['m'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),uni_untreat['predict_univ'], 1))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),uni_treat['predict_univ'], 1))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
      
    return


# In[35]:


#plot_cov_univ(gbm_cls)


# In[36]:


def plot_figure4_uni(data):
    cov_plot = gbm_cls[['age','immig','married','univ','post','m','ipost_1']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0.1, 0.5, 0.1)
     
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Fraction mothers with university degree')
    plt.plot(gbm_cls.m, gbm_cls.univ.round(3), 'o')
    plt.grid(True)
    plot_cov_univ(data)

    #plt.title("Figure 2-4. Fraction mothers with university degree by month of birth")
    return


# In[37]:


#plot_figure4_uni(gbm_cls)


# In[80]:


def plot_figure2(data): 
    plt.figure(figsize=(10,8), dpi= 80)
    plt.subplot(2,2,1)
    plot_figure1_age(data)
    plot_cov_age(data)
    plt.title("Average age of the mother by month of birth",fontsize = 10)

    
    plt.subplot(2,2,2)
    plot_figure2_immig(data)
    plot_cov_immig(data)
    plt.title("Fraction foreign mothers by month of birth",fontsize = 10)
    
    
    plt.subplot(2,2,3)
    plot_figure3_married(data)
    plot_cov_married(data)
    plt.title("Fraction married mothers by month of birth",fontsize = 10)

    
    plt.subplot(2,2,4)
    plot_figure4_uni(data)
    plot_cov_univ(data)
    plt.title("Fraction mothers with university degree by month of birth",fontsize = 10)
    
    plt.suptitle('Figure 2. Balance in Covariates', verticalalignment='bottom', fontsize=14)    
    
    return


# In[81]:


#plot_figure2(gbm_cls)


# In[ ]:




