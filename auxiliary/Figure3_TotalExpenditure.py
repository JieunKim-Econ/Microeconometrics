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


####Household Expenditure############
#1. Control variables
##==Characteristics: age of mom and dad, education of mom and dad, mom foreing, mom not married, mom or dad not present
#sum agemom agedad educmom educdad nacmom ecivmom

plot_he=pd.read_stata("data/data_hbs_20110196.dta")


# In[3]:


#Age of mom and dad
#replace agemom=0 if agemom==.
#replace agedad=0 if agedad==.

plot_he['agemom'].fillna(0, inplace=True)
plot_he['agedad'].fillna(0, inplace=True)


# In[4]:


#Mom or dad not present
#drop nomom nodad <-> del df[<var>] or df = df.drop(<var>, axis=1)
#gen nomom=0
#gen nodad=0
#replace nomom=1 if agemom==0
#replace nodad=1 if agedad==0
#sum nomom nodad

del plot_he['nomom']
plot_he['nomom'] = 0
plot_he.loc[plot_he['agemom']==0, 'nomom'] = 1


#plot_he['nomom'].describe()


# In[5]:


#df_he['nodad'] = 0
#df_he.loc[df_he['agedad']==0, 'nodad'] = 1

del plot_he['nodad']
plot_he['nodad'] = 0
plot_he.loc[plot_he['agedad']==0, 'nodad'] = 1

#plot_he['nodad'].describe()


# In[6]:


#/* Education of mom and dad.*/
#gen sec1mom=0
#gen sec1dad=0
#gen sec2mom=0
#gen sec2dad=0
#gen unimom=0
#gen unidad=0

#replace sec1mom=1 if educmom==3
#replace sec1dad=1 if educdad==3

#replace sec2mom=1 if educmom>3 & educmom<7
#replace sec2dad=1 if educdad>3 & educdad<7

#replace unimom=1 if educmom==7 | educmom==8
#replace unidad=1 if educdad==7 | educdad==8

#sum sec1mom sec1dad sec2mom sec2dad unimom unidad

plot_he['sec1mom']=0
plot_he['sec1dad']=0
plot_he['sec2mom']=0
plot_he['sec2dad']=0
plot_he['unimom']=0
plot_he['unidad']=0

plot_he.loc[plot_he['educmom']==3, 'sec1mom'] = 1
plot_he.loc[plot_he['educdad']==3, 'sec1dad'] = 1

plot_he.loc[(plot_he['educmom']>3)&(plot_he['educmom']<7), 'sec2mom'] = 1
plot_he.loc[(plot_he['educdad']>3)&(plot_he['educdad']<7), 'sec2dad'] = 1

plot_he.loc[(plot_he['educmom']==7)|(plot_he['educmom']==8), 'unimom'] = 1
plot_he.loc[(plot_he['educdad']==7)|(plot_he['educdad']==8), 'unidad'] = 1


# In[7]:


#/* Immigrant.*/
#/* (Dummy for mom with foreing nationality.)*/
#gen immig=0
#replace immig=1 if nacmom==2 | nacmom==3

#sum immig

plot_he['immig']=0
plot_he.loc[(plot_he['nacmom']==2) | (plot_he['nacmom']==3), 'immig'] = 1
           
#plot_he['immig'].describe()


# In[8]:


#/* Mom not married.*/
#gen smom=0
#replace smom=1 if ecivmom!=2

plot_he['smom'] = 0
plot_he.loc[plot_he['ecivmom']!=2, 'smom'] = 1

#sum agemom agedad nomom nodad sec1mom sec1dad sec2mom sec2dad unimom unidad immig smom
#plot_he['smom'].describe()


# In[9]:


#/* Siblings.*/
#gen sib=0
#replace sib=1 if nmiem2>1

#gen age2=agemom*agemom
#gen age3=agemom*agemom*agemom

#gen daycare_bin=0
#replace daycare_bin=1 if m_exp12312>0 & m_exp12312!=.

#sum gastmon c_m_exp dur_exp m_exp12312 post month agemom sec1mom sec2mom unimom immig sib if month>-10 & month<9
#centile gastmon c_m_exp dur_exp m_exp12312 post month agemom sec1mom sec2mom unimom immig sib if month>-10 & month<9

plot_he['sib']=0
plot_he.loc[plot_he['nmiem2']>1, 'sib'] = 1

plot_he['age2']=plot_he['agemom']*plot_he['agemom']
plot_he['age3']=plot_he['agemom']*plot_he['agemom']*plot_he['agemom']

plot_he['daycare_bin']=0
plot_he.loc[(plot_he['m_exp12312']>0) &(plot_he['m_exp12312']!=np.nan), 'daycare_bin'] = 1

#df[<condition>][<var>].describe()
#plot_he[(plot_he['month']>-10) & (plot_he['month']<9)]['sib'].describe()


# In[10]:


#'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
gbm_he=plot_he.groupby(['month'], as_index=False).agg({'gastmon':'mean', 'c_m_exp':'mean','m_exp12312':'mean','post':'mean','nomom':'mean','agemom':'mean','sec1mom':'mean',
                                                       'sec2mom':'mean','unimom':'mean','immig':'mean','sib':'mean'})
gbm_he['month2'] = gbm_he['month']*gbm_he['month']
gbm_he['age2']=gbm_he['agemom']*gbm_he['agemom']
#gbm_he


# In[ ]:





# In[11]:


#**************************RRD: Linear Regression (1)******************************************************************************/
#Create interaction dummies
gbm_he['ipost_1']=gbm_he['post']*gbm_he['month']
gbm_he['ipost_2'] =gbm_he['post']*gbm_he['month2']

#xi:reg gastmon post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-30 & month<20, robust
pY_he= ['gastmon','m_exp12312','c_m_exp']
pX_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

plot_he_index = gbm_he[['c_m_exp','m_exp12312','gastmon','post','month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]

result_he=mt.reg(plot_he_index[(gbm_he['month']>-30) & (gbm_he['month']< 20)], 'gastmon', ['month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2'] 
    ,addcons=True)
predict_gast = result_he.yhat.round(0)


# In[12]:


predict_gast
predict_gast_int=predict_gast.astype(int)


# In[13]:


predict_plot_gast = pd.DataFrame(predict_gast_int)
predict_plot_gast.rename(columns={0:'predict_gast'}, inplace=True)
#predict_plot_gast


# In[14]:


gbm_he.loc[gbm_he['month']==-29, 'predict_gast']= predict_plot_gast.iloc[0,0]
gbm_he.loc[gbm_he['month']==-28, 'predict_gast']= predict_plot_gast.iloc[1,0]
gbm_he.loc[gbm_he['month']==-27, 'predict_gast']= predict_plot_gast.iloc[2,0]
gbm_he.loc[gbm_he['month']==-26, 'predict_gast']= predict_plot_gast.iloc[3,0]
gbm_he.loc[gbm_he['month']==-25, 'predict_gast']= predict_plot_gast.iloc[4,0]
gbm_he.loc[gbm_he['month']==-24, 'predict_gast']= predict_plot_gast.iloc[5,0]
gbm_he.loc[gbm_he['month']==-23, 'predict_gast']= predict_plot_gast.iloc[6,0]
gbm_he.loc[gbm_he['month']==-22, 'predict_gast']= predict_plot_gast.iloc[7,0]
gbm_he.loc[gbm_he['month']==-21, 'predict_gast']= predict_plot_gast.iloc[8,0]
gbm_he.loc[gbm_he['month']==-20, 'predict_gast']= predict_plot_gast.iloc[9,0]
gbm_he.loc[gbm_he['month']==-19, 'predict_gast']= predict_plot_gast.iloc[10,0]
gbm_he.loc[gbm_he['month']==-18, 'predict_gast']= predict_plot_gast.iloc[11,0]
gbm_he.loc[gbm_he['month']==-17, 'predict_gast']= predict_plot_gast.iloc[12,0]
gbm_he.loc[gbm_he['month']==-16, 'predict_gast']= predict_plot_gast.iloc[13,0]
gbm_he.loc[gbm_he['month']==-15, 'predict_gast']= predict_plot_gast.iloc[14,0]
gbm_he.loc[gbm_he['month']==-14, 'predict_gast']= predict_plot_gast.iloc[15,0]
gbm_he.loc[gbm_he['month']==-13, 'predict_gast']= predict_plot_gast.iloc[16,0]
gbm_he.loc[gbm_he['month']==-12, 'predict_gast']= predict_plot_gast.iloc[17,0]
gbm_he.loc[gbm_he['month']==-11, 'predict_gast']= predict_plot_gast.iloc[18,0]
gbm_he.loc[gbm_he['month']==-10, 'predict_gast']= predict_plot_gast.iloc[19,0]
gbm_he.loc[gbm_he['month']==-9, 'predict_gast']= predict_plot_gast.iloc[20,0]
gbm_he.loc[gbm_he['month']==-8, 'predict_gast']= predict_plot_gast.iloc[21,0]
gbm_he.loc[gbm_he['month']==-7, 'predict_gast']= predict_plot_gast.iloc[22,0]
gbm_he.loc[gbm_he['month']==-6, 'predict_gast']= predict_plot_gast.iloc[23,0]
gbm_he.loc[gbm_he['month']==-5, 'predict_gast']= predict_plot_gast.iloc[24,0]
gbm_he.loc[gbm_he['month']==-4, 'predict_gast']= predict_plot_gast.iloc[25,0]
gbm_he.loc[gbm_he['month']==-3, 'predict_gast']= predict_plot_gast.iloc[26,0]
gbm_he.loc[gbm_he['month']==-2, 'predict_gast']= predict_plot_gast.iloc[27,0]
gbm_he.loc[gbm_he['month']==-1, 'predict_gast']= predict_plot_gast.iloc[28,0]

gbm_he.loc[gbm_he['month']==0, 'predict_gast']= predict_plot_gast.iloc[29,0]
gbm_he.loc[gbm_he['month']==1, 'predict_gast']= predict_plot_gast.iloc[30,0]
gbm_he.loc[gbm_he['month']==2, 'predict_gast']= predict_plot_gast.iloc[31,0]
gbm_he.loc[gbm_he['month']==3, 'predict_gast']= predict_plot_gast.iloc[32,0]
gbm_he.loc[gbm_he['month']==4, 'predict_gast']= predict_plot_gast.iloc[33,0]
gbm_he.loc[gbm_he['month']==5, 'predict_gast']= predict_plot_gast.iloc[34,0]
gbm_he.loc[gbm_he['month']==6, 'predict_gast']= predict_plot_gast.iloc[35,0]
gbm_he.loc[gbm_he['month']==7, 'predict_gast']= predict_plot_gast.iloc[36,0]
gbm_he.loc[gbm_he['month']==8, 'predict_gast']= predict_plot_gast.iloc[37,0]
gbm_he.loc[gbm_he['month']==9, 'predict_gast']= predict_plot_gast.iloc[38,0]
gbm_he.loc[gbm_he['month']==10, 'predict_gast']= predict_plot_gast.iloc[39,0]
gbm_he.loc[gbm_he['month']==11, 'predict_gast']= predict_plot_gast.iloc[40,0]
gbm_he.loc[gbm_he['month']==12, 'predict_gast']= predict_plot_gast.iloc[41,0]
gbm_he.loc[gbm_he['month']==13, 'predict_gast']= predict_plot_gast.iloc[42,0]
gbm_he.loc[gbm_he['month']==14, 'predict_gast']= predict_plot_gast.iloc[43,0]
gbm_he.loc[gbm_he['month']==15, 'predict_gast']= predict_plot_gast.iloc[44,0]
gbm_he.loc[gbm_he['month']==16, 'predict_gast']= predict_plot_gast.iloc[45,0]
gbm_he.loc[gbm_he['month']==17, 'predict_gast']= predict_plot_gast.iloc[46,0]


# In[15]:


def plot_RRD_curve_tot(data):
    plt.grid(True)
    
    he_p1 = gbm_he[['predict_gast','month']]
    he_plot = he_p1.dropna()
    he_untreat = he_plot[he_plot['month'] < 0]
    he_treat = he_plot[he_plot['month'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),he_untreat['predict_gast'], 3))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),he_treat['predict_gast'], 3))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 60000, 10000)
 
    return


# In[16]:


#plot_RRD_curve_tot(gbm_he)


# In[17]:


def plot_figure3_tot(data):
    tot_plot = gbm_he[['predict_gast','month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 60000, 10000)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0=July 2007)')
    #plt.ylabel('Total expenditure')
    plt.plot(gbm_he.month, gbm_he.gastmon, 'o')
    plt.grid(True)
    plot_RRD_curve_tot(data)

    #plt.title("Figure 3-1 Total expenditure by month of birth")
    return


# In[18]:


#plot_figure3_tot(gbm_he)


# In[19]:


#==================Figure 3-2. Child-related expenditure


# In[20]:


#RRD: Linear Regression (1)
#xi:reg c_m_exp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust

pY_he= ['gastmon','m_exp12312','c_m_exp']
pX_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

plot_he_index = gbm_he[['c_m_exp','m_exp12312','gastmon','post','month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
result_cre=mt.reg(plot_he_index[(gbm_he['month']>-30) & (gbm_he['month']< 20)], 'c_m_exp', ['month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2'] 
    ,addcons=True)
predict_cre = result_cre.yhat.round(0)    


# In[21]:


predict_cre
predict_cre_int=predict_cre.astype(int)


# In[22]:


predict_plot_cre = pd.DataFrame(predict_cre_int)
predict_plot_cre.rename(columns={0:'predict_cre'}, inplace=True)
#predict_plot_cre


# In[23]:


gbm_he.loc[gbm_he['month']==-29, 'predict_cre']= predict_plot_cre.iloc[0,0]
gbm_he.loc[gbm_he['month']==-28, 'predict_cre']= predict_plot_cre.iloc[1,0]
gbm_he.loc[gbm_he['month']==-27, 'predict_cre']= predict_plot_cre.iloc[2,0]
gbm_he.loc[gbm_he['month']==-26, 'predict_cre']= predict_plot_cre.iloc[3,0]
gbm_he.loc[gbm_he['month']==-25, 'predict_cre']= predict_plot_cre.iloc[4,0]
gbm_he.loc[gbm_he['month']==-24, 'predict_cre']= predict_plot_cre.iloc[5,0]
gbm_he.loc[gbm_he['month']==-23, 'predict_cre']= predict_plot_cre.iloc[6,0]
gbm_he.loc[gbm_he['month']==-22, 'predict_cre']= predict_plot_cre.iloc[7,0]
gbm_he.loc[gbm_he['month']==-21, 'predict_cre']= predict_plot_cre.iloc[8,0]
gbm_he.loc[gbm_he['month']==-20, 'predict_cre']= predict_plot_cre.iloc[9,0]
gbm_he.loc[gbm_he['month']==-19, 'predict_cre']= predict_plot_cre.iloc[10,0]
gbm_he.loc[gbm_he['month']==-18, 'predict_cre']= predict_plot_cre.iloc[11,0]
gbm_he.loc[gbm_he['month']==-17, 'predict_cre']= predict_plot_cre.iloc[12,0]
gbm_he.loc[gbm_he['month']==-16, 'predict_cre']= predict_plot_cre.iloc[13,0]
gbm_he.loc[gbm_he['month']==-15, 'predict_cre']= predict_plot_cre.iloc[14,0]
gbm_he.loc[gbm_he['month']==-14, 'predict_cre']= predict_plot_cre.iloc[15,0]
gbm_he.loc[gbm_he['month']==-13, 'predict_cre']= predict_plot_cre.iloc[16,0]
gbm_he.loc[gbm_he['month']==-12, 'predict_cre']= predict_plot_cre.iloc[17,0]
gbm_he.loc[gbm_he['month']==-11, 'predict_cre']= predict_plot_cre.iloc[18,0]
gbm_he.loc[gbm_he['month']==-10, 'predict_cre']= predict_plot_cre.iloc[19,0]
gbm_he.loc[gbm_he['month']==-9, 'predict_cre']= predict_plot_cre.iloc[20,0]
gbm_he.loc[gbm_he['month']==-8, 'predict_cre']= predict_plot_cre.iloc[21,0]
gbm_he.loc[gbm_he['month']==-7, 'predict_cre']= predict_plot_cre.iloc[22,0]
gbm_he.loc[gbm_he['month']==-6, 'predict_cre']= predict_plot_cre.iloc[23,0]
gbm_he.loc[gbm_he['month']==-5, 'predict_cre']= predict_plot_cre.iloc[24,0]
gbm_he.loc[gbm_he['month']==-4, 'predict_cre']= predict_plot_cre.iloc[25,0]
gbm_he.loc[gbm_he['month']==-3, 'predict_cre']= predict_plot_cre.iloc[26,0]
gbm_he.loc[gbm_he['month']==-2, 'predict_cre']= predict_plot_cre.iloc[27,0]
gbm_he.loc[gbm_he['month']==-1, 'predict_cre']= predict_plot_cre.iloc[28,0]

gbm_he.loc[gbm_he['month']==0, 'predict_cre']= predict_plot_cre.iloc[29,0]
gbm_he.loc[gbm_he['month']==1, 'predict_cre']= predict_plot_cre.iloc[30,0]
gbm_he.loc[gbm_he['month']==2, 'predict_cre']= predict_plot_cre.iloc[31,0]
gbm_he.loc[gbm_he['month']==3, 'predict_cre']= predict_plot_cre.iloc[32,0]
gbm_he.loc[gbm_he['month']==4, 'predict_cre']= predict_plot_cre.iloc[33,0]
gbm_he.loc[gbm_he['month']==5, 'predict_cre']= predict_plot_cre.iloc[34,0]
gbm_he.loc[gbm_he['month']==6, 'predict_cre']= predict_plot_cre.iloc[35,0]
gbm_he.loc[gbm_he['month']==7, 'predict_cre']= predict_plot_cre.iloc[36,0]
gbm_he.loc[gbm_he['month']==8, 'predict_cre']= predict_plot_cre.iloc[37,0]
gbm_he.loc[gbm_he['month']==9, 'predict_cre']= predict_plot_cre.iloc[38,0]
gbm_he.loc[gbm_he['month']==10, 'predict_cre']= predict_plot_cre.iloc[39,0]
gbm_he.loc[gbm_he['month']==11, 'predict_cre']= predict_plot_cre.iloc[40,0]
gbm_he.loc[gbm_he['month']==12, 'predict_cre']= predict_plot_cre.iloc[41,0]
gbm_he.loc[gbm_he['month']==13, 'predict_cre']= predict_plot_cre.iloc[42,0]
gbm_he.loc[gbm_he['month']==14, 'predict_cre']= predict_plot_cre.iloc[43,0]
gbm_he.loc[gbm_he['month']==15, 'predict_cre']= predict_plot_cre.iloc[44,0]
gbm_he.loc[gbm_he['month']==16, 'predict_cre']= predict_plot_cre.iloc[45,0]
gbm_he.loc[gbm_he['month']==17, 'predict_cre']= predict_plot_cre.iloc[46,0]


# In[24]:


#gbm_he


# In[25]:


def plot_RRD_curve_cre(data):
    plt.grid(True)
    
    cre_p = gbm_he[['predict_cre','month']]
    cre_plot = cre_p.dropna()
    cre_untreat = cre_plot[cre_plot['month'] < 0]
    cre_treat = cre_plot[cre_plot['month'] >= 0]    
    
    poly_fit = np.poly1d(np.polyfit(np.arange(-29,0,1),cre_untreat['predict_cre'], 2))
    plt.plot(np.arange(-29,0,1), poly_fit(np.arange(-29,0,1)), c='orange',linestyle='-')
    
    poly_fit = np.poly1d(np.polyfit(np.arange(0,18,1),cre_treat['predict_cre'], 2))
    plt.plot(np.arange(0,18,1), poly_fit(np.arange(0,18,1)), c='green',linestyle='-')    
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 10000, 2000)
 
    return


# In[26]:


#plot_RRD_curve_cre(gbm_he)


# In[27]:


def plot_figure3_cre(data):
    cre_plot = gbm_he[['predict_cre','month','month2','nomom','agemom','age2','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 60000, 10000)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0=July 2007)')
    #plt.ylabel('Child-related expenditure')
    plt.plot(gbm_he.month, gbm_he.c_m_exp, 'o')
    plt.grid(True)
    plot_RRD_curve_cre(data)

    #plt.title("Figure 3-2 Child-related expenditure by month of birth")
    return


# In[28]:


#plot_figure3_cre(gbm_he)


# In[29]:


def plot_figure3(data):   
    plt.figure(figsize=(13, 4))
    plt.subplot(1, 2, 1)
    plot_figure3_tot(data)
    plot_RRD_curve_tot(data)
    plt.title("Total expenditure by month of birth",fontsize = 13)

    plt.subplot(1, 2, 2)
    plot_figure3_cre(data)
    plot_RRD_curve_cre(data)
    plt.title("Child-related expenditure by month of birth",fontsize = 13)
    plt.suptitle("Figure 3. Household Expenditure (Annual) by Month of Birth (HBS 2008)", verticalalignment='bottom', fontsize=14)   
    return


# In[30]:


#plot_figure3(gbm_he)


# In[ ]:




