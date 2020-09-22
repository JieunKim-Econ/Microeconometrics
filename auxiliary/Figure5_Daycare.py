#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
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

dc=pd.read_stata("data/data_hbs_20110196.dta")


# In[3]:


#Age of mom and dad
#replace agemom=0 if agemom==.
#replace agedad=0 if agedad==.

dc['agemom'].fillna(0, inplace=True)
dc['agedad'].fillna(0, inplace=True)


# In[4]:


#Mom or dad not present
#drop nomom nodad <-> del df[<var>] or df = df.drop(<var>, axis=1)
#gen nomom=0
#gen nodad=0
#replace nomom=1 if agemom==0
#replace nodad=1 if agedad==0
#sum nomom nodad

del dc['nomom']
dc['nomom'] = 0
dc.loc[dc['agemom']==0, 'nomom'] = 1


#plot_he['nomom'].describe()


# In[5]:


#df_he['nodad'] = 0
#df_he.loc[df_he['agedad']==0, 'nodad'] = 1

del dc['nodad']
dc['nodad'] = 0
dc.loc[dc['agedad']==0, 'nodad'] = 1

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

dc['sec1mom']=0
dc['sec1dad']=0
dc['sec2mom']=0
dc['sec2dad']=0
dc['unimom']=0
dc['unidad']=0

dc.loc[dc['educmom']==3, 'sec1mom'] = 1
dc.loc[dc['educdad']==3, 'sec1dad'] = 1

dc.loc[(dc['educmom']>3)&(dc['educmom']<7), 'sec2mom'] = 1
dc.loc[(dc['educdad']>3)&(dc['educdad']<7), 'sec2dad'] = 1

dc.loc[(dc['educmom']==7)|(dc['educmom']==8), 'unimom'] = 1
dc.loc[(dc['educdad']==7)|(dc['educdad']==8), 'unidad'] = 1


# In[7]:


#/* Immigrant.*/
#/* (Dummy for mom with foreing nationality.)*/
#gen immig=0
#replace immig=1 if nacmom==2 | nacmom==3

#sum immig

dc['immig']=0
dc.loc[(dc['nacmom']==2) | (dc['nacmom']==3), 'immig'] = 1
           
#plot_he['immig'].describe()


# In[8]:


#/* Mom not married.*/
#gen smom=0
#replace smom=1 if ecivmom!=2

dc['smom'] = 0
dc.loc[dc['ecivmom']!=2, 'smom'] = 1

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

dc['sib']=0
dc.loc[dc['nmiem2']>1, 'sib'] = 1

dc['age2']=dc['agemom']*dc['agemom']
dc['age3']=dc['agemom']*dc['agemom']*dc['agemom']

dc['daycare_bin']=0
dc.loc[(dc['m_exp12312']>0) &(dc['m_exp12312']!=np.nan), 'daycare_bin'] = 1


#df[<condition>][<var>].describe()
#plot_he[(plot_he['month']>-10) & (plot_he['month']<9)]['sib'].describe()


# In[10]:


#'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
bim_dc=dc.groupby(['month'], as_index=False).agg({'gastmon':'mean', 'c_m_exp':'mean','m_exp12312':'mean','daycare_bin':'mean','post':'mean','nomom':'mean','agemom':'mean','sec1mom':'mean',
                                                       'sec2mom':'mean','unimom':'mean','immig':'mean','sib':'mean'})
bim_dc['month2'] = bim_dc['month']*bim_dc['month']
bim_dc['age2']=bim_dc['agemom']*bim_dc['agemom']
bim_dc['age3']=bim_dc['agemom']*bim_dc['agemom']*bim_dc['agemom']

#bim_dc


# In[11]:


#Create Bimonthly daycare expenditure

#df['bim_dce']=np.nan
bim_dc.loc[(bim_dc['month'] ==-29), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-29,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-30,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-27), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-27,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-28,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-25), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-25,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-26,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-23), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-23,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-24,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-21), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-21,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-22,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-19), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-19,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-20,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-17), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-17,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-18,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-15), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-15,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-16,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-13), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-13,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-14,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-11), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-11,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-12,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-9), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-9,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-10,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-7), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-7,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-8,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-5), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-5,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-6,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-3), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-3,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-4,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-1), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==-1,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==-2,'m_exp12312'].sum())/2

bim_dc.loc[(bim_dc['month'] ==1), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==1,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==0,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==3), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==3,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==2,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==5), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==5,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==4,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==7), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==7,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==6,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==9), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==9,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==8,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==11), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==11,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==10,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==13), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==13,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==12,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==15), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==15,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==14,'m_exp12312'].sum())/2
bim_dc.loc[(bim_dc['month'] ==17), 'bim_dce'] = (bim_dc.loc[bim_dc['month']==17,'m_exp12312'].sum() + bim_dc.loc[bim_dc['month']==16,'m_exp12312'].sum())/2


# In[12]:


#Create Bimonthly "Binary" daycare expenditure

#df['bim_bindce']=np.nan
bim_dc.loc[(bim_dc['month'] ==-29), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-29,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-30,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-27), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-27,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-28,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-25), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-25,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-26,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-23), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-23,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-24,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-21), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-21,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-22,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-19), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-19,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-20,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-17), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-17,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-18,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-15), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-15,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-16,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-13), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-13,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-14,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-11), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-11,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-12,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-9), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-9,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-10,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-7), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-7,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-8,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-5), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-5,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-6,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-3), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-3,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-4,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==-1), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==-1,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==-2,'daycare_bin'].sum())/2

bim_dc.loc[(bim_dc['month'] ==1), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==1,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==0,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==3), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==3,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==2,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==5), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==5,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==4,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==7), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==7,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==6,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==9), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==9,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==8,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==11), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==11,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==10,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==13), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==13,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==12,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==15), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==15,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==14,'daycare_bin'].sum())/2
bim_dc.loc[(bim_dc['month'] ==17), 'bim_bindce'] = (bim_dc.loc[bim_dc['month']==17,'daycare_bin'].sum() + bim_dc.loc[bim_dc['month']==16,'daycare_bin'].sum())/2


# In[13]:


bim_dc


# In[14]:


#**************************RRD: Linear Regression *****************************************************************************/
##========== Private childcare 
#Create interaction dummies
bim_dc['ipost_1']=bim_dc['post']*bim_dc['month']
bim_dc['ipost_2'] =bim_dc['post']*bim_dc['month2']

#(1)xi:reg m_exp12312 post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust

Y_dce= ['gastmon','m_exp12312','c_m_exp','bim_dce']
X_dce = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']
plot_dce_index = bim_dc[['bim_dce','c_m_exp','m_exp12312','gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
dce_result=mt.reg(plot_dce_index[(bim_dc['month']> -30) & (bim_dc['month']< 20)], 'bim_dce', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2'], 
      addcons=True)
predict_dce = dce_result.yhat.round(0)    


# In[15]:


predict_intg=predict_dce.astype(int)


# In[16]:


predict_dce = pd.DataFrame(predict_intg)
predict_dce.rename(columns={0:'predict_bim_dce'}, inplace=True)


# In[17]:


#df['predict_bim_n'] = np.nan
bim_dc.loc[bim_dc['month']==-29, 'predict_bim_dce']= predict_dce.iloc[0,0]
bim_dc.loc[bim_dc['month']==-27, 'predict_bim_dce']= predict_dce.iloc[1,0]
bim_dc.loc[bim_dc['month']==-25, 'predict_bim_dce']= predict_dce.iloc[2,0]
bim_dc.loc[bim_dc['month']==-23, 'predict_bim_dce']= predict_dce.iloc[3,0]
bim_dc.loc[bim_dc['month']==-21, 'predict_bim_dce']= predict_dce.iloc[4,0]
bim_dc.loc[bim_dc['month']==-19, 'predict_bim_dce']= predict_dce.iloc[5,0]
bim_dc.loc[bim_dc['month']==-17, 'predict_bim_dce']= predict_dce.iloc[6,0]
bim_dc.loc[bim_dc['month']==-15, 'predict_bim_dce']= predict_dce.iloc[7,0]
bim_dc.loc[bim_dc['month']==-13, 'predict_bim_dce']= predict_dce.iloc[8,0]
bim_dc.loc[bim_dc['month']==-11, 'predict_bim_dce']= predict_dce.iloc[9,0]
bim_dc.loc[bim_dc['month']==-9, 'predict_bim_dce']= predict_dce.iloc[10,0]
bim_dc.loc[bim_dc['month']==-7, 'predict_bim_dce']= predict_dce.iloc[11,0]
bim_dc.loc[bim_dc['month']==-5, 'predict_bim_dce']= predict_dce.iloc[12,0]
bim_dc.loc[bim_dc['month']==-3, 'predict_bim_dce']= predict_dce.iloc[13,0]
bim_dc.loc[bim_dc['month']==-1, 'predict_bim_dce']= predict_dce.iloc[14,0]
bim_dc.loc[bim_dc['month']==1, 'predict_bim_dce']= predict_dce.iloc[15,0]
bim_dc.loc[bim_dc['month']==3, 'predict_bim_dce']= predict_dce.iloc[16,0]
bim_dc.loc[bim_dc['month']==5, 'predict_bim_dce']= predict_dce.iloc[17,0]
bim_dc.loc[bim_dc['month']==7, 'predict_bim_dce']= predict_dce.iloc[18,0]
bim_dc.loc[bim_dc['month']==9, 'predict_bim_dce']= predict_dce.iloc[19,0]
bim_dc.loc[bim_dc['month']==11, 'predict_bim_dce']= predict_dce.iloc[20,0]
bim_dc.loc[bim_dc['month']==13, 'predict_bim_dce']= predict_dce.iloc[21,0]
bim_dc.loc[bim_dc['month']==15, 'predict_bim_dce']= predict_dce.iloc[22,0]
bim_dc.loc[bim_dc['month']==17, 'predict_bim_dce']= predict_dce.iloc[23,0]


# In[18]:


def plot_RRD_curve_dce(data):
    
    plt.grid(True)
    p_dce = bim_dc[['predict_bim_dce','month']]
    plot_dce = p_dce.dropna()
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1000, 200)
       
    untreat_dce = plot_dce[plot_dce['month'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),untreat_dce['predict_bim_dce'], 1)
    plt.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')
    
    treat_dce = plot_dce[plot_dce['month'] >= 0]
    m, b = np.polyfit(np.arange(0,18,2),treat_dce['predict_bim_dce'], 1)
    plt.plot(np.arange(0,18,2), m*np.arange(0,18,2) + b, color='green')   
 
    return


# In[19]:


#plot_RRD_curve_dce(bim_dc)


# In[20]:


def plot_figure5_dce(data):
    dce_plot = bim_dc[['predict_bim_dce','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 1000, 200)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    #plt.ylabel('Day care expenditure')
    plt.plot(bim_dc.month, bim_dc.bim_dce, 'o')
    plt.grid(True)
    plot_RRD_curve_dce(data)

    plt.title("Figure 5-1 Day care expenditure by month of birth")
    return


# In[21]:


#plot_figure5_dce(bim_dc)


# In[22]:


#**************************Figure 5-2 Fraction with positive day care expenditure by month of birth


# In[23]:


##[BINARY] Private childcare 
#(1)xi:reg bim_bindce post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 if month>-30 & month<20, robust

Y_bdce = 'bim_bindce'
X_bdce = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']
bdce_index = bim_dc[['bim_bindce','c_m_exp','m_exp12312','gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
bdce_result=mt.reg(bdce_index[(bim_dc['month']> -30) & (bim_dc['month']< 20)], 'bim_bindce', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2'], 
      addcons=True)
predict_bin_dce = bdce_result.yhat.round(2)    


# In[24]:


#predict_bin_dce=predict_bin_dce.astype(int)
predict_bin_dce


# In[25]:


predict_bindce = pd.DataFrame(predict_bin_dce)
predict_bindce.rename(columns={0:'predict_bim_bindce'}, inplace=True)


# In[29]:


#df['predict_bim_bindce'] = np.nan
bim_dc.loc[bim_dc['month']==-29, 'predict_bim_bindce']= predict_bindce.iloc[0,0]
bim_dc.loc[bim_dc['month']==-27, 'predict_bim_bindce']= predict_bindce.iloc[1,0]
bim_dc.loc[bim_dc['month']==-25, 'predict_bim_bindce']= predict_bindce.iloc[2,0]
bim_dc.loc[bim_dc['month']==-23, 'predict_bim_bindce']= predict_bindce.iloc[3,0]
bim_dc.loc[bim_dc['month']==-21, 'predict_bim_bindce']= predict_bindce.iloc[4,0]
bim_dc.loc[bim_dc['month']==-19, 'predict_bim_bindce']= predict_bindce.iloc[5,0]
bim_dc.loc[bim_dc['month']==-17, 'predict_bim_bindce']= predict_bindce.iloc[6,0]
bim_dc.loc[bim_dc['month']==-15, 'predict_bim_bindce']= predict_bindce.iloc[7,0]
bim_dc.loc[bim_dc['month']==-13, 'predict_bim_bindce']= predict_bindce.iloc[8,0]
bim_dc.loc[bim_dc['month']==-11, 'predict_bim_bindce']= predict_bindce.iloc[9,0]
bim_dc.loc[bim_dc['month']==-9, 'predict_bim_bindce']= predict_bindce.iloc[10,0]
bim_dc.loc[bim_dc['month']==-7, 'predict_bim_bindce']= predict_bindce.iloc[11,0]
bim_dc.loc[bim_dc['month']==-5, 'predict_bim_bindce']= predict_bindce.iloc[12,0]
bim_dc.loc[bim_dc['month']==-3, 'predict_bim_bindce']= predict_bindce.iloc[13,0]
bim_dc.loc[bim_dc['month']==-1, 'predict_bim_bindce']= predict_bindce.iloc[14,0]
bim_dc.loc[bim_dc['month']==1, 'predict_bim_bindce']= predict_bindce.iloc[15,0]
bim_dc.loc[bim_dc['month']==3, 'predict_bim_bindce']= predict_bindce.iloc[16,0]
bim_dc.loc[bim_dc['month']==5, 'predict_bim_bindce']= predict_bindce.iloc[17,0]
bim_dc.loc[bim_dc['month']==7, 'predict_bim_bindce']= predict_bindce.iloc[18,0]
bim_dc.loc[bim_dc['month']==9, 'predict_bim_bindce']= predict_bindce.iloc[19,0]
bim_dc.loc[bim_dc['month']==11, 'predict_bim_bindce']= predict_bindce.iloc[20,0]
bim_dc.loc[bim_dc['month']==13, 'predict_bim_bindce']= predict_bindce.iloc[21,0]
bim_dc.loc[bim_dc['month']==15, 'predict_bim_bindce']= predict_bindce.iloc[22,0]
bim_dc.loc[bim_dc['month']==17, 'predict_bim_bindce']= predict_bindce.iloc[23,0]


# In[30]:


def plot_RRD_curve_bindce(data):
    
    plt.grid(True)
    p_bdce = bim_dc[['predict_bim_bindce','month']]
    plot_bdce = p_bdce.dropna()
    
    plt.xlim(-30, 20, 10)
    plt.ylim(0, 0.7, 0.1)
       
    untreat_bdce = plot_bdce[plot_bdce['month'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),untreat_bdce['predict_bim_bindce'], 1)
    plt.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')
    
    treat_bdce = plot_bdce[plot_bdce['month'] >= 0]
    m, b = np.polyfit(np.arange(0,18,2),treat_bdce['predict_bim_bindce'], 1)
    plt.plot(np.arange(0,18,2), m*np.arange(0,18,2) + b, color='green')   
 
    return


# In[31]:


#plot_RRD_curve_bindce(bim_dc)


# In[35]:


def plot_figure5_bdce(data):
    bdce_plot = bim_dc[['predict_bim_dce','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2']]
    plt.xlim(0, 0.7, 0.1)
    plt.ylim(0, 1000, 200)
    plt.axvline(x=0, color='r')
    plt.xlabel('Month of birth (0 = July 2007)')
    plt.plot(bim_dc.month, bim_dc.bim_bindce, 'o')
    plt.grid(True)
    plot_RRD_curve_bindce(data)

    plt.title("Figure 5-2 Fraction with positive day care expenditure by month of birth")
    return


# In[36]:


#plot_figure5_bdce(bim_dc)


# In[45]:


def plot_figure5(data):   
    plt.figure(figsize=(13, 4))
    plt.subplot(1, 2, 1)
    plot_figure5_dce(data)
    plot_RRD_curve_dce(data)
    plt.title("Day care expenditure by month of birth",fontsize = 13)

    
    plt.subplot(1, 2, 2)
    plot_figure5_bdce(data)
    plot_RRD_curve_bindce(data)
    plt.title("Fraction with positive day care expenditure by month of birth",fontsize = 13)
    plt.suptitle("Figure 5. Day Care Expenditure by Month of Birth (HBS 2008)", verticalalignment='bottom', fontsize=14)   
    return




