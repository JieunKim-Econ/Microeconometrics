#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib as plt
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

co_he=pd.read_stata("data/data_hbs_20110196.dta")


# In[3]:


#Age of mom and dad
#replace agemom=0 if agemom==.
#replace agedad=0 if agedad==.

co_he['agemom'].fillna(0, inplace=True)
co_he['agedad'].fillna(0, inplace=True)


# In[4]:


#Mom or dad not present
#drop nomom nodad <-> del df[<var>] or df = df.drop(<var>, axis=1)
#gen nomom=0
#gen nodad=0
#replace nomom=1 if agemom==0
#replace nodad=1 if agedad==0
#sum nomom nodad

del co_he['nomom']
co_he['nomom'] = 0
co_he.loc[co_he['agemom']==0, 'nomom'] = 1


# In[5]:


del co_he['nodad']
co_he['nodad'] = 0
co_he.loc[co_he['agedad']==0, 'nodad'] = 1


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

co_he['sec1mom']=0
co_he['sec1dad']=0
co_he['sec2mom']=0
co_he['sec2dad']=0
co_he['unimom']=0
co_he['unidad']=0

co_he.loc[co_he['educmom']==3, 'sec1mom'] = 1
co_he.loc[co_he['educdad']==3, 'sec1dad'] = 1

co_he.loc[(co_he['educmom']>3)&(co_he['educmom']<7), 'sec2mom'] = 1
co_he.loc[(co_he['educdad']>3)&(co_he['educdad']<7), 'sec2dad'] = 1

co_he.loc[(co_he['educmom']==7)|(co_he['educmom']==8), 'unimom'] = 1
co_he.loc[(co_he['educdad']==7)|(co_he['educdad']==8), 'unidad'] = 1


# In[7]:


#/* Immigrant.*/
#/* (Dummy for mom with foreing nationality.)*/
#gen immig=0
#replace immig=1 if nacmom==2 | nacmom==3

#sum immig

co_he['immig']=0
co_he.loc[(co_he['nacmom']==2) | (co_he['nacmom']==3), 'immig'] = 1
           


# In[8]:


#/* Mom not married.*/
#gen smom=0
#replace smom=1 if ecivmom!=2

co_he['smom'] = 0
co_he.loc[co_he['ecivmom']!=2, 'smom'] = 1

#sum agemom agedad nomom nodad sec1mom sec1dad sec2mom sec2dad unimom unidad immig smom
#df_he['smom'].describe()


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

co_he['sib']=0
co_he.loc[co_he['nmiem2']>1, 'sib'] = 1

co_he['age2']=co_he['agemom']*co_he['agemom']
co_he['age3']=co_he['agemom']*co_he['agemom']*co_he['agemom']

co_he['daycare_bin']=0
co_he.loc[(co_he['m_exp12312']>0) &(co_he['m_exp12312']!=np.nan), 'daycare_bin'] = 1

#df[<condition>][<var>].describe()
#df_he[(df_he['month']>-10) & (df_he['month']<9)]['sib'].describe()


# In[10]:


#Create interaction dummies
co_he['ipost_1']=co_he['post']*co_he['month']
co_he['ipost_2'] =co_he['post']*co_he['month2']

#create imes_enc dummies: imes_enc_1 for January, imes_enc_2 for February and so on
for j in range(1,13):
    co_he['imes_enc_'+str(j)] = 0
    for i in range(len(co_he)):
        if co_he.loc[i,'mes_enc'] == j:
            co_he.loc[i, 'imes_enc_'+str(j)] = 1


# In[11]:


#print(co_he.columns)


# In[ ]:





# In[12]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-1 Age of mother 
             
#RRD: Linear Regression (1)
#xi:reg agemom post month month2 i.post|month i.post|month2 
def HBS_Agemom1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<10)], 'agemom', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[13]:


#RRD: Linear Regression (2)
def HBS_Agemom2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']>-6) & (co_he['month']<5)], 'agemom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[14]:


#RRD: Linear Regression (3)
#xi:reg gastmon post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def HBS_Agemom3(co_he):    
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'agemom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[15]:


#RRD: Linear Regression (4)
#xi:reg gastmon post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def HBS_Agemom4(co_he):    
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']>-4) & (co_he['month']<3)], 'agemom', ['post'],addcons=True)
    return(result)


# In[16]:


#RRD: Linear Regression (5)
#xi:reg gastmon post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def HBS_Agemom5(co_he):  
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']>-3) & (co_he['month']<2)], 'agemom', ['post'],addcons=True)
    return(result)


# In[17]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-2 Age of father
             
#RRD: Linear Regression (1)
def HBS_Agedad1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','post','month','month2','ipost_1', 'ipost_2','month3']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'agedad', ['post','month','month2','month3','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[18]:


#RRD: Linear Regression (2)
def HBS_Agedad2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'agedad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[19]:


#RRD: Linear Regression (3)
def HBS_Agedad3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'agedad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[20]:


#RRD: Linear Regression (4)
def HBS_Agedad4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'agedad', ['post'],addcons=True)
    return(result)


# In[21]:


#RRD: Linear Regression (5)
def HBS_Agedad5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'agedad', ['post'],addcons=True)
    return(result)


# In[22]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-3 Mother Secondary
def HBS_secmom1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'sec1mom', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[23]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secmom2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'sec1mom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[24]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secmom3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'sec1mom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[25]:


#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secmom4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'sec1mom', ['post'],addcons=True)
    return(result)


# In[26]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secmom5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'sec1mom', ['post'],addcons=True)
    return(result)


# In[27]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-4 Mother Highschool
def HBS_hsmom1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'sec2mom', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[28]:


#RDD(2)
def HBS_hsmom2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'sec2mom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[29]:


#RDD(3)
def HBS_hsmom3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'sec2mom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[30]:


#RDD9m(4)

def HBS_hsmom4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'sec2mom', ['post'],addcons=True)
    return(result)


# In[31]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_hsmom5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'sec2mom', ['post'],addcons=True)
    return(result)


# In[32]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-5 Mother college
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cm1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'unimom', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[33]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cm2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'unimom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[34]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cm3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'unimom', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[35]:


#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cm4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'unimom', ['post'],addcons=True)
    return(result)


# In[36]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cm5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'unimom', ['post'],addcons=True)
    return(result)


# In[37]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-6 Father Secondary
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secdad1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'sec1dad', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[38]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secdad2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'sec1dad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[39]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secdad3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'sec1dad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[117]:


#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secdad4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'sec1dad', ['post'],addcons=True)
    return(result)


# In[41]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_secdad5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'sec1dad', ['post'],addcons=True)
    return(result)


# In[42]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-7 Father Highschool
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_hsdad1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'sec2dad', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[43]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_hsdad2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'sec2dad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[44]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_hsdad3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'sec2dad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[45]:



#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_hsdad4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'sec2dad', ['post'],addcons=True)
    return(result)


# In[46]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_hsdad5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'sec2dad', ['post'],addcons=True)
    return(result)


# In[47]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-8 Father College
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cd1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['educdad','agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'unidad', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[48]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cd2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'unidad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[49]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cd3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['educdad','agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'unidad', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[50]:


#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cd4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'unidad', ['post'],addcons=True)
    return(result)


# In[51]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_cd5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'unidad', ['post'],addcons=True)
    return(result)


# In[52]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-9 Mother immigrant
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_immig1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'immig', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[53]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_immig2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'immig', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[54]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_immig3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'immig', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[55]:


#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_immig4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'immig', ['post'],addcons=True)
    return(result)


# In[56]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_immig5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'immig', ['post'],addcons=True)
    return(result)


# In[57]:


#**********************************Table3-A Covariate Regressions**************************************************
##**********************************3-10 Not first born (sibling)**
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_sib1(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -10) & (co_he['month']<9)], 'sib', ['post','month','month2','ipost_1','ipost_2'],addcons=True)
    return(result)


# In[58]:


#RDD9m(2)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_sib2(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -7) & (co_he['month']<6)], 'sib', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[59]:


#RDD9m(3)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_sib3(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -5) & (co_he['month']<4)], 'sib', ['post','month','ipost_1'],addcons=True)
    return(result)


# In[60]:


#RDD9m(4)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_sib4(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -4) & (co_he['month']<3)], 'sib', ['post'],addcons=True)
    return(result)


# In[61]:


#RDD9m(5)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def HBS_sib5(co_he):
    Y_co= ['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib']
    X_co = ['post','month','month2','ipost_1', 'ipost_2']
    co_he_index = co_he[['agemom','agedad','sec1mom','sec2mom','unimom','sec1dad','sec2dad','unidad','immig','sib','post','month','month2','ipost_1', 'ipost_2']]

    result=mt.reg(co_he_index[(co_he['month']> -3) & (co_he['month']<2)], 'sib', ['post'],addcons=True)
    return(result)


# In[62]:


#*****************Female Labor Supply*********************************
co_ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[63]:


#1.Control variables
#gen m2 = m*m
co_ls['m2']=co_ls['m']*co_ls['m']

#No father present.*/
#gen nodad=0
#replace nodad=1 if dadid==0
co_ls['nodad']=0
co_ls.loc[co_ls['dadid']==0, 'nodad']=1


# Mother not married 
#gen smom=0
#replace smom=1 if eciv!=2
co_ls['smom']=0
co_ls.loc[co_ls['eciv']!=2, 'smom']=1

#Mother single.*/
#gen single=0
#replace single=1 if eciv==1
co_ls['single']=0
co_ls.loc[co_ls['eciv']==1, 'single']=1

#Mother separated or divorced.*/
#gen sepdiv=0
#replace sepdiv=1 if eciv==4
co_ls['sepdiv']=0
co_ls.loc[co_ls['eciv']==4, 'sepdiv']=1

# No partner in the household.*/
#gen nopart=0
#replace nopart=1 if partner==0
co_ls['nopart']=0
co_ls.loc[co_ls['partner']==0, 'nopart']=1


# In[64]:


#***************Probability of the mother being in the maternity leave period at the time of the interview
#gen pleave=0
#replace pleave=0.17 if (q==1 & m==2) | (q==2 & m==5) | (q==3 & m==8) | (q==4 & m==11)
#replace pleave=0.5 if (q==1 & m==3) | (q==2 & m==6) | (q==3 & m==9) | (q==4 & m==12)
#replace pleave=0.83 if (q==1 & m==4) | (q==2 & m==7) | (q==3 & m==10) | (q==4 & m==13)
#replace pleave=1 if (q==1 & m>4 & m<9) | (q==2 & m>7 & m<12) | (q==3 & m>10 & m<15) | (q==4 & m>13)

co_ls['pleave']=0

co_ls.loc[(co_ls['q']==1) & (co_ls['m']==2)|(co_ls['q']==2) & (co_ls['m']==5)|(co_ls['q']==3) & (co_ls['m']==8)|(co_ls['q']==4) & (co_ls['m']==11) ,'pleave']=0.17
co_ls.loc[((co_ls['q']==1) & (co_ls['m']==3)) | ((co_ls['q']==2) & (co_ls['m']==6))  | ((co_ls['q']==3) & (co_ls['m']==9)) |((co_ls['q']==4) & (co_ls['m']==12)), 'pleave'] = 0.5
co_ls.loc[((co_ls['q']==1) & (co_ls['m']==4)) | ((co_ls['q']==2) & (co_ls['m']==7)) | ((co_ls['q']==3) & (co_ls['m']==10))  | ((co_ls['q']==4) & (co_ls['m']==13)), 'pleave'] = 0.83
co_ls.loc[((co_ls['q']==1) & (co_ls['m']>4) & (co_ls['m']<9)) | ((co_ls['q']==2) & (co_ls['m']>7) & (co_ls['m']<12)) | ((co_ls['q']==3) & (co_ls['m']>10) & (co_ls['m']<15))| ((co_ls['q']==4) & (co_ls['m']>13)), 'pleave'] = 1


# In[65]:


#**********************************Table3-B Covariate Regressions**************************************************
##**********************************3-1 Age of mother 

#Create interaction dummies
co_ls['ipost_1']=co_ls['post']*co_ls['m']
co_ls['ipost_2'] =co_ls['post']*co_ls['m2']

#create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2 and so on
for j in range(1,5):
    co_ls['iq_'+str(j)] = 0
    for i in range(len(co_ls)):
        if co_ls.loc[i,'q'] == j:
            co_ls.loc[i, 'iq_'+str(j)] = 1

#RDD9m(1)
#xi: reg work post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_Agemom1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]
    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'age', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)

    return(result)


# In[66]:


#RDD6m(2)xi: reg work post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-7 & m<6, robust 
def LFS_Agemom2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]
    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']<6)], 'age', ['post','m','ipost_1'],addcons=True)
    return(result)  


# In[67]:


def LFS_Agemom3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]
    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']<4)], 'age', ['post','m','ipost_1'],addcons=True)
    return(result)  


# In[68]:


def LFS_Agemom4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]
    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']<3)], 'age', ['post'],addcons=True)
    return(result)  


# In[69]:


#RDD3m(5)xi: reg work post if m>-3 & m<2, robust 
def LFS_Agemom5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]
    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']<2)], 'age', ['post'],addcons=True)
    return(result)  


# In[70]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-2 Age of FATHER

#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_Agedad1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'agedad', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[71]:


def LFS_Agedad2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']< 6)], 'agedad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[72]:


def LFS_Agedad3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']< 4)], 'agedad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[73]:


def LFS_Agedad4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']< 3)], 'agedad', ['post'], addcons=True)
    return(result)


# In[74]:


def LFS_Agedad5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']< 2)], 'agedad', ['post'], addcons=True)
    return(result)


# In[75]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-3 Mother Secondary
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_secmom1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']<9)], 'primary', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[76]:


def LFS_secmom2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']<6)], 'primary', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[77]:


def LFS_secmom3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']<4)], 'primary', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[78]:


def LFS_secmom4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']<3)], 'primary', ['post'], addcons=True)
    return(result)


# In[79]:


def LFS_secmom5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']<2)], 'primary', ['post'], addcons=True)
    return(result)


# In[152]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-4 Mother highschool graduate
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_hsmom1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'hsgrad', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[120]:


def LFS_hsmom2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']< 6)], 'hsgrad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[82]:


def LFS_hsmom3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']< 4)], 'hsgrad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[83]:


def LFS_hsmom4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']<3)], 'hsgrad', ['post'], addcons=True)
    return(result)


# In[84]:


def LFS_hsmom5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']<2)], 'hsgrad', ['post'], addcons=True)
    return(result)


# In[85]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-5 Mother college graduate
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_cm1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'univ', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[86]:


def LFS_cm2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']< 6)], 'univ', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[87]:


def LFS_cm3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']<4)], 'univ', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[88]:


def LFS_cm4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']< 3)], 'univ', ['post'], addcons=True)
    return(result)


# In[89]:


def LFS_cm5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]


    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']< 2)], 'univ', ['post'], addcons=True)
    return(result)


# In[90]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-6 Father Secondary
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_secdad1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'primary_dad', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[91]:


def LFS_secdad2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']< 6)], 'primary_dad', ['post','m','ipost_1',], addcons=True)
    return(result)


# In[92]:


def LFS_secdad3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -6) & (co_ls['m']< 4)], 'primary_dad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[93]:


def LFS_secdad4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']<3)], 'primary_dad', ['post'], addcons=True)
    return(result)


# In[94]:


def LFS_secdad5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']< 2)], 'primary_dad', ['post'], addcons=True)
    return(result)


# In[95]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-7 Father Highschool graduate
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_hsdad1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'hsgrad_dad', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[96]:


def LFS_hsdad2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']<6)], 'hsgrad_dad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[97]:


def LFS_hsdad3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']<4)], 'hsgrad_dad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[98]:


def LFS_hsdad4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']< 3)], 'hsgrad_dad', ['post'], addcons=True)
    return(result)


# In[99]:


def LFS_hsdad5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']< 2)], 'hsgrad_dad', ['post'], addcons=True)
    return(result)


# In[101]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-8 Father college graduate
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_cd1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'univ_dad', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[102]:


def LFS_cd2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']< 6)], 'univ_dad',  ['post','m','ipost_1'], addcons=True)
    return(result)


# In[103]:


def LFS_cd3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']<4)], 'univ_dad', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[104]:


def LFS_cd4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']< 3)], 'univ_dad', ['post'], addcons=True)
    return(result)


# In[105]:


def LFS_cd5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']<2)], 'univ_dad', ['post'], addcons=True)
    return(result)


# In[106]:


#**********************************Table 3 Covariate Regressions**************************************************
##**********************************3-9 Mother immigrant
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_immig1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'immig', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[107]:


def LFS_immig2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']<6)], 'immig', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[108]:


def LFS_immig3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']< 4)], 'immig', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[109]:


def LFS_immig4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']< 3)], 'immig', ['post'], addcons=True)
    return(result)


# In[110]:


def LFS_immig5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']< 2)], 'immig', ['post'], addcons=True)
    return(result)


# In[111]:


##**********************************3-10 Not First Born (sibling)**
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def LFS_sib1(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -10) & (co_ls['m']< 9)], 'sib', ['post','m','m2','ipost_1', 'ipost_2'], addcons=True)
    return(result)


# In[123]:


def LFS_sib2(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -7) & (co_ls['m']<6)], 'sib', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[113]:


def LFS_sib3(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -5) & (co_ls['m']< 4)], 'sib', ['post','m','ipost_1'], addcons=True)
    return(result)


# In[114]:


def LFS_sib4(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -4) & (co_ls['m']< 3)], 'sib', ['post'], addcons=True)
    return(result)


# In[115]:


def LFS_sib5(co_ls):
    Y_co_ls= ['work']
    X_co_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','agedad','immig','primary','hsgrad','univ','sib','primary_dad','hsgrad_dad','univ_dad']
    co_ls_index = co_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','agedad','primary_dad','hsgrad_dad','univ_dad','immig','primary','hsgrad','univ','sib','pleave']]

    result=mt.reg(co_ls_index[(co_ls['m']> -3) & (co_ls['m']< 2)], 'sib', ['post'], addcons=True)
    return(result)


# In[ ]:





# In[161]:


#=================================Create Table 3
def table3_HBS(co_he):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': []})
        result = ('Age of mother', 'AoM_SE', 'Age of father', 'AoF_SE',
                  'Mother secondary','MS_SE','Mother highschool graduate','MHG_SE',
                  'Mother college graduate','MCG_SE','Father seconary','FS_SE',
                  'Father highschool graduate', 'FHG_SE',
                  'Father college graduate','FCG_SE','Mother immigrant','MI_SE',
                  'Not first born','NFB_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Number of months')
        
        table['HBS: Balance in Covariate'] = result
        table = table.set_index('HBS: Balance in Covariate')

#==============================Panel A. HBS
        
        #Age of mother 
        HBS_AOM1 = HBS_Agemom1(co_he)
        HBS_AOM2 = HBS_Agemom2(co_he)
        HBS_AOM3 = HBS_Agemom3(co_he)
        HBS_AOM4 = HBS_Agemom4(co_he)
        HBS_AOM5= HBS_Agemom5(co_he)
        
        hbs_aom = [HBS_AOM1.beta['post'],HBS_AOM2.beta['post'],HBS_AOM3.beta['post'], HBS_AOM4.beta['post'],HBS_AOM5.beta['post']]
        table.loc["Age of mother"] = hbs_aom
        
        #AoM Standard Error
        hbs_aom_se = [HBS_AOM1.se['post'],HBS_AOM2.se['post'],HBS_AOM3.se['post'], HBS_AOM4.se['post'], HBS_AOM5.se['post']]
        table.loc["AoM_SE"]=hbs_aom_se
     
        #Age of father 
        HBS_AOF1 = HBS_Agedad1(co_he)
        HBS_AOF2 = HBS_Agedad2(co_he)
        HBS_AOF3 = HBS_Agedad3(co_he)
        HBS_AOF4 = HBS_Agedad4(co_he)
        HBS_AOF5= HBS_Agedad5(co_he)
        
        hbs_aod = [HBS_AOF1.beta['post'],HBS_AOF2.beta['post'],HBS_AOF3.beta['post'], HBS_AOF4.beta['post'],HBS_AOF5.beta['post']]
        table.loc["Age of father"] = hbs_aod
        
        #AoF Standard Error
        hbs_aod_se = [HBS_AOF1.se['post'],HBS_AOF2.se['post'],HBS_AOF3.se['post'], HBS_AOF4.se['post'], HBS_AOF5.se['post']]
        table.loc["AoF_SE"]=hbs_aod_se
            
        #Mother Secondary 
        HBS_MS1 = HBS_secmom1(co_he)
        HBS_MS2 = HBS_secmom2(co_he)
        HBS_MS3 = HBS_secmom3(co_he)
        HBS_MS4 = HBS_secmom4(co_he)
        HBS_MS5 = HBS_secmom5(co_he)
        hbs_ms = [HBS_MS1.beta['post'],HBS_MS2.beta['post'],HBS_MS3.beta['post'], HBS_MS4.beta['post'], HBS_MS5.beta['post']]
        table.loc["Mother secondary"] = hbs_ms
        
        #MS Standard Error
        hbs_ms_se = [HBS_MS1.se['post'],HBS_MS2.se['post'],HBS_MS3.se['post'], HBS_MS4.se['post'], HBS_MS5.se['post']]
        table.loc["MS_SE"]=hbs_ms_se
 
        #Mother highschool graduate
        HBS_MHG1 = HBS_hsmom1(co_he)
        HBS_MHG2 = HBS_hsmom2(co_he)
        HBS_MHG3 = HBS_hsmom3(co_he)
        HBS_MHG4 = HBS_hsmom4(co_he)
        HBS_MHG5 = HBS_hsmom5(co_he)
        hbs_mhg = [ HBS_MHG1.beta['post'], HBS_MHG2.beta['post'], HBS_MHG3.beta['post'],  HBS_MHG4.beta['post'],  HBS_MHG5.beta['post']]
        table.loc["Mother highschool graduate"] = hbs_mhg
        
        #MHG Standard Error
        hbs_mhg_se = [ HBS_MHG1.se['post'],HBS_MHG2.se['post'],HBS_MHG3.se['post'], HBS_MHG4.se['post'], HBS_MHG5.se['post']]
        table.loc["MHG_SE"]= hbs_mhg_se
        
        #Mother college graduate
        HBS_MCG1 = HBS_cm1(co_he)
        HBS_MCG2 = HBS_cm2(co_he)
        HBS_MCG3 = HBS_cm3(co_he)
        HBS_MCG4 = HBS_cm4(co_he)
        HBS_MCG5 = HBS_cm5(co_he)
        hbs_mcg = [ HBS_MCG1.beta['post'],HBS_MCG2.beta['post'],HBS_MCG3.beta['post'], HBS_MCG4.beta['post'], HBS_MCG5.beta['post']]
        table.loc["Mother college graduate"] = hbs_mcg
        
        #MCG_SE
        hbs_mcg_se = [HBS_MCG1.se['post'],HBS_MCG2.se['post'],HBS_MCG3.se['post'], HBS_MCG4.se['post'], HBS_MCG5.se['post']]
        table.loc["MCG_SE"]=hbs_mcg_se
 
        #Father seconary
        HBS_FS1 = HBS_secdad1(co_he)
        HBS_FS2 = HBS_secdad2(co_he)
        HBS_FS3 = HBS_secdad3(co_he)
        HBS_FS4 = HBS_secdad4(co_he)
        HBS_FS5 = HBS_secdad5(co_he)
        hbs_fs = [HBS_FS1.beta['post'],HBS_FS2.beta['post'],HBS_FS3.beta['post'], HBS_FS4.beta['post'], HBS_FS5.beta['post']]
        table.loc["Father seconary"] = hbs_fs
         
        #FS_SE
        hbs_fs_se =[HBS_FS1.se['post'],HBS_FS2.se['post'],HBS_FS3.se['post'], HBS_FS4.se['post'], HBS_FS5.se['post']]
        table.loc["FS_SE"]= hbs_fs_se
        
        #'Father highschool graduate'
        HBS_FHG1 = HBS_hsdad1(co_he)
        HBS_FHG2 = HBS_hsdad2(co_he)
        HBS_FHG3 = HBS_hsdad3(co_he)
        HBS_FHG4 = HBS_hsdad4(co_he)
        HBS_FHG5 = HBS_hsdad5(co_he)
        hbs_fhg = [HBS_FHG1.beta['post'], HBS_FHG2.beta['post'], HBS_FHG3.beta['post'],  HBS_FHG4.beta['post'],  HBS_FHG5.beta['post']]
        table.loc["Father highschool graduate"] = hbs_fhg
        
        #FHG Standard Error
        hbs_fhg_se = [ HBS_FHG1.se['post'],HBS_FHG2.se['post'],HBS_FHG3.se['post'], HBS_FHG4.se['post'], HBS_FHG5.se['post']]
        table.loc["FHG_SE"]= hbs_fhg_se
        
        #Father college graduate
        HBS_FCG1 = HBS_cd1(co_he)
        HBS_FCG2 = HBS_cd2(co_he)
        HBS_FCG3 = HBS_cd3(co_he)
        HBS_FCG4 = HBS_cd4(co_he)
        HBS_FCG5 = HBS_cd5(co_he)
        hbs_FCG = [ HBS_FCG1.beta['post'],HBS_FCG2.beta['post'],HBS_FCG3.beta['post'], HBS_FCG4.beta['post'], HBS_FCG5.beta['post']]
        table.loc["Father college graduate"] = hbs_FCG
        
        #FCG_SE
        hbs_fcg_se = [HBS_FCG1.se['post'],HBS_FCG2.se['post'],HBS_FCG3.se['post'], HBS_FCG4.se['post'], HBS_FCG5.se['post']]
        table.loc["FCG_SE"]=hbs_fcg_se

        #Mother immigrant
        HBS_MI1 = HBS_immig1(co_he) 
        HBS_MI2 = HBS_immig2(co_he) 
        HBS_MI3 = HBS_immig3(co_he) 
        HBS_MI4 = HBS_immig4(co_he) 
        HBS_MI5 =HBS_immig5(co_he)

        hbs_MI = [ HBS_MI1.beta['post'],HBS_MI2.beta['post'],HBS_MI3.beta['post'], HBS_MI4.beta['post'], HBS_MI5.beta['post']]
        table.loc["Mother immigrant"] = hbs_MI
        
        #MI_SE
        hbs_MI_se = [HBS_MI1.se['post'],HBS_MI2.se['post'],HBS_MI3.se['post'], HBS_MI4.se['post'], HBS_MI5.se['post']]
        table.loc['MI_SE']=hbs_MI_se
        
        #Not first born
        HBS_NFB1 = HBS_sib1(co_he) 
        HBS_NFB2 = HBS_sib2(co_he) 
        HBS_NFB3 = HBS_sib3(co_he) 
        HBS_NFB4 = HBS_sib4(co_he) 
        HBS_NFB5 = HBS_sib5(co_he)

        hbs_NFB = [HBS_NFB1.beta['post'],HBS_NFB2.beta['post'],HBS_NFB3.beta['post'], HBS_NFB4.beta['post'], HBS_NFB5.beta['post']]
        table.loc["Not first born"] = hbs_NFB

        #'NFB_SE'
        hbs_NFB_se = [HBS_NFB1.se['post'],HBS_NFB2.se['post'],HBS_NFB3.se['post'], HBS_NFB4.se['post'], HBS_NFB5.se['post']]
        table.loc['NFB_SE']=hbs_NFB_se

        #Observations
        table=table.astype(float).round(3)
        obs =[HBS_NFB1.N, HBS_NFB2.N, HBS_NFB3.N, HBS_NFB4.N, HBS_NFB5.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N"]
        table.loc["Quadric trend in m"] = quadric
        
        #Number of months
        months = [18,12,8,6,4]
        table.loc["Number of months"] = months
        return(table)


# In[162]:


#table3_HBS(co_he)


# In[159]:


def table3_LFS(co_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': []})
        result = ('Age of mother', 'AoM_SE', 'Age of father', 'AoF_SE',
                  'Mother secondary','MS_SE','Mother highschool graduate','MHG_SE',
                  'Mother college graduate','MCG_SE','Father seconary','FS_SE',
                  'Father highschool graduate','FHG_SE',
                  'Father college graduate','FCG_SE','Mother immigrant','MI_SE',
                  'Not first born','NFB_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Number of months')

        
        table['LFS: Balance in Covariate'] = result
        table = table.set_index('LFS: Balance in Covariate')

#===================================Panel B.LFS
        #Age of mother 
        LFS_AOM1 = LFS_Agemom1(co_ls)
        LFS_AOM2 = LFS_Agemom2(co_ls)
        LFS_AOM3 = LFS_Agemom3(co_ls)
        LFS_AOM4 = LFS_Agemom4(co_ls)
        LFS_AOM5= LFS_Agemom5(co_ls)
        
        LFS_aom = [LFS_AOM1.beta['post'],LFS_AOM2.beta['post'],LFS_AOM3.beta['post'], LFS_AOM4.beta['post'],LFS_AOM5.beta['post']]
        table.loc["Age of mother"] = LFS_aom
        
        #AoM Standard Error
        LFS_aom_se = [LFS_AOM1.se['post'],LFS_AOM2.se['post'],LFS_AOM3.se['post'], LFS_AOM4.se['post'], LFS_AOM5.se['post']]
        table.loc["AoM_SE"]=LFS_aom_se
     
        #Age of father 
        LFS_AOF1 = LFS_Agedad1(co_ls)
        LFS_AOF2 = LFS_Agedad2(co_ls)
        LFS_AOF3 = LFS_Agedad3(co_ls)
        LFS_AOF4 = LFS_Agedad4(co_ls)
        LFS_AOF5= LFS_Agedad5(co_ls)
        
        LFS_aod = [LFS_AOF1.beta['post'],LFS_AOF2.beta['post'],LFS_AOF3.beta['post'], LFS_AOF4.beta['post'],LFS_AOF5.beta['post']]
        table.loc["Age of father"] = LFS_aod
        
        #AoF Standard Error
        LFS_aod_se = [LFS_AOF1.se['post'],LFS_AOF2.se['post'],LFS_AOF3.se['post'], LFS_AOF4.se['post'], LFS_AOF5.se['post']]
        table.loc["AoF_SE"]=LFS_aod_se
            
        #Mother Secondary 
        LFS_MS1 = LFS_secmom1(co_ls)
        LFS_MS2 = LFS_secmom2(co_ls)
        LFS_MS3 = LFS_secmom3(co_ls)
        LFS_MS4 = LFS_secmom4(co_ls)
        LFS_MS5 = LFS_secmom5(co_ls)
        LFS_ms = [LFS_MS1.beta['post'],LFS_MS2.beta['post'],LFS_MS3.beta['post'], LFS_MS4.beta['post'], LFS_MS5.beta['post']]
        table.loc["Mother secondary"] = LFS_ms
        
        #MS Standard Error
        LFS_ms_se = [LFS_MS1.se['post'],LFS_MS2.se['post'],LFS_MS3.se['post'], LFS_MS4.se['post'], LFS_MS5.se['post']]
        table.loc["MS_SE"]=LFS_ms_se
 
        #Mother highschool graduate
        LFS_MHG1 = LFS_hsmom1(co_ls)
        LFS_MHG2 = LFS_hsmom2(co_ls)
        LFS_MHG3 = LFS_hsmom3(co_ls)
        LFS_MHG4 = LFS_hsmom4(co_ls)
        LFS_MHG5 = LFS_hsmom5(co_ls)
        LFS_mhg = [ LFS_MHG1.beta['post'], LFS_MHG2.beta['post'], LFS_MHG3.beta['post'],  LFS_MHG4.beta['post'],  LFS_MHG5.beta['post']]
        table.loc["Mother highschool graduate"] = LFS_mhg
        
        #MHG Standard Error
        LFS_mhg_se = [ LFS_MHG1.se['post'],LFS_MHG2.se['post'],LFS_MHG3.se['post'], LFS_MHG4.se['post'], LFS_MHG5.se['post']]
        table.loc["MHG_SE"]= LFS_mhg_se
        
        #Mother college graduate
        LFS_MCG1 = LFS_cm1(co_ls)
        LFS_MCG2 = LFS_cm2(co_ls)
        LFS_MCG3 = LFS_cm3(co_ls)
        LFS_MCG4 = LFS_cm4(co_ls)
        LFS_MCG5 = LFS_cm5(co_ls)
        LFS_mcg = [ LFS_MCG1.beta['post'],LFS_MCG2.beta['post'],LFS_MCG3.beta['post'], LFS_MCG4.beta['post'], LFS_MCG5.beta['post']]
        table.loc["Mother college graduate"] = LFS_mcg
        
        #MCG_SE
        LFS_mcg_se = [LFS_MCG1.se['post'],LFS_MCG2.se['post'],LFS_MCG3.se['post'], LFS_MCG4.se['post'], LFS_MCG5.se['post']]
        table.loc["MCG_SE"]=LFS_mcg_se
 
        #Father seconary
        LFS_FS1 = LFS_secdad1(co_ls)
        LFS_FS2 = LFS_secdad2(co_ls)
        LFS_FS3 = LFS_secdad3(co_ls)
        LFS_FS4 = LFS_secdad4(co_ls)
        LFS_FS5 = LFS_secdad5(co_ls)
        LFS_fs = [LFS_FS1.beta['post'],LFS_FS2.beta['post'],LFS_FS3.beta['post'], LFS_FS4.beta['post'], LFS_FS5.beta['post']]
        table.loc["Father seconary"] = LFS_fs
         
        #FS_SE
        LFS_fs_se =[LFS_FS1.se['post'],LFS_FS2.se['post'],LFS_FS3.se['post'], LFS_FS4.se['post'], LFS_FS5.se['post']]
        table.loc["FS_SE"]= LFS_fs_se
        
        #'Father highschool graduate'
        LFS_FHG1 = LFS_hsdad1(co_ls)
        LFS_FHG2 = LFS_hsdad2(co_ls)
        LFS_FHG3 = LFS_hsdad3(co_ls)
        LFS_FHG4 = LFS_hsdad4(co_ls)
        LFS_FHG5 = LFS_hsdad5(co_ls)
        LFS_fhg = [LFS_FHG1.beta['post'], LFS_FHG2.beta['post'], LFS_FHG3.beta['post'],  LFS_FHG4.beta['post'],  LFS_FHG5.beta['post']]
        table.loc["Father highschool graduate"] = LFS_fhg
        
        #FHG Standard Error
        LFS_fhg_se = [ LFS_FHG1.se['post'],LFS_FHG2.se['post'],LFS_FHG3.se['post'], LFS_FHG4.se['post'], LFS_FHG5.se['post']]
        table.loc["FHG_SE"]= LFS_fhg_se
        
        #Father college graduate
        LFS_FCG1 = LFS_cd1(co_ls)
        LFS_FCG2 = LFS_cd2(co_ls)
        LFS_FCG3 = LFS_cd3(co_ls)
        LFS_FCG4 = LFS_cd4(co_ls)
        LFS_FCG5 = LFS_cd5(co_ls)
        LFS_FCG = [ LFS_FCG1.beta['post'],LFS_FCG2.beta['post'],LFS_FCG3.beta['post'], LFS_FCG4.beta['post'], LFS_FCG5.beta['post']]
        table.loc["Father college graduate"] = LFS_FCG
        
        #FCG_SE
        LFS_fcg_se = [LFS_FCG1.se['post'],LFS_FCG2.se['post'],LFS_FCG3.se['post'], LFS_FCG4.se['post'], LFS_FCG5.se['post']]
        table.loc["FCG_SE"]=LFS_fcg_se

        #Mother immigrant
        LFS_MI1 = LFS_immig1(co_ls) 
        LFS_MI2 = LFS_immig2(co_ls) 
        LFS_MI3 = LFS_immig3(co_ls) 
        LFS_MI4 = LFS_immig4(co_ls) 
        LFS_MI5 =LFS_immig5(co_ls)

        LFS_MI = [ LFS_MI1.beta['post'],LFS_MI2.beta['post'],LFS_MI3.beta['post'], LFS_MI4.beta['post'], LFS_MI5.beta['post']]
        table.loc["Mother immigrant"] = LFS_MI
        
        #MI_SE
        LFS_MI_se = [LFS_MI1.se['post'],LFS_MI2.se['post'],LFS_MI3.se['post'], LFS_MI4.se['post'], LFS_MI5.se['post']]
        table.loc['MI_SE']=LFS_MI_se
        
        #Not first born
        LFS_NFB1 = LFS_sib1(co_ls) 
        LFS_NFB2 = LFS_sib2(co_ls) 
        LFS_NFB3 = LFS_sib3(co_ls) 
        LFS_NFB4 = LFS_sib4(co_ls) 
        LFS_NFB5 = LFS_sib5(co_ls)

        LFS_NFB = [LFS_NFB1.beta['post'],LFS_NFB2.beta['post'],LFS_NFB3.beta['post'], LFS_NFB4.beta['post'], LFS_NFB5.beta['post']]
        table.loc["Not first born"] = LFS_NFB

        #'NFB_SE'
        LFS_NFB_se = [LFS_NFB1.se['post'],LFS_NFB2.se['post'],LFS_NFB3.se['post'], LFS_NFB4.se['post'], LFS_NFB5.se['post']]
        table.loc['NFB_SE']=LFS_NFB_se

        #Observations
        table=table.astype(float).round(3)
        LFS_obs = [LFS_NFB1.N, LFS_NFB2.N, LFS_NFB3.N, LFS_NFB4.N, LFS_NFB5.N]
        table.loc["Observations"] = LFS_obs        
               
        #Linar trend in m
        LFS_linear = ["Y","Y","Y","N","N"]
        table.loc["Linear trend in m"] = LFS_linear
        
        #Quadric trend in m
        LFS_quadric = ["Y","N","N","N","N"]
        table.loc["Quadric trend in m"] = LFS_quadric
        
        #Number of months
        LFS_months = [18,12,8,6,4]
        table.loc["Number of months"] = LFS_months
        
        return(table)


# In[160]:


#table3_LFS(co_ls)

