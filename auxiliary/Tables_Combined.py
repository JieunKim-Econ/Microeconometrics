#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import localreg
import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm

import econtools
import econtools.metrics as mt


# In[2]:


##============1. Fertility
##====================1-1 Conceptions========================
#Variables
#mesp: Month of birth
#year: Year of birth
#prem: Prematurity indicator
#semanas: Weeks of gestation at birth

df=pd.read_stata("data/data_births_20110196.dta")


# In[3]:


#1. Create month of birth variable: (0 = July 2007, 1 = August 2007, etc)

df.loc[(df['year']==2010), 'm'] = df['mesp']+29
df.loc[(df['year']==2009), 'm'] = df['mesp'] + 17 #replace m = mesp + 17 if year==2009
df.loc[(df['year']==2008), 'm'] = df['mesp'] + 5  #replace m = mesp + 5 if year==2008
df.loc[(df['year']==2007), 'm'] = df['mesp'] - 7  #replace m = mesp - 7 if year==2007
df.loc[(df['year']==2006), 'm'] = df['mesp'] - 19 #replace m = mesp - 19 if year==2006
df.loc[(df['year']==2005), 'm'] = df['mesp'] - 31 #replace m = mesp - 31 if year==2005
df.loc[(df['year']==2004), 'm'] = df['mesp'] - 43 #replace m = mesp - 43 if year==2004
df.loc[(df['year']==2003), 'm'] = df['mesp'] - 55 #replace m = mesp - 55 if year==2003
df.loc[(df['year']==2002), 'm'] = df['mesp'] - 67
df.loc[(df['year']==2001), 'm'] = df['mesp'] - 79 #replace m = mesp - 79 if year==2001
df.loc[(df['year']==2000), 'm'] = df['mesp'] - 91 #replace m = mesp - 91 if year==2000


# In[4]:


#2. Create month of conception variable.
## 2.1. Naive definition (9 months before the birth month) 
#gen mc1 = m - 9

df['mc1'] = df['m'] - 9


# In[5]:


##2.2. Naivee plus prematures. (9 months before the birth month, 8 if premature)
#gen mc2 = m - 9
#replace mc2 = m - 8 if (semanas>0 & semanas<38) | prem==2

df['mc2'] = df['m'] - 9
df.loc[((df['semanas'] > 0) & (df['semanas'] < 38)) | (df['prem']==2), 'mc2'] = df['m'] - 8 


# In[6]:


##2.3. Sophisticated (calculated based on weeks of gestation)
##(**This is the one used in the paper**)##
#gen mc3 = m - 9
#replace mc3 = m - 10 if semanas > 43 & semanas!=.
#replace mc3 = m - 8 if (semanas < 39 & semanas!=0) | prem==2

df['mc3'] = df['m'] - 9
df.loc[((df['semanas'] > 43) & (df['semanas']!=np.nan)), 'mc3'] = df['m'] - 10 
df.loc[((df['semanas'] < 39) & (df['semanas']!=0)) | (df['prem']==2), 'mc3'] = df['m'] - 8 


# In[7]:


#4-1. Collapse by month of conception. 
#gen n=1
#collapse (count) n, by(mc3)
#rename mc3 mc

df=df.groupby('mc3', as_index=False)['mc3'].agg({'n':'count'})


# In[8]:


#4-2. Collapse by month of conception.
#rename mc3 mc
#sum
df= df.rename(columns={'mc3': 'mc'})


# In[9]:


# 5. Calendar month of conception. 

df['month']=1 
df.loc[(df['mc']==-30) | (df['mc']==-18) | (df['mc']==-6) | (df['mc']==6) | (df['mc']==18) | (df['mc']==30) | (df['mc']==-42) | (df['mc']==-54) | (df['mc']==-66) | (df['mc']==-87)| (df['mc']==-99), 'month'] = 1
df.loc[(df['mc']==-29) | (df['mc']==-17) | (df['mc']==-5) | (df['mc']==7) | (df['mc']==19) | (df['mc']==31) | (df['mc']==-41) | (df['mc']==-53) | (df['mc']==-65) | (df['mc']==-86)| (df['mc']==-98), 'month'] = 2
df.loc[(df['mc']==-28) | (df['mc']==-16) | (df['mc']==-4) | (df['mc']==8) | (df['mc']==20) | (df['mc']==32) | (df['mc']==-40) | (df['mc']==-52) | (df['mc']==-64) | (df['mc']==-85)| (df['mc']==-97), 'month'] = 3
df.loc[(df['mc']==-27) | (df['mc']==-15) | (df['mc']==-3) | (df['mc']==9) | (df['mc']==21) | (df['mc']==33) | (df['mc']==-39) | (df['mc']==-51) | (df['mc']==-63) | (df['mc']==-84)| (df['mc']==-96), 'month'] = 4
df.loc[(df['mc']==-26) | (df['mc']==-14) | (df['mc']==-2) | (df['mc']==10) | (df['mc']==22) | (df['mc']==-38) | (df['mc']==-50) | (df['mc']==-62) | (df['mc']==-74)| (df['mc']==-83)| (df['mc']==-95), 'month'] = 5
df.loc[(df['mc']==-25) | (df['mc']==-13) | (df['mc']==-1) | (df['mc']==11) | (df['mc']==23) | (df['mc']==-37) | (df['mc']==-49) | (df['mc']==-61) | (df['mc']==-73)| (df['mc']==-82)| (df['mc']==-94), 'month'] = 6
df.loc[(df['mc']==-24) | (df['mc']==-12) | (df['mc']==0) | (df['mc']==12) | (df['mc']==24) | (df['mc']==-36) | (df['mc']==-48) | (df['mc']==-60) | (df['mc']==-72)| (df['mc']==-81)| (df['mc']==-93), 'month'] = 7
df.loc[(df['mc']==-23) | (df['mc']==-11) | (df['mc']==1) | (df['mc']==13) | (df['mc']==25) | (df['mc']==-35) | (df['mc']==-47) | (df['mc']==-59) | (df['mc']==-71)| (df['mc']==-80)| (df['mc']==-92), 'month'] = 8
df.loc[(df['mc']==-22) | (df['mc']==-10) | (df['mc']==2) | (df['mc']==14) | (df['mc']==26) | (df['mc']==-34) | (df['mc']==-46) | (df['mc']==-58) | (df['mc']==-70)| (df['mc']==-79)| (df['mc']==-91), 'month'] = 9
df.loc[(df['mc']==-21) | (df['mc']==-9) | (df['mc']==3) | (df['mc']==15) | (df['mc']==27) | (df['mc']==-33) | (df['mc']==-45) | (df['mc']==-57) | (df['mc']==-69)| (df['mc']==-78)| (df['mc']==-90), 'month'] = 10
df.loc[(df['mc']==-20) | (df['mc']==-8) | (df['mc']==4) | (df['mc']==16) | (df['mc']==28) | (df['mc']==-32) | (df['mc']==-44) | (df['mc']==-56) | (df['mc']==-68)| (df['mc']==-77)| (df['mc']==-89), 'month'] = 11
df.loc[(df['mc']==-19) | (df['mc']==-7) | (df['mc']==5) | (df['mc']==17) | (df['mc']==29) | (df['mc']==-31) | (df['mc']==-43) | (df['mc']==-55) | (df['mc']==-67)| (df['mc']==-76)| (df['mc']==-88), 'month'] = 12


# In[10]:


#6. July indicator 
#gen july=n if month==7

df.loc[df['month']==7, 'july'] = df['n']


# In[11]:


#7. Number of days in a month
#gen days=31
#replace days=28 if month==2
#replace days=29 if mc==7
#replace days=30 if month==4 | month==6 | month==9 | month==11

df['days'] = 31
df.loc[(df['month']==2), 'days'] = 28
df.loc[(df['mc']==7), 'days'] = 29
df.loc[(df['month']==4)| (df['month']==6)|(df['month']==9)|(df['month']==11),'days'] = 30


# In[12]:


# 8. A post indicator for post-policy conception
#gen post=0
#replace post=1 if mc>=0
#gen mc2=mc*mc
#gen mc3=mc*mc*mc

df['post']=0
df.loc[(df['mc']>=0), 'post']=1

df['mc2']=df['mc']*df['mc']
df['mc3']=df['mc']*df['mc']*df['mc']


# In[13]:


#9. Natural log
#gen ln = ln(n)

df['ln'] = np.log(df['n'])


# In[15]:


####=======================10. Regression
#Create interaction dummies
df['ipost_1']=df['post']*df['mc']
df['ipost_2'] =df['post']*df['mc2']
df['ipost_3'] =df['post']*df['mc3']

#RRD: Linear Regression (1)
#xi: reg ln post i.post|mc i.post|mc2 i.post|mc3 days if mc>-91 & mc<30, robust

def RD_Conception_1(df):
    Y= 'ln'
    X = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_index = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_index[(df['mc']> -91) & (df['mc']< 30)], 'ln', ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days'], addcons=True)
    return(result)


# In[16]:


#RRD: Linear Regression (2)
#xi: reg ln post i.post|mc i.post|mc2 days if mc>-31 & mc<30, robust

def RD_Conception_2(df):
    Y= 'ln'
    X = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_index = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_index[(df['mc']> -31) & (df['mc']< 30)], 'ln', ['mc','mc2','post', 'ipost_1', 'ipost_2','days'], addcons=True)
    return(result)


# In[17]:


#RRD: Linear Regression (3)
#xi: reg ln post i.post|mc i.post|mc2 days if mc>-13 & mc<12, robust

def RD_Conception_3(df):
    Y= 'ln'
    X = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_index = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_index[(df['mc']> -13) & (df['mc']< 12)], 'ln', ['mc','mc2','post', 'ipost_1', 'ipost_2','days'], addcons=True)
    return(result)


# In[18]:


#RRD: Linear Regression (4)
#xi: reg ln post i.post|mc days if mc>-10 & mc<9, robust

def RD_Conception_4(df):
    Y= 'ln'
    X = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_index = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_index[(df['mc']> -10) & (df['mc']<9)], 'ln', ['mc','post', 'ipost_1','days'], addcons=True)
    return(result)


# In[19]:


#RRD: Linear Regression (5)
#xi: reg ln post days if mc>-4 & mc<3, robust

def RD_Conception_5(df):
    Y= 'ln'
    X = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_index = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_index[(df['mc']> -4) & (df['mc']<3)], 'ln', ['post','days'], addcons=True)
    return(result)


# In[20]:


#DID (6) including "i.month"
#create imonth dummies
# imonth_1 for January, imonth2 for February, imonth3 for March, and so on
for j in range(1,13):
    df['imonth_'+str(j)] = 0
    for i in range(len(df)):
        if df.loc[i,'month'] == j:
            df.loc[i, 'imonth_'+str(j)] = 1
#(Alternative way for creating dummies)
#df['imonth_i']=np.NaN
#df['imonth_i'] = pd.get_dummies(df[df['month']==i])
#df['imonth_i'] = df['imonth_1i] / df['mc']

#xi: reg ln post i.post|mc i.post|mc2 i.post|mc3 days i.month if mc>-91 & mc<30, robust /* 10 years */

def DID_Conception_6(df):
    X_DID = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']
    df_DID = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']]
    mt.reg(df_DID[(df['mc']> -91) & (df['mc']<30)], 'ln', ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    result=mt.reg(df_DID[(df['mc']> -91) & (df['mc']<30)], 'ln', ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    return(result)


# In[21]:


#DID (7)
#xi: reg ln post i.post|mc i.post|mc2 days i.month if mc>-67 & mc<30, robust /* 8 years */
#mt.reg(df_DID[(df['mc']> -67) & (df['mc']<30)], 'ln', ['mc','mc2','post', 'ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)

def DID_Conception_7(df):
    X_DID = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']
    df_DID = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']]
    mt.reg(df_DID[(df['mc']> -91) & (df['mc']<30)], 'ln', ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    result=mt.reg(df_DID[(df['mc']> -67) & (df['mc']<30)], 'ln', ['mc','mc2','post', 'ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    return(result)


# In[22]:


#DID (8)
#xi: reg ln post i.post|mc i.post|mc2 days i.month if mc>-31 & mc<30, robust /* 5 years */
#mt.reg(df_DID[(df['mc']> -31) & (df['mc']<30)], 'ln', ['mc','mc2','post', 'ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)

def DID_Conception_8(df):
    X_DID = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']
    df_DID = df[['ln','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']]
    mt.reg(df_DID[(df['mc']> -91) & (df['mc']<30)], 'ln', ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    result=mt.reg(df_DID[(df['mc']> -31) & (df['mc']<30)], 'ln', ['mc','mc2','post', 'ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    return(result)


# In[23]:


##====================1-2 Abortions========================
#1&2 access data and summary
df_ab=pd.read_stata("data/data_abortions_20110196.dta")


# In[24]:


#3. Sum abortions across all regions
#gen n_tot = n_ive_and + n_ive_val + n_ive_rioja + n_ive_cat + n_ive_can + n_ive_mad + n_ive_gal + n_ive_bal + n_ive_pv + n_ive_castlm + n_ive_ast + n_ive_arag 
df_ab['n_tot']=pd.concat([df_ab.n_ive_and, df_ab.n_ive_val, df_ab.n_ive_rioja, df_ab.n_ive_cat, df_ab.n_ive_can,df_ab.n_ive_mad, df_ab.n_ive_gal, df_ab.n_ive_bal, df_ab.n_ive_pv, df_ab.n_ive_castlm, df_ab.n_ive_ast, df_ab.n_ive_arag],1).sum(1)
#df_ab


# In[25]:


#4. Create month variable that takes value 0 in July 2007.*
#gen m = _n
#replace m = m - 103 

#m = the number of current observation (_n in STATA) Since python row starts from 0, I add 1
df_ab['m']= df_ab.reset_index().index + 1 
df_ab['m'] = df_ab['m'] - 103


# In[26]:


#5. Now I generate a variable indicating number of days in a month
#gen days=31
#replace days=30 if month==4 | month==6 | month==9 | month==11
#replace days=28 if month==2
#replace days=29 if month==2 & (year==2000 | year==2004 | year==2008)
#sum days

df_ab['days']=31
df_ab.loc[(df_ab['month']==4) | (df_ab['month']==6) | (df_ab['month']==9) | (df_ab['month']==11), 'days'] = 30
df_ab.loc[(df_ab['month']==2), 'days'] = 28
df_ab.loc[(df_ab['month']==2) & ((df_ab['year']==2000) | (df_ab['year']==2004) | (df_ab['year']==2008)), 'days'] = 29


# In[27]:


# 6. Now I generate log-abortions
#gen log_ive = ln(n_tot)

df_ab['log_ive'] = np.log(df_ab['n_tot'])


# In[28]:


# 7&8. Squared and cubed terms in m & Create Post dummy
#gen m2=m*m
#gen m3=m*m*m

df_ab['m2'] = df_ab['m']*df_ab['m']
df_ab['m3'] = df_ab['m']*df_ab['m']*df_ab['m']

#gen post=0
#replace post=1 if m>=0

df_ab['post']=0
df_ab.loc[(df_ab['m']>=0), 'post']=1


# In[29]:


# 9. Restrict period
#drop if m<-90 
#drop if m>29

df_ab = df_ab[~(df_ab.m < -90) & ~(df_ab.m > 29)]


# In[30]:


############======================= 10. [Abortions] Regressions 
#Create interaction dummies
df_ab['ipost_1']=df_ab['post']*df_ab['m']
df_ab['ipost_2'] =df_ab['post']*df_ab['m2']
df_ab['ipost_3'] =df_ab['post']*df_ab['m3']

#RRD: Linear Regression (1)
#xi: reg log_ive post i.post|m i.post|m2 i.post|m3 days, robust
def RD_Abortion_1(df_ab):
    y= ['log_ive']
    x = ['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_group, 'log_ive',['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days'], addcons=True)
    return(result)


# In[31]:


#RRD: Linear Regression (2)
#xi: reg log_ive post i.post|m i.post|m2 days if m>-31, robust


def RD_Abortion_2(df_ab):
    y= ['log_ive']
    x = ['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_group[(df_group['m']> -31)], 'log_ive',['m','m2','post','ipost_1', 'ipost_2', 'days'], addcons=True)
    return(result)


# In[32]:


#RRD: Linear Regression (3)
#xi: reg log_ive post i.post|m i.post|m2 days if m>-13 & m<12, robust
def RD_Abortion_3(df_ab):
    y= ['log_ive']
    x = ['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_group[(df_group['m']> -13)&(df_group['m']<12)], 'log_ive',['m','m2','post','ipost_1', 'ipost_2', 'days'], addcons=True)
    return(result)


# In[33]:


#RRD: Linear Regression (4)
#xi: reg log_ive post i.post|m days if m>-10 & m<9, robust
def RD_Abortion_4(df_ab):
    y= ['log_ive']
    x = ['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_group[(df_group['m']> -10)&(df_group['m']<9)], 'log_ive',['m','post','ipost_1', 'days'], addcons=True)    
    return(result)


# In[34]:


#RRD: Linear Regression (5)
#xi: reg log_ive post days if m>-4 & m<3, robust
def RD_Abortion_5(df_ab):
    y= ['log_ive']
    x = ['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_group[(df_group['m']> -4)&(df_group['m']<3)], 'log_ive',['post', 'days'], addcons=True)    
    return(result)


# In[35]:


#create imonth dummies for DID
# imonth_1 for January, imonth_2 for February, and so on
for j in range(1,13):
    df_ab['imonth_'+str(j)] = 0
    for i in range(12,132):
        if df_ab.loc[i,'month'] == j:
            df_ab.loc[i, 'imonth_'+str(j)] = 1


# In[36]:


#DID (6) with "imonth" for 10 years
#xi: reg log_ive post i.post|m i.post|m2 i.post|m3 days i.month, robust /* 10 years */
def DID_Abortion_6(df_ab):
    x_DID = ['m','m','m','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']
    df_DID = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']]
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_DID, 'log_ive',['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)    
    return(result)


# In[37]:


#DID (7) with "imonth" for 8 years
#xi: reg log_ive post i.post|m i.post|m2 days i.month if m>-67, robust /* 8 years */
#mt.reg(df_DID[(df_group['m']> -67)], 'log_ive',['m','m2','post','ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)

def DID_Abortion_7(df_ab):
    x_DID = ['m','m','m','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']
    df_DID = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']]
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_DID[(df_group['m']> -67)], 'log_ive',['m','m2','post','ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)
    return(result)


# In[38]:


#DID (8) with "imonth" for 5 years
#xi: reg log_ive post i.post|m i.post|m2 days i.month if m>-31, robust /* 5 years */

def DID_Abortion_8(df_ab):
    x_DID = ['m','m','m','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']
    df_DID = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12']]
    df_group = df_ab[['log_ive','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    result=mt.reg(df_DID[(df_group['m']> -31)], 'log_ive',['m','m2','post','ipost_1', 'ipost_2', 'days', 'imonth_2', 'imonth_3', 'imonth_4', 'imonth_5', 'imonth_6', 'imonth_7', 'imonth_8', 'imonth_9','imonth_10', 'imonth_11', 'imonth_12'], addcons=True)    
    return(result)


# In[39]:


#============Create Table 2==================================#
def table2(df,df_ab):
        table = pd.DataFrame({'RDD_10yrs(1)': [], 'RDD_5yrs(2)': [], 'RDD_12-12m(3)': [],
                          'RDD_9-9m(4)': [], 'RDD_3-3m(5)': [], 'DID_10yrs(6)': [],
                          'DID_7yrs(7)': [], 'DID_5yrs(8)': []})
        result = ('Conceptions', 'Con_SE', 'Abortions', 'Abo_SE', 'Years included', 'Number of months', 'Linear trend in m', 
               'Quadric trend in m', 'Cubic trend in m', 'Number days of month', 'Calendar month dummies')
        table['Fertility'] = result
        table = table.set_index('Fertility')
        
        #Conception coefficient of post
        C_RD1 = RD_Conception_1(df)
        C_RD2 = RD_Conception_2(df)
        C_RD3 = RD_Conception_3(df)
        C_RD4 = RD_Conception_4(df)
        C_RD5 = RD_Conception_5(df)
        C_DID6 = DID_Conception_6(df)
        C_DID7 = DID_Conception_7(df)
        C_DID8 = DID_Conception_8(df)
        conceptions = [C_RD1.beta['post'],C_RD2.beta['post'],C_RD3.beta['post'], C_RD4.beta['post'], C_RD5.beta['post'],
                 C_DID6.beta['post'], C_DID7.beta['post'], C_DID8.beta['post']]
        table.loc["Conceptions"] = conceptions
        
        #Conception Standard Error
        con_se = [C_RD1.se['post'],C_RD2.se['post'],C_RD3.se['post'], C_RD4.se['post'], C_RD5.se['post'],
                 C_DID6.se['post'], C_DID7.se['post'], C_DID8.se['post']]
        table.loc["Con_SE"]=con_se
         
        #Abortion coefficient of post
        A_RD1 = RD_Abortion_1(df_ab)
        A_RD2 = RD_Abortion_2(df_ab)
        A_RD3 = RD_Abortion_3(df_ab)
        A_RD4 = RD_Abortion_4(df_ab)
        A_RD5 = RD_Abortion_5(df_ab)
        A_DID6 = DID_Abortion_6(df_ab)
        A_DID7 = DID_Abortion_7(df_ab)
        A_DID8 = DID_Abortion_8(df_ab)
        abortions = [A_RD1.beta['post'],A_RD2.beta['post'],A_RD3.beta['post'], A_RD4.beta['post'], A_RD5.beta['post'],
                 A_DID6.beta['post'], A_DID7.beta['post'], A_DID8.beta['post']]
        table.loc["Abortions"] = abortions
        
        #Abortion Standard Error
        Abo_se = [A_RD1.se['post'],A_RD2.se['post'],A_RD3.se['post'], A_RD4.se['post'], A_RD5.se['post'],
                 A_DID6.se['post'], A_DID7.se['post'], A_DID8.se['post']]
        table.loc["Abo_SE"] = Abo_se
    
        #Years included
        table=table.astype(float).round(4)
        years =["2000_2009","2005_2009","2006_2008","2006_2008","2007", "2000_2009","2003_2009","2005_2009"]
        table.loc["Years included"] = years        
        
        #Number of months
        months = [C_RD1.N, C_RD2.N, C_RD3.N, C_RD4.N, C_RD5.N, C_DID6.N, C_DID7.N, C_DID8.N]
        table.loc["Number of months"] = months
        
        #Linar trend in m
        linear = ["Y","Y","Y","Y","N","Y","Y","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","Y","Y","N","N","Y","Y","Y"]
        table.loc["Quadric trend in m"] = quadric
        
        #Cubic trend in m
        cubic = ["Y","N","N","N","N","Y","N","N"]
        table.loc["Cubic trend in m"] = cubic
        
        #Number days of month
        days_m = ["Y","Y","Y","Y","Y","Y","Y","Y"]
        table.loc["Number days of month"] = days_m
        
        #Calendar month dummies
        dummies = ["N","N","N","N","N","Y","Y","Y"]
        table.loc["Calendar month dummies"] = dummies
    
        
        return(table)

table2(df,df_ab)


# In[40]:


##=============================2. Household Expenditure
#1. Control variables
##==Characteristics: age of mom and dad, education of mom and dad, mom foreing, mom not married, mom or dad not present
#sum agemom agedad educmom educdad nacmom ecivmom

df_he=pd.read_stata("data/data_hbs_20110196.dta")


# In[41]:


#Age of mom and dad
#replace agemom=0 if agemom==.
#replace agedad=0 if agedad==.

df_he['agemom'].fillna(0, inplace=True)
df_he['agedad'].fillna(0, inplace=True)


# In[42]:


#Mom or dad not present
#drop nomom nodad <-> del df[<var>] or df = df.drop(<var>, axis=1)
#gen nomom=0
#gen nodad=0
#replace nomom=1 if agemom==0
#replace nodad=1 if agedad==0
#sum nomom nodad

del df_he['nomom']
df_he['nomom'] = 0
df_he.loc[df_he['agemom']==0, 'nomom'] = 1


# In[43]:


del df_he['nodad']
df_he['nodad'] = 0
df_he.loc[df_he['agedad']==0, 'nodad'] = 1


# In[44]:


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

df_he['sec1mom']=0
df_he['sec1dad']=0
df_he['sec2mom']=0
df_he['sec2dad']=0
df_he['unimom']=0
df_he['unidad']=0

df_he.loc[df_he['educmom']==3, 'sec1mom'] = 1
df_he.loc[df_he['educdad']==3, 'sec1dad'] = 1

df_he.loc[(df_he['educmom']>3)&(df_he['educmom']<7), 'sec2mom'] = 1
df_he.loc[(df_he['educdad']>3)&(df_he['educdad']<7), 'sec2dad'] = 1

df_he.loc[(df_he['educmom']==7)|(df_he['educmom']==8), 'unimom'] = 1
df_he.loc[(df_he['educdad']==7)|(df_he['educdad']==8), 'unidad'] = 1


# In[45]:


#/* Immigrant.*/
#/* (Dummy for mom with foreing nationality.)*/
#gen immig=0
#replace immig=1 if nacmom==2 | nacmom==3

#sum immig

df_he['immig']=0
df_he.loc[(df_he['nacmom']==2) | (df_he['nacmom']==3), 'immig'] = 1


# In[47]:


#/* Mom not married.*/
#gen smom=0
#replace smom=1 if ecivmom!=2

df_he['smom'] = 0
df_he.loc[df_he['ecivmom']!=2, 'smom'] = 1

#sum agemom agedad nomom nodad sec1mom sec1dad sec2mom sec2dad unimom unidad immig smom


# In[48]:


#/* Siblings.*/
#gen sib=0
#replace sib=1 if nmiem2>1

#gen age2=agemom*agemom
#gen age3=agemom*agemom*agemom

#gen daycare_bin=0
#replace daycare_bin=1 if m_exp12312>0 & m_exp12312!=.

#sum gastmon c_m_exp dur_exp m_exp12312 post month agemom sec1mom sec2mom unimom immig sib if month>-10 & month<9
#centile gastmon c_m_exp dur_exp m_exp12312 post month agemom sec1mom sec2mom unimom immig sib if month>-10 & month<9

df_he['sib']=0
df_he.loc[df_he['nmiem2']>1, 'sib'] = 1

df_he['age2']=df_he['agemom']*df_he['agemom']
df_he['age3']=df_he['agemom']*df_he['agemom']*df_he['agemom']

df_he['daycare_bin']=0
df_he.loc[(df_he['m_exp12312']>0) &(df_he['m_exp12312']!=np.nan), 'daycare_bin'] = 1


# In[49]:


#*************************TABLE 4 [Regression] MAIN EXPENDITURE RESULTS******************************************************************************/
#***************************TABLE 4-1.Total expenditure.*************************************************/
#Create interaction dummies
df_he['ipost_1']=df_he['post']*df_he['month']
df_he['ipost_2'] =df_he['post']*df_he['month2']

#create imes_enc dummies: imes_enc_1 for January, imes_enc_2 for February and so on
for j in range(1,13):
    df_he['imes_enc_'+str(j)] = 0
    for i in range(len(df_he)):
        if df_he.loc[i,'mes_enc'] == j:
            df_he.loc[i, 'imes_enc_'+str(j)] = 1

df_he['ltotexp']=np.log(df_he['gastmon'])
df_he['lcexp']=np.log(df_he['c_m_exp'].replace(0, np.nan))
df_he['ldurexp']=np.log(df_he['dur_exp'].replace(0, np.nan))
    
#RRD: Linear Regression (1)
#xi:reg gastmon post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_TE_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'gastmon', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
    addcons=True)
    return(result)


# In[50]:


#RRD: Linear Regression (2)
#xi:reg gastmon post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_TE_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']<6)], 'gastmon', 
       ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
     addcons=True)
    return(result)


# In[51]:


#RRD: Linear Regression (3)
#xi:reg gastmon post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust

def RD_TE_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']<4)], 'gastmon', 
       ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
     addcons=True)
    return(result)


# In[52]:


#RRD: Linear Regression (4)
#xi:reg gastmon post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_TE_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']<3)], 'gastmon', 
       ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
     addcons=True)
    return(result)


# In[53]:


#RRD: Linear Regression (5)
#xi:reg gastmon post if month>-3 & month<2, robust
def RD_TE_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]



    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'gastmon', ['post'], addcons=True)
    return(result)


# In[54]:


#RRD: Linear Regression (6)
#xi:reg gastmon post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_TE_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'gastmon', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], addcons=True)
    return(result)


# In[55]:


#DID (7) with "i.n_month": Calendar month of birth dummies
#create in_month: in_month_1 for January, in_month_2 for February and so on
for j in range(1,13):
    df_he['i_n_month_'+str(j)] = 0
    for i in range(len(df_he)):
        if df_he.loc[i,'n_month'] == j:
            df_he.loc[i, 'i_n_month_'+str(j)] = 1

#xi:reg gastmon post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc i.n_month, robust

def DID_TE_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]
    
    result=mt.reg(df_he_DID, 'gastmon', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    
    return(result)


# In[56]:


#[수정. 수수께끼니. 이건 어디서 튀어나온거니...] DID (8) with "i.n_month" (No Quadratic in m)
#xi:reg gastmon post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc i.n_month, robust
def DID_TE_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]
    
    result=mt.reg(df_he_DID, 'gastmon', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    
    return(result)


# In[57]:


#***************************TABLE 4-2.Child-related expenditure.*************************************************/
#xi:reg c_m_exp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
#RRD: Linear Regression (1)

def RD_CRE_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]
    
    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'c_m_exp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[58]:


#RRD: Linear Regression (2)
#xi:reg c_m_exp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_CRE_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]
    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']<6)], 'c_m_exp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[59]:


#RRD: Linear Regression (3)
#xi:reg c_m_exp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_CRE_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']<4)], 'c_m_exp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[60]:


#RRD: Linear Regression (4)
#xi:reg c_m_exp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_CRE_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]



    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']<3)], 'c_m_exp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[61]:


#RRD: Linear Regression (5)
#xi:reg c_m_exp post if month>-3 & month<2, robust
def RD_CRE_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'c_m_exp', ['post'], addcons=True)
    return(result)


# In[62]:


#RRD: Linear Regression (6)
#xi:reg c_m_exp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_CRE_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'c_m_exp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[63]:


#DID (7) with "i.n_month" (No Quadratic in m)
#xi:reg c_m_exp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc i.n_month, robust

def DID_CRE_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]
    
    result=mt.reg(df_he_DID, 'c_m_exp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)

    return(result)


# In[64]:


#[수정. 수수께끼니. 이건 어디서 튀어나온거니...] DID (8) with "i.n_month" (No Quadratic in m)
def DID_CRE_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'c_m_exp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[65]:


#***************************TABLE 4-3.Durable goods expenditure.*************************************************/
#RRD: Linear Regression (1)
#xi:reg dur_exp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_DGE_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'dur_exp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[66]:


#RRD: Linear Regression (2)
#xi:reg dur_exp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_DGE_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']< 6)], 'dur_exp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1',
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[67]:


#RRD: Linear Regression (3)
#xi:reg dur_exp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_DGE_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']< 4)], 'dur_exp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[68]:


#RRD: Linear Regression (4)
#xi:reg dur_exp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_DGE_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']< 3)], 'dur_exp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[69]:


#RRD: Linear Regression (5)
#xi:reg dur_exp post if month>-3 & month<2, robust
def RD_DGE_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'dur_exp', ['post'], addcons=True)
    return(result)


# In[70]:


#RRD: Linear Regression (6)
#xi:reg dur_exp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_DGE_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']< 2)], 'dur_exp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[71]:


#DID (7)
#xi:reg dur_exp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc i.n_month, robust
def DID_DGE_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]


    result=mt.reg(df_he_DID, 'dur_exp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[72]:


#DID (8)
#xi:reg dur_exp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc i.n_month, robust
def DID_DGE_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]


    result=mt.reg(df_he_DID, 'dur_exp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[73]:


#***************************TABLE 4-4.[LOG] Total expenditure.*************************************************/

#RRD: Linear Regression (1)
#xi:reg ltotexp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_logTE_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'ltotexp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[74]:


#RRD: Linear Regression (2)
#xi:reg ltotexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_logTE_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']<6)], 'ltotexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
     addcons=True)
    return(result)


# In[75]:


#RRD: Linear Regression (3)
#xi:reg ltotexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_logTE_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']<4)], 'ltotexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
     addcons=True)
    return(result)


# In[76]:


#RRD: Linear Regression (4)
#xi:reg ltotexp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_logTE_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']<3)], 'ltotexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
     addcons=True)
    return(result)


# In[77]:


#RRD: Linear Regression (5)
#xi:reg ltotexp post if month>-3 & month<2, robust
def RD_logTE_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]
    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'ltotexp', ['post'], addcons=True)
    return(result)


# In[78]:


#RRD: Linear Regression (6)
#xi:reg ltotexp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_logTE_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'ltotexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], addcons=True)
    return(result)


# In[79]:


#DID (7)
#xi:reg dur_exp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc i.n_month, robust
def DID_logTE_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]


    result=mt.reg(df_he_DID, 'ltotexp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[80]:


#[수정] DID (8)
#xi:reg ltotexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i i.mes_enc i.n_month, robust
def DID_logTE_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]


    result=mt.reg(df_he_DID, 'ltotexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[81]:


#***************************TABLE 4-5.[LOG] Child-related expenditure.*************************************************/
#RRD: Linear Regression (1)
#xi:reg lcexp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_logCRE_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'lcexp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[82]:


#RRD: Linear Regression (2)
#xi:reg lcexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_logCRE_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']<6)], 'lcexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[83]:


#RRD: Linear Regression (3)
#xi:reg lcexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_logCRE_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']<4)], 'lcexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[84]:


#RRD: Linear Regression (4)
#xi:reg lcexp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_logCRE_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']<3)], 'lcexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[85]:


#RRD: Linear Regression (5)
#xi:reg lcexp post if month>-3 & month<2, robust
def RD_logCRE_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'lcexp', ['post'], addcons=True)
    return(result)


# In[86]:


#RRD: Linear Regression (6)
#xi:reg lcexp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_logCRE_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'lcexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[87]:


#DID (7)
#xi:reg lcexp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc i.n_month, robust
def DID_logCRE_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'lcexp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[88]:


#DID (8)
#xi:reg lcexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc i.n_month, robust
def DID_logCRE_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]


    result=mt.reg(df_he_DID, 'lcexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[89]:


#***************************TABLE 4-6.[LOG] Durable goods expenditure*************************************************/

#RRD: Linear Regression (1)
#xi:reg ldurexp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_logDGE_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'ldurexp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[90]:


#RRD: Linear Regression (2)
#xi:reg ldurexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_logDGE_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']< 6)], 'ldurexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1',
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[91]:


#RRD: Linear Regression (3)
#xi:reg ldurexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_logDGE_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']< 4)], 'ldurexp', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[92]:


#RRD: Linear Regression (4)
#xi:reg ldurexp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_logDGE_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']< 3)], 'ldurexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[93]:


#RRD: Linear Regression (5)
#xi:reg ldurexp post if month>-3 & month<2, robust
def RD_logDGE_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'ldurexp', ['post'], addcons=True)
    return(result)


# In[94]:


#RRD: Linear Regression (6)
#xi:reg ldurexp post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_logDGE_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']< 2)], 'ldurexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[95]:


#DID (7)
#xi:reg ldurexp post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc i.n_month, robust
def DID_logDGE_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'ldurexp', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[96]:


#DID (8)
#xi:reg ldurexp post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc i.n_month, robust
def DID_logDGE_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]
    
    result=mt.reg(df_he_DID, 'ldurexp', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[97]:


#**************************TABLE 6. CHILDCARE RESULTS*****************************************************************************/

##========== 6-1 Private childcare 
#(1)xi:reg m_exp12312 post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_PR_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'm_exp12312', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[98]:


#(2)xi:reg m_exp12312 post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_PR_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']< 6)], 'm_exp12312', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1',
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[99]:


#(3)xi:reg m_exp12312 post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_PR_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]
    
    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']< 4)], 'm_exp12312', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[100]:


#(4)xi:reg m_exp12312 post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_PR_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]
    
    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']< 3)], 'm_exp12312', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[101]:


#(5)xi:reg m_exp12312 post if month>-3 & month<2, robust
def RD_PR_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]
    
    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'm_exp12312', ['post'], addcons=True)
    return(result)


# In[102]:


#(6)xi:reg m_exp12312 post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_PR_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']< 2)], 'm_exp12312', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[103]:


#(7)xi:reg m_exp12312 post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc i.n_month, robust
def DID_PR_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'm_exp12312', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[104]:


#(8)xi:reg m_exp12312 post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc i.n_month, robust
def DID_PR_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'm_exp12312', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[105]:


##========== 6-2 [BINARY] Private childcare 
#RDD9m(1)xi:reg daycare_bin post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.post|month2 i.mes_enc if month>-10 & month<9, robust
def RD_binPR_1(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -10) & (df_he['month']< 9)], 'daycare_bin', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[106]:


#RDD6m(2)xi:reg daycare_bin post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-7 & month<6, robust
def RD_binPR_2(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    
    result=mt.reg(df_he_index[(df_he['month']> -7) & (df_he['month']< 6)], 'daycare_bin', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1',
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[107]:


#RDD4m(3)xi:reg daycare_bin post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.post|month i.mes_enc if month>-5 & month<4, robust
def RD_binPR_3(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -5) & (df_he['month']< 4)], 'daycare_bin', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[108]:


#RDD3m(4)xi:reg daycare_bin post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-4 & month<3, robust
def RD_binPR_4(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]


    result=mt.reg(df_he_index[(df_he['month']> -4) & (df_he['month']< 3)], 'daycare_bin', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[109]:


#RDD2m(5)xi:reg daycare_bin post if month>-3 & month<2, robust
def RD_binPR_5(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']<2)], 'daycare_bin', ['post'], addcons=True)
    return(result)


# In[110]:


#RDD2m(6)xi:reg daycare_bin post nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc if month>-3 & month<2, robust
def RD_binPR_6(df_he):
    Y_he= ['gastmon', 'c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
    'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']

    df_he_index = df_he[['c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin',
                     'gastmon','post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1', 'ipost_2', 
                     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12']]

    result=mt.reg(df_he_index[(df_he['month']> -3) & (df_he['month']< 2)], 'daycare_bin', ['post','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib', 
     'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'], 
      addcons=True)
    return(result)


# In[111]:


#DID(7)xi:reg daycare_bin post month month2 nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc i.n_month, robust
def DID_binPR_7(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'daycare_bin', ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[112]:


#DID(8)xi:reg daycare_bin post month nomom agemom age2 age3 sec1mom sec2mom unimom immig sib i.mes_enc i.n_month, robust
def DID_binPR_8(df_he):
    Y_he_DID = ['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin']
    X_he_DID = ['post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12',
            'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_he_DID = df_he[['gastmon','c_m_exp','dur_exp','ltotexp','lcexp','ldurexp','m_exp12312', 'daycare_bin'
            ,'post','month','month2','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib','ipost_1','ipost_2'
            ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
            ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_he_DID, 'daycare_bin', ['post','month','nomom','agemom','age2','age3','sec1mom','sec2mom','unimom','immig','sib'
    ,'imes_enc_2','imes_enc_3','imes_enc_4','imes_enc_5','imes_enc_6','imes_enc_7','imes_enc_8','imes_enc_9','imes_enc_10','imes_enc_11','imes_enc_12'
    ,'i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
    ,addcons=True)
    return(result)


# In[113]:


#=================================Create Table 4
def table4(df_he):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID_1(7)': [], 'DID_2(8)': []})
        result = ('Total Expenditure', 'TE_SE', 'logTotal Expenditure', 'logTE_SE',
                  'Child Related Expenditure','CRE_SE','logChild Related Expenditure','logCRE_SE',
                  'Durable Goods Expenditure','DGE_SE','logDurable Goods Expenditure','logDGE_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Expenditure'] = result
        table = table.set_index('Expenditure')
        
        #Total Expenditure coefficient of post
        TE_RD1 = RD_TE_1(df_he)
        TE_RD2 = RD_TE_2(df_he)
        TE_RD3 = RD_TE_3(df_he)
        TE_RD4 = RD_TE_4(df_he)
        TE_RD5 = RD_TE_5(df_he)
        TE_RD6 = RD_TE_6(df_he)
        TE_DID7 = DID_TE_7(df_he)
        TE_DID8 = DID_TE_8(df_he)
        te = [TE_RD1.beta['post'],TE_RD2.beta['post'],TE_RD3.beta['post'], TE_RD4.beta['post'], TE_RD5.beta['post'],
                 TE_RD6.beta['post'], TE_DID7.beta['post'], TE_DID8.beta['post']]
        table.loc["Total Expenditure"] = te
        
        #TE Standard Error
        te_se = [TE_RD1.se['post'],TE_RD2.se['post'],TE_RD3.se['post'], TE_RD4.se['post'], TE_RD5.se['post'],
                 TE_RD6.se['post'], TE_DID7.se['post'], TE_DID8.se['post']]
        table.loc["TE_SE"]=te_se
         
        #LOG TE coefficient of post 
        lgTE_RD1 = RD_logTE_1(df_he)
        lgTE_RD2 = RD_logTE_2(df_he)
        lgTE_RD3 = RD_logTE_3(df_he)
        lgTE_RD4 = RD_logTE_4(df_he)
        lgTE_RD5 = RD_logTE_5(df_he)
        lgTE_RD6 = RD_logTE_6(df_he)
        lgTE_DID7 = DID_logTE_7(df_he)
        lgTE_DID8 = DID_logTE_8(df_he)
        lgte = [lgTE_RD1.beta['post'],lgTE_RD2.beta['post'],lgTE_RD3.beta['post'], lgTE_RD4.beta['post'], lgTE_RD5.beta['post'],
                 lgTE_RD6.beta['post'], lgTE_DID7.beta['post'], lgTE_DID8.beta['post']]
        table.loc["logTotal Expenditure"] = lgte
        
         #LOG TE Standard Error
        lgte_se = [lgTE_RD1.se['post'],lgTE_RD2.se['post'],lgTE_RD3.se['post'], lgTE_RD4.se['post'], lgTE_RD5.se['post'],
                 lgTE_RD6.se['post'], lgTE_DID7.se['post'], lgTE_DID8.se['post']]
        table.loc["logTE_SE"]=lgte_se
 
        #Child Related Expenditure coefficient of post
        CRE_RD1 = RD_CRE_1(df_he)
        CRE_RD2 = RD_CRE_2(df_he)
        CRE_RD3 = RD_CRE_3(df_he)
        CRE_RD4 = RD_CRE_4(df_he)
        CRE_RD5 = RD_CRE_5(df_he)
        CRE_RD6 = RD_CRE_6(df_he)
        CRE_DID7 = DID_CRE_7(df_he)
        CRE_DID8 = DID_CRE_8(df_he)
        cre = [CRE_RD1.beta['post'],CRE_RD2.beta['post'],CRE_RD3.beta['post'], CRE_RD4.beta['post'], CRE_RD5.beta['post'],
                 CRE_RD6.beta['post'], CRE_DID7.beta['post'], CRE_DID8.beta['post']]
        table.loc["Child Related Expenditure"] = cre
        
        #CRE Standard Error
        cre_se = [CRE_RD1.se['post'],CRE_RD2.se['post'],CRE_RD3.se['post'], CRE_RD4.se['post'], CRE_RD5.se['post'],
                 CRE_RD6.se['post'], CRE_DID7.se['post'], CRE_DID8.se['post']]
        table.loc["CRE_SE"]= cre_se
        
        #LOG CRE coefficient
        lgCRE_RD1 = RD_logCRE_1(df_he)
        lgCRE_RD2 = RD_logCRE_2(df_he)
        lgCRE_RD3 = RD_logCRE_3(df_he)
        lgCRE_RD4 = RD_logCRE_4(df_he)
        lgCRE_RD5 = RD_logCRE_5(df_he)
        lgCRE_RD6 = RD_logCRE_6(df_he)
        lgCRE_DID7 = DID_logCRE_7(df_he)
        lgCRE_DID8 = DID_logCRE_8(df_he)
        lgcre = [lgCRE_RD1.beta['post'],lgCRE_RD2.beta['post'],lgCRE_RD3.beta['post'], lgCRE_RD4.beta['post'], lgCRE_RD5.beta['post'],
                 lgCRE_RD6.beta['post'], lgCRE_DID7.beta['post'], lgCRE_DID8.beta['post']]
        table.loc["logChild Related Expenditure"] = lgcre
        
         #LOG CRE Standard Error
        lgcre_se = [lgCRE_RD1.se['post'],lgCRE_RD2.se['post'],lgCRE_RD3.se['post'], lgCRE_RD4.se['post'], lgCRE_RD5.se['post'],
                 lgCRE_RD6.se['post'], lgCRE_DID7.se['post'], lgCRE_DID8.se['post']]
        table.loc["logCRE_SE"]=lgcre_se
 
        #Durable goods Expenditure coefficient of post   
        DGE_RD1 = RD_DGE_1(df_he)
        DGE_RD2 = RD_DGE_2(df_he)
        DGE_RD3 = RD_DGE_3(df_he)
        DGE_RD4 = RD_DGE_4(df_he)
        DGE_RD5 = RD_DGE_5(df_he)
        DGE_RD6 = RD_DGE_6(df_he)
        DGE_DID7 = DID_DGE_7(df_he)
        DGE_DID8 = DID_DGE_8(df_he)
        dge = [DGE_RD1.beta['post'],DGE_RD2.beta['post'],DGE_RD3.beta['post'], DGE_RD4.beta['post'], DGE_RD5.beta['post'],
                 DGE_RD6.beta['post'], DGE_DID7.beta['post'], DGE_DID8.beta['post']]
        table.loc["Durable Goods Expenditure"] = dge
        
        #DGE Standard Error
        dge_se = [DGE_RD1.se['post'],DGE_RD2.se['post'],DGE_RD3.se['post'], DGE_RD4.se['post'], DGE_RD5.se['post'],
                 DGE_RD6.se['post'], DGE_DID7.se['post'], DGE_DID8.se['post']]
        table.loc["DGE_SE"]= dge_se
        
        #LOG DGE coefficient
        lgDGE_RD1 = RD_logDGE_1(df_he)
        lgDGE_RD2 = RD_logDGE_2(df_he)
        lgDGE_RD3 = RD_logDGE_3(df_he)
        lgDGE_RD4 = RD_logDGE_4(df_he)
        lgDGE_RD5 = RD_logDGE_5(df_he)
        lgDGE_RD6 = RD_logDGE_6(df_he)
        lgDGE_DID7 = DID_logDGE_7(df_he)
        lgDGE_DID8 = DID_logDGE_8(df_he)
        lgdge = [lgDGE_RD1.beta['post'],lgDGE_RD2.beta['post'],lgDGE_RD3.beta['post'], lgDGE_RD4.beta['post'], lgDGE_RD5.beta['post'],
                 lgDGE_RD6.beta['post'], lgDGE_DID7.beta['post'], lgDGE_DID8.beta['post']]
        table.loc["logDurable Goods Expenditure"] = lgdge
        
         #LOG DGE Standard Error
        lgdge_se = [lgDGE_RD1.se['post'],lgDGE_RD2.se['post'],lgDGE_RD3.se['post'], lgDGE_RD4.se['post'], lgDGE_RD5.se['post'],
                 lgDGE_RD6.se['post'], lgDGE_DID7.se['post'], lgDGE_DID8.se['post']]
        table.loc["logDGE_SE"]=lgdge_se
    
        #Observations
        table=table.astype(float).round(4)
        obs =[TE_RD1.N, TE_RD2.N, TE_RD3.N, TE_RD4.N, TE_RD5.N, TE_RD6.N, TE_DID7.N, TE_DID8.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48,48]
        table.loc["Number of months"] = months

        
        return(table)

table4(df_he)

        


# In[114]:


#=================================Create Table 6
def table6(df_he):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID_1(7)': [], 'DID_2(8)': []})
        result = ('Private Daycare', 'PR_SE', 'Private Daycare(binary)', 'BPR_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Child Care Expenditure'] = result
        table = table.set_index('Child Care Expenditure')
        
        #Private daycare coefficient of post
        PR_RD1 = RD_PR_1(df_he)
        PR_RD2 = RD_PR_2(df_he)
        PR_RD3 = RD_PR_3(df_he)
        PR_RD4 = RD_PR_4(df_he)
        PR_RD5 = RD_PR_5(df_he)
        PR_RD6 = RD_PR_6(df_he)
        PR_DID7 = DID_PR_7(df_he)
        PR_DID8 = DID_PR_8(df_he)
        pr = [PR_RD1.beta['post'],PR_RD2.beta['post'],PR_RD3.beta['post'], PR_RD4.beta['post'], PR_RD5.beta['post'],
                 PR_RD6.beta['post'], PR_DID7.beta['post'], PR_DID8.beta['post']]
        table.loc["Private Daycare"] = pr
        
        #PR Standard Error
        pr_se = [PR_RD1.se['post'], PR_RD2.se['post'],PR_RD3.se['post'], PR_RD4.se['post'], PR_RD5.se['post'],
                 PR_RD6.se['post'], PR_DID7.se['post'], PR_DID8.se['post']]
        table.loc["PR_SE"]=pr_se
        
        #[BINARY] Private daycare coefficient of post
        BPR_RD1 = RD_binPR_1(df_he)
        BPR_RD2 = RD_binPR_2(df_he)
        BPR_RD3 = RD_binPR_3(df_he)
        BPR_RD4 = RD_binPR_4(df_he)
        BPR_RD5 = RD_binPR_5(df_he)
        BPR_RD6 = RD_binPR_6(df_he)
        BPR_DID7 = DID_binPR_7(df_he)
        BPR_DID8 = DID_binPR_8(df_he)
        bpr = [BPR_RD1.beta['post'],BPR_RD2.beta['post'],BPR_RD3.beta['post'], BPR_RD4.beta['post'], BPR_RD5.beta['post'],
                 BPR_RD6.beta['post'], BPR_DID7.beta['post'], BPR_DID8.beta['post']]
        table.loc["Private Daycare(binary)"] = bpr
        
        #BPR Standard Error
        bpr_se = [BPR_RD1.se['post'], BPR_RD2.se['post'],BPR_RD3.se['post'], BPR_RD4.se['post'], BPR_RD5.se['post'],
                 BPR_RD6.se['post'], BPR_DID7.se['post'], BPR_DID8.se['post']]
        table.loc["BPR_SE"]=bpr_se
        
        #Observations
        table=table.astype(float).round(4)
        obs =[PR_RD1.N, PR_RD2.N, PR_RD3.N, PR_RD4.N, PR_RD5.N, PR_RD6.N, PR_DID7.N, PR_DID8.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48,48]
        table.loc["Number of months"] = months

        
        return(table)

table6(df_he)


# In[115]:


#*****************5. Female Labor Supply*********************************
df_ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[116]:


#1.Control variables
#gen m2 = m*m
df_ls['m2']=df_ls['m']*df_ls['m']

#No father present.*/
#gen nodad=0
#replace nodad=1 if dadid==0
df_ls['nodad']=0
df_ls.loc[df_ls['dadid']==0, 'nodad']=1


# Mother not married 
#gen smom=0
#replace smom=1 if eciv!=2
df_ls['smom']=0
df_ls.loc[df_ls['eciv']!=2, 'smom']=1

#Mother single.*/
#gen single=0
#replace single=1 if eciv==1
df_ls['single']=0
df_ls.loc[df_ls['eciv']==1, 'single']=1

#Mother separated or divorced.*/
#gen sepdiv=0
#replace sepdiv=1 if eciv==4
df_ls['sepdiv']=0
df_ls.loc[df_ls['eciv']==4, 'sepdiv']=1

# No partner in the household.*/
#gen nopart=0
#replace nopart=1 if partner==0
df_ls['nopart']=0
df_ls.loc[df_ls['partner']==0, 'nopart']=1


# In[117]:


#***************Probability of the mother being in the maternity leave period at the time of the interview
#gen pleave=0
#replace pleave=0.17 if (q==1 & m==2) | (q==2 & m==5) | (q==3 & m==8) | (q==4 & m==11)
#replace pleave=0.5 if (q==1 & m==3) | (q==2 & m==6) | (q==3 & m==9) | (q==4 & m==12)
#replace pleave=0.83 if (q==1 & m==4) | (q==2 & m==7) | (q==3 & m==10) | (q==4 & m==13)
#replace pleave=1 if (q==1 & m>4 & m<9) | (q==2 & m>7 & m<12) | (q==3 & m>10 & m<15) | (q==4 & m>13)

df_ls['pleave']=0

df_ls.loc[(df_ls['q']==1) & (df_ls['m']==2)|(df_ls['q']==2) & (df_ls['m']==5)|(df_ls['q']==3) & (df_ls['m']==8)|(df_ls['q']==4) & (df_ls['m']==11) ,'pleave']=0.17
df_ls.loc[((df_ls['q']==1) & (df_ls['m']==3)) | ((df_ls['q']==2) & (df_ls['m']==6))  | ((df_ls['q']==3) & (df_ls['m']==9)) |((df_ls['q']==4) & (df_ls['m']==12)), 'pleave'] = 0.5
df_ls.loc[((df_ls['q']==1) & (df_ls['m']==4)) | ((df_ls['q']==2) & (df_ls['m']==7)) | ((df_ls['q']==3) & (df_ls['m']==10))  | ((df_ls['q']==4) & (df_ls['m']==13)), 'pleave'] = 0.83
df_ls.loc[((df_ls['q']==1) & (df_ls['m']>4) & (df_ls['m']<9)) | ((df_ls['q']==2) & (df_ls['m']>7) & (df_ls['m']<12)) | ((df_ls['q']==3) & (df_ls['m']>10) & (df_ls['m']<15))| ((df_ls['q']==4) & (df_ls['m']>13)), 'pleave'] = 1


# In[118]:


#**********************************Table 5 [Regressions]**************************************************
##**********************************5-1Working last week

#Create interaction dummies
df_ls['ipost_1']=df_ls['post']*df_ls['m']
df_ls['ipost_2'] =df_ls['post']*df_ls['m2']

#create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2 and so on
for j in range(1,5):
    df_ls['iq_'+str(j)] = 0
    for i in range(len(df_ls)):
        if df_ls.loc[i,'q'] == j:
            df_ls.loc[i, 'iq_'+str(j)] = 1

#RDD9m(1)
#xi: reg work post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def RD_WL_1(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -10) & (df_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[119]:


#RDD6m(2)xi: reg work post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-7 & m<6, robust 
def RD_WL_2(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -7) & (df_ls['m']<6)], 'work', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[120]:


#RDD4m(3)xi: reg work post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-5 & m<4, robust 
def RD_WL_3(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -5) & (df_ls['m']<4)], 'work', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[121]:


#RDD3m(4)xi: reg work post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-4 & m<3, robust
def RD_WL_4(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -4) & (df_ls['m']<3)], 'work', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[122]:


#RDD3m(5)xi: reg work post if m>-3 & m<2, robust 
def RD_WL_5(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -3) & (df_ls['m']<2)], 'work', ['post'],addcons=True)
    return(result)


# In[123]:


#RDD2m(6)xi: reg work post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-3 & m<2, robust
##note: pleave omitted because of collinearity
def RD_WL_6(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -3) & (df_ls['m']<2)], 'work', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[124]:


#DID(7)xi: reg work post m m2 age age2 age3 immig primary hsgrad univ sib pleave i.n_month i.q, robust ""cluster(m)""

#create i_n_month: in_month_1 for January, in_month_2 for February and so on
for j in range(1,13):
    df_ls['i_n_month_'+str(j)] = 0
    for i in range(len(df_ls)):
        if df_ls.loc[i,'n_month'] == j:
            df_ls.loc[i, 'i_n_month_'+str(j)] = 1
def DID_WL_7(df_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_ls_DID = df_ls[['work','work2','post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_ls_DID, 'work', ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[125]:


#[수정필요]DID(8)xi: reg work post m age age2 age3 immig primary hsgrad univ sib pleave i.n_month i.q, robust cluster(m)
def DID_WL_8(df_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_ls_DID = df_ls[['work','work2','post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]


    result=mt.reg(df_ls_DID, 'work', ['post','m','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[126]:


##*************************************5-2 Employed******************************************************/
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
def RD_EP_1(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -10) & (df_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[127]:


#RDD6m(2): xi: reg work2 post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-7 & m<6, robust 
def RD_EP_2(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -7) & (df_ls['m']<6)], 'work2', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[128]:


#RDD4m(3)xi: reg work2 post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-5 & m<4, robust 
def RD_EP_3(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -5) & (df_ls['m']<4)], 'work2', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[129]:


#RDD3m(4)xi: reg work2 post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-4 & m<3, robust
def RD_EP_4(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -4) & (df_ls['m']<3)], 'work2', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)


# In[130]:


#RDD2m(5)xi: reg work2 post if m>-3 & m<2, robust 
def RD_EP_5(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -3) & (df_ls['m']<2)], 'work2', ['post'],addcons=True)
    return(result)


# In[131]:


#RDD2m(6)xi: reg work2 post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-3 & m<2, robust
def RD_EP_6(df_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    df_ls_index = df_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(df_ls_index[(df_ls['m']> -3) & (df_ls['m']<2)], 'work2', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','iq_2', 'iq_3', 'iq_4'],addcons=True)
    return(result)
##note: pleave omitted because of collinearity


# In[132]:


#DID(7):xi: reg work2 post m m2 age age2 age3 immig primary hsgrad univ sib pleave i.n_month i.q, robust cluster(m)
def DID_EP_7(df_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_ls_DID = df_ls[['work','work2','post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_ls_DID, 'work2', ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[133]:


#[수정필요]
#DID(8): xi: reg work2 post m age age2 age3 immig primary hsgrad univ sib pleave i.n_month i.q, robust cluster(m)
def DID_EP_8(df_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    df_ls_DID = df_ls[['work','work2','post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(df_ls_DID, 'work2', ['post','m','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[134]:


##================Create Table 5
def table5(df_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID_1(7)': [], 'DID_2(8)': []})
        result = ('Working Last Week', 'WL_SE', 'Employed', 'EP_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Labor Supply'] = result
        table = table.set_index('Labor Supply')
        
        #Working Last Week coefficient of post
        WL_RD1 = RD_WL_1(df_ls)
        WL_RD2 = RD_WL_2(df_ls)
        WL_RD3 = RD_WL_3(df_ls)
        WL_RD4 = RD_WL_4(df_ls)
        WL_RD5 = RD_WL_5(df_ls)
        WL_RD6 = RD_WL_6(df_ls) 
        WL_DID7 = DID_WL_7(df_ls)
        WL_DID8 = DID_WL_8(df_ls)
        wl = [WL_RD1.beta['post'], WL_RD2.beta['post'], WL_RD3.beta['post'], WL_RD4.beta['post'], WL_RD5.beta['post'],
                 WL_RD6.beta['post'], WL_DID7.beta['post'], WL_DID8.beta['post']]
        table.loc["Working Last Week"] = wl
        
        #WL Standard Error
        wl_se = [WL_RD1.se['post'],WL_RD2.se['post'], WL_RD3.se['post'], WL_RD4.se['post'], WL_RD5.se['post'],
                 WL_RD6.se['post'], WL_DID7.se['post'], WL_DID8.se['post']]
        table.loc["WL_SE"]=wl_se
        
        #Employed coefficient of post
        EP_RD1 = RD_EP_1(df_ls)
        EP_RD2 = RD_EP_2(df_ls)
        EP_RD3 = RD_EP_3(df_ls)
        EP_RD4 = RD_EP_4(df_ls)
        EP_RD5 = RD_EP_5(df_ls)
        EP_RD6 = RD_EP_6(df_ls) 
        EP_DID7 = DID_EP_7(df_ls)
        EP_DID8 = DID_EP_8(df_ls)
        ep = [EP_RD1.beta['post'], EP_RD2.beta['post'], EP_RD3.beta['post'], EP_RD4.beta['post'], EP_RD5.beta['post'],
                 EP_RD6.beta['post'], EP_DID7.beta['post'], EP_DID8.beta['post']]
        table.loc["Employed"] = ep
        
        #EP Standard Error
        ep_se = [EP_RD1.se['post'], EP_RD2.se['post'], EP_RD3.se['post'], EP_RD4.se['post'], EP_RD5.se['post'],
                 EP_RD6.se['post'], EP_DID7.se['post'], EP_DID8.se['post']]
        table.loc["EP_SE"] = ep_se
          
        #Observations
        table=table.astype(float).round(4)
        obs =[EP_RD1.N, EP_RD2.N, EP_RD3.N, EP_RD4.N, EP_RD5.N, EP_RD6.N, EP_DID7.N, EP_DID8.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y","N"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48,48]
        table.loc["Number of months"] = months

        
        return(table)

table5(df_ls)


# In[135]:


##############========================Create Table 1: Descriptive Summary===============================
def table1(df, df_ab, df_he, df_ls):
    table = pd.DataFrame({'Mean': [], 'SD': [], 'Median': []})
    variables = ('Monthly number of conceptions', 'Post-June 2007 dummy_con', 'Month of conception',
                 'Monthly number of abortions', 'Post-June 2007 dummy_abo', 'Month of abortions',
                 'Total Expenditure', 'Child Related Expenditure', 'Durable Goods Expenditure', 'Daycare expenditure',
                 'Post-June 2007 dummy_exp', 'Month of birth_exp', 'Age of mother', 'Mother some secondary', 'Mother high school graduate', 
                 'Mother college graduate','Mother immigrant', 'Not first born',
                 'Worked last week', 'Currently employed', 'Post-June 2007 dummy_labor', 'Month of birth_labor', 'Age of mother_labor',
                 'Mother some secondary_labor', 'Mother high school graduate_labor', 'Mother college graduate_labor', 'Mother immigrant_labor', 'Not first born_labor')

                
    table['Summary'] = variables
    table = table.set_index('Summary')
    
    ###======Panel A. Vital Statistics(2000-2009)
    #Monthly # of conceptions
    c_n = [df[(df['mc']>-91) & (df['mc']<30)]['n'].mean(), df[(df['mc']>-91) & (df['mc']<30)]['n'].std(), df[(df['mc']>-91) & (df['mc']<30)]['n'].median()]
    table.loc['Monthly number of conceptions'] = c_n
    
    #Post-June 2007 dummy_con
    c_post = [df[(df['mc']>-91) & (df['mc']<30)]['post'].mean(), df[(df['mc']>-91) & (df['mc']<30)]['post'].std(), df[(df['mc']>-91) & (df['mc']<30)]['post'].median()]
    table.loc['Post-June 2007 dummy_con'] = c_post
    
    #Month of conception
    c_mc = [df[(df['mc']>-91) & (df['mc']<30)]['mc'].mean(), df[(df['mc']>-91) & (df['mc']<30)]['mc'].std(), df[(df['mc']>-91) & (df['mc']<30)]['mc'].median()]
    table.loc['Month of conception'] = c_mc
    
    
    ###=======Panel B. Abortions Statistics(2000-2009)
    #Monthly # of abortions
    a_n = [df_ab['n_tot'].mean(), df_ab['n_tot'].std(), df_ab['n_tot'].median()]
    table.loc['Monthly number of abortions'] = a_n
    
    #Post-June 2007 dummy_abo
    a_post = [df_ab['post'].mean(), df_ab['post'].std(), df_ab['post'].median()]
    table.loc['Post-June 2007 dummy_abo'] = a_post
    
    #Month of abortion
    a_m = [df_ab['m'].mean(), df_ab['m'].std(), df_ab['m'].median()]
    table.loc['Month of abortions'] = a_m
    
    ###=======Panel C. Household Budget Survey (2008)
    #Total Expenditure
    s_te = [df_he[(df_he['month']>-10) & (df_he['month']<9)]['gastmon'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['gastmon'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['gastmon'].median()]
    table.loc['Total Expenditure'] = s_te
    
    #Child Related Expenditure
    s_cre = [df_he[(df_he['month']>-10) & (df_he['month']<9)]['c_m_exp'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['c_m_exp'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['c_m_exp'].median()]
    table.loc['Child Related Expenditure'] = s_cre
                 
    #Durable Goods Expenditure
    s_dge = [df_he[(df_he['month']>-10) & (df_he['month']<9)]['dur_exp'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['dur_exp'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['dur_exp'].median()]
    table.loc['Durable Goods Expenditure'] = s_dge
    
    #Daycare expenditure
    s_dce = [df_he[(df_he['month']>-10) & (df_he['month']<9)]['m_exp12312'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['m_exp12312'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['m_exp12312'].median()]
    table.loc['Daycare expenditure'] = s_dce
    
    #'Post-June 2007 dummy_exp'
    s_post = [df_he[(df_he['month']>-10) & (df_he['month']<9)]['post'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['post'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['post'].median()]
    table.loc['Post-June 2007 dummy_exp'] = s_post
    
    #'Month of birth_exp'
    s_birth= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['month'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['month'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['month'].median()]
    table.loc['Month of birth_exp'] = s_birth
    
    #'Age of mother'
    s_age= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['agemom'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['agemom'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['agemom'].median()]
    table.loc['Age of mother'] = s_age
    
    #'Mother some secondary'
    s_sec1mom= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['sec1mom'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['sec1mom'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['sec1mom'].median()]
    table.loc['Mother some secondary'] = s_sec1mom

    #Mother high school graduate'
    s_sec2mom= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['sec2mom'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['sec2mom'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['sec2mom'].median()]
    table.loc['Mother high school graduate'] = s_sec2mom
   
    #'Mother college graduate'
    s_unimom= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['unimom'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['unimom'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['unimom'].median()]
    table.loc['Mother college graduate'] = s_unimom
     
    #'Mother immigrant'
    s_immig= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['immig'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['immig'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['immig'].median()]
    table.loc['Mother immigrant'] = s_immig
    
    #'Not first born'
    s_sib= [df_he[(df_he['month']>-10) & (df_he['month']<9)]['sib'].mean(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['sib'].std(), df_he[(df_he['month']>-10) & (df_he['month']<9)]['sib'].median()]
    table.loc['Not first born'] = s_sib
     
    ###=======Panel D.Labor force survey (2008) #sum work work2 post m age primary hsgrad univ immig sib if m>-10 & m<9
    #'Worked last week' 
    ls_work= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['work'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['work'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['work'].median()]
    table.loc['Worked last week'] = ls_work
     
    #'Currently employed' 
    ls_work2= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['work2'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['work2'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['work2'].median()]
    table.loc['Currently employed'] = ls_work2
    
    #'Post-June 2007 dummy_labor' 
    ls_post= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['post'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['post'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['post'].median()]
    table.loc['Post-June 2007 dummy_labor'] = ls_post
    
    #'Month of birth_labor' 
    ls_m= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['m'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['m'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['m'].median()]
    table.loc['Month of birth_labor'] = ls_m
    
    #'Age of mother_labor'
    ls_age= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['age'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['age'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['age'].median()]
    table.loc['Age of mother_labor'] = ls_age
        
    #'Mother some secondary_labor' 
    ls_primary= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['primary'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['primary'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['primary'].median()]
    table.loc['Mother some secondary_labor'] = ls_primary
     
    #'Mother high school graduate_labor'
    ls_hsgrad= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['hsgrad'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['hsgrad'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['hsgrad'].median()]
    table.loc['Mother high school graduate_labor'] = ls_hsgrad
    
    
    #'Mother college graduate_labor'
    ls_univ= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['univ'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['univ'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['univ'].median()]
    table.loc['Mother college graduate_labor'] = ls_univ
        
    #'Mother immigrant_labor'
    ls_immig= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['immig'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['immig'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['immig'].median()]
    table.loc['Mother immigrant_labor'] = ls_immig
  
    
    #'Not first born_labor'   
    ls_sib= [df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['sib'].mean(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['sib'].std(), df_ls[(df_ls['m']>-10) & (df_ls['m']<9)]['sib'].median()]
    table.loc['Not first born_labor'] = ls_sib
        
    
    table = table.astype(float).round(3)
    return(table)

table1(df, df_ab,df_he,df_ls)


# In[ ]:




