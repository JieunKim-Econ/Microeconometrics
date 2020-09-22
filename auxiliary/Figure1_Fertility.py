#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns

import econtools
import econtools.metrics as mt


# In[2]:


##====================[Conceptions]========================
#Variables
#mesp: Month of birth
#year: Year of birth
#prem: Prematurity indicator
#semanas: Weeks of gestation at birth

df_con=pd.read_stata("data/data_births_20110196.dta")
df_con


# In[3]:


#1. Create month of birth variable: (0 = July 2007, 1 = August 2007, etc)

df_con.loc[(df_con['year']==2010), 'm'] = df_con['mesp']+29
df_con.loc[(df_con['year']==2009), 'm'] = df_con['mesp'] + 17 #replace m = mesp + 17 if year==2009
df_con.loc[(df_con['year']==2008), 'm'] = df_con['mesp'] + 5  #replace m = mesp + 5 if year==2008
df_con.loc[(df_con['year']==2007), 'm'] = df_con['mesp'] - 7  #replace m = mesp - 7 if year==2007
df_con.loc[(df_con['year']==2006), 'm'] = df_con['mesp'] - 19 #replace m = mesp - 19 if year==2006
df_con.loc[(df_con['year']==2005), 'm'] = df_con['mesp'] - 31 #replace m = mesp - 31 if year==2005
df_con.loc[(df_con['year']==2004), 'm'] = df_con['mesp'] - 43 #replace m = mesp - 43 if year==2004
df_con.loc[(df_con['year']==2003), 'm'] = df_con['mesp'] - 55 #replace m = mesp - 55 if year==2003
df_con.loc[(df_con['year']==2002), 'm'] = df_con['mesp'] - 67
df_con.loc[(df_con['year']==2001), 'm'] = df_con['mesp'] - 79 #replace m = mesp - 79 if year==2001
df_con.loc[(df_con['year']==2000), 'm'] = df_con['mesp'] - 91 #replace m = mesp - 91 if year==2000

#df_con.head(10)


# In[4]:


#df['m'].describe()


# In[5]:


#2. Create month of conception variable.
## 2.1. Naive definition (9 months before the birth month) 
#gen mc1 = m - 9

df_con['mc1'] = df_con['m'] - 9

# df.head(40)


# In[6]:


##2.2. Naivee plus prematures. (9 months before the birth month, 8 if premature)
#gen mc2 = m - 9
#replace mc2 = m - 8 if (semanas>0 & semanas<38) | prem==2

df_con['mc2'] = df_con['m'] - 9
df_con.loc[((df_con['semanas'] > 0) & (df_con['semanas'] < 38)) | (df_con['prem']==2), 'mc2'] =df_con['m'] - 8 

# df.head(40)


# In[7]:


#df['mc2'].describe()


# In[8]:


##2.3. Sophisticated (calculated based on weeks of gestation)
##(**This is the one used in the paper**)##
#gen mc3 = m - 9
#replace mc3 = m - 10 if semanas > 43 & semanas!=.
#replace mc3 = m - 8 if (semanas < 39 & semanas!=0) | prem==2

df_con['mc3'] = df_con['m'] - 9
df_con.loc[((df_con['semanas'] > 43) & (df_con['semanas']!=np.nan)), 'mc3'] = df_con['m'] - 10 
df_con.loc[((df_con['semanas'] < 39) & (df_con['semanas']!=0)) | (df_con['prem']==2), 'mc3'] = df_con['m'] - 8

#df.head(40)


# In[9]:


#4-1. Collapse by month of conception. 
#gen n=1
#collapse (count) n, by(mc3)
#rename mc3 mc
#sum

#df.groupby('mc3')['n'].count()

df_con=df_con.groupby('mc3', as_index=False)['mc3'].agg({'n':'count'})
#df_con


# In[10]:


#4-2. Collapse by month of conception.
#rename mc3 mc
#sum
df_con= df_con.rename(columns={'mc3': 'mc'})
#df_con


# In[11]:


# 5. Calendar month of conception. 

df_con['month']=1 
df_con.loc[(df_con['mc']==-30) | (df_con['mc']==-18) | (df_con['mc']==-6) | (df_con['mc']==6) | (df_con['mc']==18) | (df_con['mc']==30) | (df_con['mc']==-42) | (df_con['mc']==-54) | (df_con['mc']==-66) | (df_con['mc']==-87)| (df_con['mc']==-99), 'month'] = 1
df_con.loc[(df_con['mc']==-29) | (df_con['mc']==-17) | (df_con['mc']==-5) | (df_con['mc']==7) | (df_con['mc']==19) | (df_con['mc']==31) | (df_con['mc']==-41) | (df_con['mc']==-53) | (df_con['mc']==-65) | (df_con['mc']==-86)| (df_con['mc']==-98), 'month'] = 2
df_con.loc[(df_con['mc']==-28) | (df_con['mc']==-16) | (df_con['mc']==-4) | (df_con['mc']==8) | (df_con['mc']==20) | (df_con['mc']==32) | (df_con['mc']==-40) | (df_con['mc']==-52) | (df_con['mc']==-64) | (df_con['mc']==-85)| (df_con['mc']==-97), 'month'] = 3
df_con.loc[(df_con['mc']==-27) | (df_con['mc']==-15) | (df_con['mc']==-3) | (df_con['mc']==9) | (df_con['mc']==21) | (df_con['mc']==33) | (df_con['mc']==-39) | (df_con['mc']==-51) | (df_con['mc']==-63) | (df_con['mc']==-84)| (df_con['mc']==-96), 'month'] = 4
df_con.loc[(df_con['mc']==-26) | (df_con['mc']==-14) | (df_con['mc']==-2) | (df_con['mc']==10) | (df_con['mc']==22) | (df_con['mc']==-38) | (df_con['mc']==-50) | (df_con['mc']==-62) | (df_con['mc']==-74)| (df_con['mc']==-83)| (df_con['mc']==-95), 'month'] = 5
df_con.loc[(df_con['mc']==-25) | (df_con['mc']==-13) | (df_con['mc']==-1) | (df_con['mc']==11) | (df_con['mc']==23) | (df_con['mc']==-37) | (df_con['mc']==-49) | (df_con['mc']==-61) | (df_con['mc']==-73)| (df_con['mc']==-82)| (df_con['mc']==-94), 'month'] = 6
df_con.loc[(df_con['mc']==-24) | (df_con['mc']==-12) | (df_con['mc']==0) | (df_con['mc']==12) | (df_con['mc']==24) | (df_con['mc']==-36) | (df_con['mc']==-48) | (df_con['mc']==-60) | (df_con['mc']==-72)| (df_con['mc']==-81)| (df_con['mc']==-93), 'month'] = 7
df_con.loc[(df_con['mc']==-23) | (df_con['mc']==-11) | (df_con['mc']==1) | (df_con['mc']==13) | (df_con['mc']==25) | (df_con['mc']==-35) | (df_con['mc']==-47) | (df_con['mc']==-59) | (df_con['mc']==-71)| (df_con['mc']==-80)| (df_con['mc']==-92), 'month'] = 8
df_con.loc[(df_con['mc']==-22) | (df_con['mc']==-10) | (df_con['mc']==2) | (df_con['mc']==14) | (df_con['mc']==26) | (df_con['mc']==-34) | (df_con['mc']==-46) | (df_con['mc']==-58) | (df_con['mc']==-70)| (df_con['mc']==-79)| (df_con['mc']==-91), 'month'] = 9
df_con.loc[(df_con['mc']==-21) | (df_con['mc']==-9) | (df_con['mc']==3) | (df_con['mc']==15) | (df_con['mc']==27) | (df_con['mc']==-33) | (df_con['mc']==-45) | (df_con['mc']==-57) | (df_con['mc']==-69)| (df_con['mc']==-78)| (df_con['mc']==-90), 'month'] = 10
df_con.loc[(df_con['mc']==-20) | (df_con['mc']==-8) | (df_con['mc']==4) | (df_con['mc']==16) | (df_con['mc']==28) | (df_con['mc']==-32) | (df_con['mc']==-44) | (df_con['mc']==-56) | (df_con['mc']==-68)| (df_con['mc']==-77)| (df_con['mc']==-89), 'month'] = 11
df_con.loc[(df_con['mc']==-19) | (df_con['mc']==-7) | (df_con['mc']==5) | (df_con['mc']==17) | (df_con['mc']==29) | (df_con['mc']==-31) | (df_con['mc']==-43) | (df_con['mc']==-55) | (df_con['mc']==-67)| (df_con['mc']==-76)| (df_con['mc']==-88), 'month'] = 12
#df_con


# In[12]:


#6. July indicator 
#gen july=n if month==7

df_con.loc[df_con['month']==7, 'july'] = df_con['n']
#df.head(20)


# In[13]:


#7. Number of days in a month
#gen days=31
#replace days=28 if month==2
#replace days=29 if mc==7
#replace days=30 if month==4 | month==6 | month==9 | month==11

df_con['days'] = 31
df_con.loc[(df_con['month']==2), 'days'] = 28
df_con.loc[(df_con['mc']==7), 'days'] = 29
df_con.loc[(df_con['month']==4)| (df_con['month']==6)|(df_con['month']==9)|(df_con['month']==11),'days'] = 30

#df


# In[14]:


# 8. A post indicator for post-policy conception
#gen post=0
#replace post=1 if mc>=0
#gen mc2=mc*mc
#gen mc3=mc*mc*mc

df_con['post']=0
df_con.loc[(df_con['mc']>=0), 'post']=1

df_con['mc2']=df_con['mc']*df_con['mc']
df_con['mc3']=df_con['mc']*df_con['mc']*df_con['mc']

#df


# In[15]:


#df.tail(65)


# In[16]:


#Create Bimonthly number of conceptions

#df['bim_n']=np.nan
df_con.loc[(df_con['mc'] ==-29), 'bim_n'] = df_con.loc[df_con['mc']==-29,'n'].sum() + df_con.loc[df_con['mc']==-30,'n'].sum()
df_con.loc[(df_con['mc'] ==-27), 'bim_n'] = df_con.loc[df_con['mc']==-27,'n'].sum() + df_con.loc[df_con['mc']==-28,'n'].sum()
df_con.loc[(df_con['mc'] ==-25), 'bim_n'] = df_con.loc[df_con['mc']==-25,'n'].sum() + df_con.loc[df_con['mc']==-26,'n'].sum()
df_con.loc[(df_con['mc'] ==-23), 'bim_n'] = df_con.loc[df_con['mc']==-23,'n'].sum() + df_con.loc[df_con['mc']==-24,'n'].sum()
df_con.loc[(df_con['mc'] ==-21), 'bim_n'] = df_con.loc[df_con['mc']==-21,'n'].sum() + df_con.loc[df_con['mc']==-22,'n'].sum()
df_con.loc[(df_con['mc'] ==-19), 'bim_n'] = df_con.loc[df_con['mc']==-19,'n'].sum() + df_con.loc[df_con['mc']==-20,'n'].sum()
df_con.loc[(df_con['mc'] ==-17), 'bim_n'] = df_con.loc[df_con['mc']==-17,'n'].sum() + df_con.loc[df_con['mc']==-18,'n'].sum()
df_con.loc[(df_con['mc'] ==-15), 'bim_n'] = df_con.loc[df_con['mc']==-15,'n'].sum() + df_con.loc[df_con['mc']==-16,'n'].sum()
df_con.loc[(df_con['mc'] ==-13), 'bim_n'] = df_con.loc[df_con['mc']==-13,'n'].sum() + df_con.loc[df_con['mc']==-14,'n'].sum()
df_con.loc[(df_con['mc'] ==-11), 'bim_n'] = df_con.loc[df_con['mc']==-11,'n'].sum() + df_con.loc[df_con['mc']==-12,'n'].sum()
df_con.loc[(df_con['mc'] ==-9), 'bim_n'] = df_con.loc[df_con['mc']==-9,'n'].sum() + df_con.loc[df_con['mc']==-10,'n'].sum()
df_con.loc[(df_con['mc'] ==-7), 'bim_n'] = df_con.loc[df_con['mc']==-7,'n'].sum() + df_con.loc[df_con['mc']==-8,'n'].sum()
df_con.loc[(df_con['mc'] ==-5), 'bim_n'] = df_con.loc[df_con['mc']==-5,'n'].sum() + df_con.loc[df_con['mc']==-6,'n'].sum()
df_con.loc[(df_con['mc'] ==-3), 'bim_n'] = df_con.loc[df_con['mc']==-3,'n'].sum() + df_con.loc[df_con['mc']==-4,'n'].sum()
df_con.loc[(df_con['mc'] ==-1), 'bim_n'] = df_con.loc[df_con['mc']==-1,'n'].sum() + df_con.loc[df_con['mc']==-2,'n'].sum()

df_con.loc[(df_con['mc'] ==1), 'bim_n'] = df_con.loc[df_con['mc']==1,'n'].sum() + df_con.loc[df_con['mc']==0,'n'].sum()
df_con.loc[(df_con['mc'] ==3), 'bim_n'] = df_con.loc[df_con['mc']==3,'n'].sum() + df_con.loc[df_con['mc']==2,'n'].sum()
df_con.loc[(df_con['mc'] ==5), 'bim_n'] = df_con.loc[df_con['mc']==5,'n'].sum() + df_con.loc[df_con['mc']==4,'n'].sum()
df_con.loc[(df_con['mc'] ==7), 'bim_n'] = df_con.loc[df_con['mc']==7,'n'].sum() + df_con.loc[df_con['mc']==6,'n'].sum()
df_con.loc[(df_con['mc'] ==9), 'bim_n'] = df_con.loc[df_con['mc']==9,'n'].sum() + df_con.loc[df_con['mc']==8,'n'].sum()
df_con.loc[(df_con['mc'] ==11), 'bim_n'] = df_con.loc[df_con['mc']==11,'n'].sum() + df_con.loc[df_con['mc']==10,'n'].sum()
df_con.loc[(df_con['mc'] ==13), 'bim_n'] = df_con.loc[df_con['mc']==13,'n'].sum() + df_con.loc[df_con['mc']==12,'n'].sum()
df_con.loc[(df_con['mc'] ==15), 'bim_n'] = df_con.loc[df_con['mc']==15,'n'].sum() + df_con.loc[df_con['mc']==14,'n'].sum()
df_con.loc[(df_con['mc'] ==17), 'bim_n'] = df_con.loc[df_con['mc']==17,'n'].sum() + df_con.loc[df_con['mc']==16,'n'].sum()
df_con.loc[(df_con['mc'] ==19), 'bim_n'] = df_con.loc[df_con['mc']==19,'n'].sum() + df_con.loc[df_con['mc']==18,'n'].sum()


#df['bi_n'].tail(60)


# In[17]:


#df[(df['mc']>-30) & (df['mc']<20)]['bim_n'].describe()


# In[18]:


# df[(df['mc']>-30) & (df['mc']<20)]['bim_n']


# In[19]:


#df[(df['mc']==1)]['bim_n']


# In[20]:


#Regressions Table 
#Create interaction dummies
df_con['ipost_1']=df_con['post']*df_con['mc']
df_con['ipost_2'] =df_con['post']*df_con['mc2']
df_con['ipost_3'] =df_con['post']*df_con['mc3']

#RRD: Linear Regression (1)
#xi: reg ln post i.post|mc i.post|mc2 i.post|mc3 days if mc>-91 & mc<30, robust

#def figure1_conceptions(df):
Y= 'bim_n'
X = ['mc','mc2','mc3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
con_index = df_con[['bim_n','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
results=mt.reg(con_index[(df_con['mc']> -30) & (df_con['mc']< 20)], 'bim_n', ['mc','mc2','mc3','post', 'ipost_1', 'ipost_2', 'ipost_3', 'days'], addcons=True)
predict_bim_n = results.yhat.round(0)


# In[21]:


predict_bim_n
predict_int=predict_bim_n.astype(int)


# In[22]:


predict_df = pd.DataFrame(predict_int)
predict_df.rename(columns={0:'predict_bim_n'}, inplace=True)


# In[23]:


#df['predict_bim_n'] = np.nan
df_con.loc[df_con['mc']==-29, 'predict_bim_n']= predict_df.iloc[0,0]
df_con.loc[df_con['mc']==-27, 'predict_bim_n']= predict_df.iloc[1,0]
df_con.loc[df_con['mc']==-25, 'predict_bim_n']= predict_df.iloc[2,0]
df_con.loc[df_con['mc']==-23, 'predict_bim_n']= predict_df.iloc[3,0]
df_con.loc[df_con['mc']==-21, 'predict_bim_n']= predict_df.iloc[4,0]
df_con.loc[df_con['mc']==-19, 'predict_bim_n']= predict_df.iloc[5,0]
df_con.loc[df_con['mc']==-17, 'predict_bim_n']= predict_df.iloc[6,0]
df_con.loc[df_con['mc']==-15, 'predict_bim_n']= predict_df.iloc[7,0]
df_con.loc[df_con['mc']==-13, 'predict_bim_n']= predict_df.iloc[8,0]
df_con.loc[df_con['mc']==-11, 'predict_bim_n']= predict_df.iloc[9,0]
df_con.loc[df_con['mc']==-9, 'predict_bim_n']= predict_df.iloc[10,0]
df_con.loc[df_con['mc']==-7, 'predict_bim_n']= predict_df.iloc[11,0]
df_con.loc[df_con['mc']==-5, 'predict_bim_n']= predict_df.iloc[12,0]
df_con.loc[df_con['mc']==-3, 'predict_bim_n']= predict_df.iloc[13,0]
df_con.loc[df_con['mc']==-1, 'predict_bim_n']= predict_df.iloc[14,0]
df_con.loc[df_con['mc']==1, 'predict_bim_n']= predict_df.iloc[15,0]
df_con.loc[df_con['mc']==3, 'predict_bim_n']= predict_df.iloc[16,0]
df_con.loc[df_con['mc']==5, 'predict_bim_n']= predict_df.iloc[17,0]
df_con.loc[df_con['mc']==7, 'predict_bim_n']= predict_df.iloc[18,0]
df_con.loc[df_con['mc']==9, 'predict_bim_n']= predict_df.iloc[19,0]
df_con.loc[df_con['mc']==11, 'predict_bim_n']= predict_df.iloc[20,0]
df_con.loc[df_con['mc']==13, 'predict_bim_n']= predict_df.iloc[21,0]
df_con.loc[df_con['mc']==15, 'predict_bim_n']= predict_df.iloc[22,0]
df_con.loc[df_con['mc']==17, 'predict_bim_n']= predict_df.iloc[23,0]
df_con.loc[df_con['mc']==19, 'predict_bim_n']= predict_df.iloc[24,0]


# In[24]:


#df[(df['mc']<0)]['predict_bim_n'].describe()


# In[25]:


def plot_RRD_curve_con(data):
    
    plt.pyplot.grid(True)
    df_p_c = df_con[['predict_bim_n','mc']]
    df_plot_c = df_p_c.dropna()
    
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(70000, 95000, 5000)
       
    df_untreat_c = df_plot_c[df_plot_c['mc'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),df_untreat_c['predict_bim_n'], 1)
    plt.pyplot.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')
    
    df_treat_c = df_plot_c[df_plot_c['mc'] >= 0]
    m, b = np.polyfit(np.arange(1,20,2),df_treat_c['predict_bim_n'], 1)
    plt.pyplot.plot(np.arange(1,20,2), m*np.arange(1,20,2) + b, color='green')   
 
    return


# In[26]:


#plot_RRD_curve_con(df_con)


# In[27]:


df_p = df_con[['predict_bim_n','mc']]
df_plot = df_p.dropna()
df_treat = df_plot[df_plot['mc'] >= 0]
    #m, b = np.polyfit(np.arange(-29,0,2),df_untreat['predict_bim_n'], 1)
#df_treat


# In[28]:


def plot_figure1_con(data):
    df_plot = df_con[['predict_bim_n','mc','mc2','mc3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(70000, 95000, 5000)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('Month of conception (0=July 2007)', fontsize = 11)
    #plt.pyplot.ylabel('Number of conception')
    plt.pyplot.plot(df_con.mc, df_con.bim_n, 'o')
    plt.pyplot.grid(True)    
    plot_RRD_curve_con(data)

    plt.pyplot.title("Figure 1. Number of conceptions by month, 2005-2009")
    return


# In[29]:


#plot_figure1_con(df_con)


# In[ ]:





# In[30]:


##========[Abortions]================
#1&2 access data and summary
df_abo=pd.read_stata("data/data_abortions_20110196.dta")
#df_ab


# In[31]:


#3. Sum abortions across all regions
#gen n_tot = n_ive_and + n_ive_val + n_ive_rioja + n_ive_cat + n_ive_can + n_ive_mad + n_ive_gal + n_ive_bal + n_ive_pv + n_ive_castlm + n_ive_ast + n_ive_arag 
df_abo['n_tot']=pd.concat([df_abo.n_ive_and, df_abo.n_ive_val, df_abo.n_ive_rioja, df_abo.n_ive_cat, df_abo.n_ive_can,df_abo.n_ive_mad, df_abo.n_ive_gal, df_abo.n_ive_bal, df_abo.n_ive_pv, df_abo.n_ive_castlm, df_abo.n_ive_ast, df_abo.n_ive_arag],1).sum(1)#df[(df['mc']>-30) & (df['mc']<20)]['predict_bim_n']#df_ab

#Alternative way: df_ab['n_tot'] = df_ab.n_ive_and.fillna(0) + df_ab.n_ive_val.fillna(0) + df_ab.n_ive_rioja.fillna(0) + df_ab.n_ive_arag .fillna(0)


# In[32]:


#4. Create month variable that takes value 0 in July 2007.*
#gen m = _n
#replace m = m - 103 

#m = the number of current observation (_n in STATA) Since python row starts from 0, I add 1
df_abo['m']= df_abo.reset_index().index + 1 
df_abo['m'] = df_abo['m'] - 103
df_abo['m'].describe()


# In[33]:


#5. Now I generate a variable indicating number of days in a month
#gen days=31
#replace days=30 if month==4 | month==6 | month==9 | month==11
#replace days=28 if month==2
#replace days=29 if month==2 & (year==2000 | year==2004 | year==2008)
#sum days

df_abo['days']=31
df_abo.loc[(df_abo['month']==4) | (df_abo['month']==6) | (df_abo['month']==9) | (df_abo['month']==11), 'days'] = 30
df_abo.loc[(df_abo['month']==2), 'days'] = 28
df_abo.loc[(df_abo['month']==2) & ((df_abo['year']==2000) | (df_abo['year']==2004) | (df_abo['year']==2008)), 'days'] = 29

#df_ab['days'].describe()


# In[34]:


# 7&8. Squared and cubed terms in m & Create Post dummy
#gen m2=m*m
#gen m3=m*m*m

df_abo['m2'] = df_abo['m']*df_abo['m']
df_abo['m3'] = df_abo['m']*df_abo['m']*df_abo['m']

#gen post=0
#replace post=1 if m>=0

df_abo['post']=0
df_abo.loc[(df_abo['m']>=0), 'post']=1


# In[35]:


# 9. Restrict period
#drop if m<-90 
#drop if m>29

df_abo = df_abo[~(df_abo.m < -90) & ~(df_abo.m > 29)]
#df_ab


# In[36]:


#Create Bimonthly number of conceptions

df_abo['bim_ab']=np.nan
df_abo.loc[(df_abo['m'] ==-29), 'bim_ab'] = df_abo.loc[df_abo['m']==-29,'n_tot'].sum() + df_abo.loc[df_abo['m']==-30,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-27), 'bim_ab'] = df_abo.loc[df_abo['m']==-27,'n_tot'].sum() + df_abo.loc[df_abo['m']==-28,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-25), 'bim_ab'] = df_abo.loc[df_abo['m']==-25,'n_tot'].sum() + df_abo.loc[df_abo['m']==-26,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-23), 'bim_ab'] = df_abo.loc[df_abo['m']==-23,'n_tot'].sum() + df_abo.loc[df_abo['m']==-24,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-21), 'bim_ab'] = df_abo.loc[df_abo['m']==-21,'n_tot'].sum() + df_abo.loc[df_abo['m']==-22,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-19), 'bim_ab'] = df_abo.loc[df_abo['m']==-19,'n_tot'].sum() + df_abo.loc[df_abo['m']==-20,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-17), 'bim_ab'] = df_abo.loc[df_abo['m']==-17,'n_tot'].sum() + df_abo.loc[df_abo['m']==-18,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-15), 'bim_ab'] = df_abo.loc[df_abo['m']==-15,'n_tot'].sum() + df_abo.loc[df_abo['m']==-16,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-13), 'bim_ab'] = df_abo.loc[df_abo['m']==-13,'n_tot'].sum() +df_abo.loc[df_abo['m']==-14,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-11), 'bim_ab'] = df_abo.loc[df_abo['m']==-11,'n_tot'].sum() +df_abo.loc[df_abo['m']==-12,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-9), 'bim_ab'] = df_abo.loc[df_abo['m']==-9,'n_tot'].sum() + df_abo.loc[df_abo['m']==-10,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-7), 'bim_ab'] = df_abo.loc[df_abo['m']==-7,'n_tot'].sum() + df_abo.loc[df_abo['m']==-8,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-5), 'bim_ab'] = df_abo.loc[df_abo['m']==-5,'n_tot'].sum() + df_abo.loc[df_abo['m']==-6,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-3), 'bim_ab'] = df_abo.loc[df_abo['m']==-3,'n_tot'].sum() +df_abo.loc[df_abo['m']==-4,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==-1), 'bim_ab'] = df_abo.loc[df_abo['m']==-1,'n_tot'].sum() +df_abo.loc[df_abo['m']==-2,'n_tot'].sum()

df_abo.loc[(df_abo['m'] ==1), 'bim_ab'] = df_abo.loc[df_abo['m']==1,'n_tot'].sum() + df_abo.loc[df_abo['m']==0,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==3), 'bim_ab'] = df_abo.loc[df_abo['m']==3,'n_tot'].sum() + df_abo.loc[df_abo['m']==2,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==5), 'bim_ab'] = df_abo.loc[df_abo['m']==5,'n_tot'].sum() + df_abo.loc[df_abo['m']==4,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==7), 'bim_ab'] = df_abo.loc[df_abo['m']==7,'n_tot'].sum() + df_abo.loc[df_abo['m']==6,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==9), 'bim_ab'] = df_abo.loc[df_abo['m']==9,'n_tot'].sum() + df_abo.loc[df_abo['m']==8,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==11), 'bim_ab'] = df_abo.loc[df_abo['m']==11,'n_tot'].sum() + df_abo.loc[df_abo['m']==10,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==13), 'bim_ab'] = df_abo.loc[df_abo['m']==13,'n_tot'].sum() + df_abo.loc[df_abo['m']==12,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==15), 'bim_ab'] = df_abo.loc[df_abo['m']==15,'n_tot'].sum() + df_abo.loc[df_abo['m']==14,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==17), 'bim_ab'] = df_abo.loc[df_abo['m']==17,'n_tot'].sum() + df_abo.loc[df_abo['m']==16,'n_tot'].sum()
df_abo.loc[(df_abo['m'] ==19), 'bim_ab'] = df_abo.loc[df_abo['m']==19,'n_tot'].sum() + df_abo.loc[df_abo['m']==18,'n_tot'].sum()

#df_abo['bim_ab'].tail(60)


# In[37]:


#df_ab[(df_ab['m']>-30) & (df_ab['m']<20)]['bim_ab'].describe()


# In[38]:


# 10. Regressions Table 
#Create interaction dummies
df_abo['ipost_1']=df_abo['post']*df_abo['m']
df_abo['ipost_2'] =df_abo['post']*df_abo['m2']
df_abo['ipost_3'] =df_abo['post']*df_abo['m3']

#RRD: Linear Regression (1)
#xi: reg bim_ab post i.post|m i.post|m2 i.post|m3 days, robust

#def figure1_conceptions(df):
Y_abo= 'bim_ab'
X_abo = ['m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']
abo_group = df_abo[['bim_ab','m','m2','m3','post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
result=mt.reg(abo_group[(df_abo['m']> -30)&(df_abo['m']<20)], 'bim_ab',['m','m2','post','ipost_1', 'ipost_2', 'days'], addcons=True)
predict_bim_ab = result.yhat.round(0)


# In[39]:


predict_bim_ab
predict_ab_int=predict_bim_ab.astype(int)


# In[40]:


predict_df_ab = pd.DataFrame(predict_ab_int)
predict_df_ab.rename(columns={0:'predict_bim_ab'}, inplace=True)
#predict_df_ab


# In[41]:


#df['predict_bim_n'] = np.nan
df_abo.loc[df_abo['m']==-29, 'predict_bim_ab']= predict_df_ab.iloc[0,0]
df_abo.loc[df_abo['m']==-27, 'predict_bim_ab']= predict_df_ab.iloc[1,0]
df_abo.loc[df_abo['m']==-25, 'predict_bim_ab']= predict_df_ab.iloc[2,0]
df_abo.loc[df_abo['m']==-23, 'predict_bim_ab']= predict_df_ab.iloc[3,0]
df_abo.loc[df_abo['m']==-21, 'predict_bim_ab']= predict_df_ab.iloc[4,0]
df_abo.loc[df_abo['m']==-19, 'predict_bim_ab']= predict_df_ab.iloc[5,0]
df_abo.loc[df_abo['m']==-17, 'predict_bim_ab']= predict_df_ab.iloc[6,0]
df_abo.loc[df_abo['m']==-15, 'predict_bim_ab']= predict_df_ab.iloc[7,0]
df_abo.loc[df_abo['m']==-13, 'predict_bim_ab']= predict_df_ab.iloc[8,0]
df_abo.loc[df_abo['m']==-11, 'predict_bim_ab']= predict_df_ab.iloc[9,0]
df_abo.loc[df_abo['m']==-9, 'predict_bim_ab']= predict_df_ab.iloc[10,0]
df_abo.loc[df_abo['m']==-7, 'predict_bim_ab']= predict_df_ab.iloc[11,0]
df_abo.loc[df_abo['m']==-5, 'predict_bim_ab']= predict_df_ab.iloc[12,0]
df_abo.loc[df_abo['m']==-3, 'predict_bim_ab']= predict_df_ab.iloc[13,0]
df_abo.loc[df_abo['m']==-1, 'predict_bim_ab']= predict_df_ab.iloc[14,0]
df_abo.loc[df_abo['m']==1, 'predict_bim_ab']= predict_df_ab.iloc[15,0]
df_abo.loc[df_abo['m']==3, 'predict_bim_ab']= predict_df_ab.iloc[16,0]
df_abo.loc[df_abo['m']==5, 'predict_bim_ab']= predict_df_ab.iloc[17,0]
df_abo.loc[df_abo['m']==7, 'predict_bim_ab']= predict_df_ab.iloc[18,0]
df_abo.loc[df_abo['m']==9, 'predict_bim_ab']= predict_df_ab.iloc[19,0]
df_abo.loc[df_abo['m']==11, 'predict_bim_ab']= predict_df_ab.iloc[20,0]
df_abo.loc[df_abo['m']==13, 'predict_bim_ab']= predict_df_ab.iloc[21,0]
df_abo.loc[df_abo['m']==15, 'predict_bim_ab']= predict_df_ab.iloc[22,0]
df_abo.loc[df_abo['m']==17, 'predict_bim_ab']= predict_df_ab.iloc[23,0]
df_abo.loc[df_abo['m']==19, 'predict_bim_ab']= predict_df_ab.iloc[24,0]

#df_ab[(df_ab['m']>-30) & (df_ab['m']<20)]['predict_bim_ab']


# In[42]:


def plot_figure1_ab(data):
    df_plot = df_abo[['predict_bim_ab','m','m2','m3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(10000, 22000, 2000)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('Month of abortion (0=July 2007)')
    #plt.pyplot.ylabel('Number of abortion')
    plt.pyplot.plot(df_abo.m, df_abo.bim_ab, 'o')
    plt.pyplot.grid(True)
    #plot_RRD_curve(df=df, running_variable="mc",outcome="predict_bim_n", cutoff=0)

    plt.pyplot.title("Figure 1-2. Number of abortions by month, 2005-2009")
    return


# In[43]:


#plot_figure1_ab(df_ab)


# In[44]:


def plot_RRD_curve_ab(data):
    plt.pyplot.grid(True)
    
    df_p1 = df_abo[['predict_bim_ab','m']]
    df_plot = df_p1.dropna()
       
    df_untreat = df_plot[df_plot['m'] < 0]
    m, b = np.polyfit(np.arange(-29,0,2),df_untreat['predict_bim_ab'], 1)
    plt.pyplot.plot(np.arange(-29,0,2), m*np.arange(-29,0,2) + b, color='orange')

    
    df_treat = df_plot[df_plot['m'] >= 0]        
    m, b = np.polyfit(np.arange(1,20,2),df_treat['predict_bim_ab'], 1)
    plt.pyplot.plot(np.arange(1,20,2), m*np.arange(1,20,2) + b, color='green')
    
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(10000, 22000, 2000)
 
    return


# In[45]:


#plot_RRD_curve_ab(df_ab)


# In[50]:


def plot_figure1_ab(data):
    df_plot = df_abo[['predict_bim_ab','m','m2','m3', 'post','ipost_1', 'ipost_2', 'ipost_3', 'days']]
    plt.pyplot.xlim(-30, 20, 10)
    plt.pyplot.ylim(10000, 22000, 2000)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('Month of abortion (0=July 2007)', fontsize = 11)
    #plt.pyplot.ylabel('Number of abortion')
    plt.pyplot.plot(df_abo.m, df_abo.bim_ab, 'o')
    plt.pyplot.grid(True)
    plot_RRD_curve_ab(df_abo)

    plt.pyplot.title("Figure 1-2. Number of abortions by month, 2005-2009")
    return


# In[51]:


#plot_figure1_ab(df_ab)


# In[52]:


def plot_figure1(data, data2):   
    plt.pyplot.figure(figsize=(13, 4))
    plt.pyplot.subplot(1, 2, 1)
    plot_figure1_con(data)
    plot_RRD_curve_con(data)
    plt.pyplot.title('Number of conceptions by month 2005-2009',fontsize = 13)
    
    plt.pyplot.subplot(1, 2, 2)
    plot_figure1_ab(data2)
    plot_RRD_curve_ab(data2)
    plt.pyplot.title('Number of abortions by month 2005-2009',fontsize = 13)
    plt.pyplot.suptitle("Figure 1. Fertility Effect: Conceptions and Abortions by Month", verticalalignment='bottom', fontsize=14)   
    return



#plot_figure1(df_con, df_abo)



