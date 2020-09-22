#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#*****************Female Labor Supply*********************************
et_ls=pd.read_stata("data/data_lfs_20110196.dta")


# In[3]:


#1.Control variables
#gen m2 = m*m
et_ls['m2']=et_ls['m']*et_ls['m']

#No father present.*/
#gen nodad=0
#replace nodad=1 if dadid==0
et_ls['nodad']=0
et_ls.loc[et_ls['dadid']==0, 'nodad']=1

# Mother not married 
#gen smom=0
#replace smom=1 if eciv!=2
et_ls['smom']=0
et_ls.loc[et_ls['eciv']!=2, 'smom']=1

#Mother single.*/
#gen single=0
#replace single=1 if eciv==1
et_ls['single']=0
et_ls.loc[et_ls['eciv']==1, 'single']=1

#Mother separated or divorced.*/
#gen sepdiv=0
#replace sepdiv=1 if eciv==4
et_ls['sepdiv']=0
et_ls.loc[et_ls['eciv']==4, 'sepdiv']=1

# No partner in the household.*/
#gen nopart=0
#replace nopart=1 if partner==0
et_ls['nopart']=0
et_ls.loc[et_ls['partner']==0, 'nopart']=1


# In[4]:


#***************Probability of the mother being in the maternity leave period at the time of the interview
#gen pleave=0
#replace pleave=0.17 if (q==1 & m==2) | (q==2 & m==5) | (q==3 & m==8) | (q==4 & m==11)
#replace pleave=0.5 if (q==1 & m==3) | (q==2 & m==6) | (q==3 & m==9) | (q==4 & m==12)
#replace pleave=0.83 if (q==1 & m==4) | (q==2 & m==7) | (q==3 & m==10) | (q==4 & m==13)
#replace pleave=1 if (q==1 & m>4 & m<9) | (q==2 & m>7 & m<12) | (q==3 & m>10 & m<15) | (q==4 & m>13)

et_ls['pleave']=0

et_ls.loc[(et_ls['q']==1) & (et_ls['m']==2)|(et_ls['q']==2) & (et_ls['m']==5)|(et_ls['q']==3) & (et_ls['m']==8)|(et_ls['q']==4) & (et_ls['m']==11) ,'pleave']=0.17
et_ls.loc[((et_ls['q']==1) & (et_ls['m']==3)) | ((et_ls['q']==2) & (et_ls['m']==6))  | ((et_ls['q']==3) & (et_ls['m']==9)) |((et_ls['q']==4) & (et_ls['m']==12)), 'pleave'] = 0.5
et_ls.loc[((et_ls['q']==1) & (et_ls['m']==4)) | ((et_ls['q']==2) & (et_ls['m']==7)) | ((et_ls['q']==3) & (et_ls['m']==10))  | ((et_ls['q']==4) & (et_ls['m']==13)), 'pleave'] = 0.83
et_ls.loc[((et_ls['q']==1) & (et_ls['m']>4) & (et_ls['m']<9)) | ((et_ls['q']==2) & (et_ls['m']>7) & (et_ls['m']<12)) | ((et_ls['q']==3) & (et_ls['m']>10) & (et_ls['m']<15))| ((et_ls['q']==4) & (et_ls['m']>13)), 'pleave'] = 1



# In[5]:


#**********************************Table 5 Regressions**************************************************
##**********************************5-1Working last week

#Create interaction dummies
et_ls['ipost_1']=et_ls['post']*et_ls['m']
et_ls['ipost_2'] =et_ls['post']*et_ls['m2']

#create iq dummies: iq_1 for quarter 1, iq_2 for quarter 2 and so on
for j in range(1,5):
    et_ls['iq_'+str(j)] = 0
    for i in range(len(et_ls)):
        if et_ls.loc[i,'q'] == j:
            et_ls.loc[i, 'iq_'+str(j)] = 1


# In[6]:


def figure_ET6(data):
    features1=list(['primary','hsgrad','univ','primary_dad','hsgrad_dad','univ_dad','sib','nodad' ,'smom','single' ,'sepdiv', 'nopart'])
    et_ls[features1].corr()
    #df[features2].corr()

    # Plot
    plt.figure(figsize=(12,9), dpi= 80)
    sns.heatmap(et_ls[features1].corr().corr(), xticklabels=et_ls[features1].corr(), yticklabels=et_ls[features1].corr().corr(), cmap='RdYlGn', center=0, annot=True)

    # Decorations
    plt.title('Correlogram of parents education level and marital status', fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
    return


# In[7]:


#figure_ET6(et_ls)


# In[8]:


def figure_ET3(data):
    x_var = 'm'
    groupby_var = 'eciv'
    df_agg = et_ls.loc[:, [x_var, groupby_var]].groupby(groupby_var)
    vals = [et_ls[x_var].values.tolist() for i, et_ls in df_agg]

    # Draw
    plt.figure(figsize=(13,7), dpi= 80)
    colors = [plt.cm.Spectral(i/float(len(vals)-1)) for i in range(len(vals))]
    n, bins, patches = plt.hist(vals, 50, stacked=True, density=False, color=colors[:len(vals)])

    # Decoration
    plt.legend({group:col for group, col in zip(np.unique(et_ls[groupby_var]).tolist(), colors[:len(vals)])})
    plt.title(f"Stacked Histogram of marital status", fontsize=20)
    plt.xlabel("m: month of birth (0=July 2007)")
    plt.ylabel("Frequency")
    plt.ylim(0, 1000)
    #plt.xticks(ticks=bins[::10], labels=[round(b,1) for b in bins[::10]])
    plt.show()

    #1: single, #2: married, #4: divorced
    return


# In[9]:


#figure_ET3(et_ls)


# In[ ]:





# In[10]:


def figure_ET4(data):
    # Import Data

    # Draw Plot
    plt.figure(figsize=(13,6), dpi= 80)
    sns.distplot(et_ls.loc[et_ls['univ'] == 1, "m"], color="dodgerblue", label="univ", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['primary'] == 1, "m"], color="orange", label="primary", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['hsgrad'] == 1, "m"], color="salmon", label="hsgrad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})


    plt.ylim(0, 0.05)

    # Decoration
    plt.title('Density Plot of Mother Education Attainment', fontsize=20)
    plt.legend()
    plt.show()
    return


# In[11]:


#figure_ET4(et_ls)


# In[12]:


def figure_ET5(data):

    # Draw Plot
    plt.figure(figsize=(13,6), dpi= 80)
    sns.distplot(et_ls.loc[et_ls['univ_dad'] == 1, "m"], color="g", label="univ_dad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['primary_dad'] == 1, "m"], color="dodgerblue", label="primary_dad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    sns.distplot(et_ls.loc[et_ls['hsgrad_dad'] == 1, "m"], color="orange", label="hsgrad_dad", hist_kws={'alpha':.6}, kde_kws={'linewidth':3})
    plt.ylim(0, 0.05)

    # Decoration
    plt.title('Density Plot of Father Education Attainment', fontsize=20)
    plt.legend()
    plt.show()
    return


# In[13]:


#figure_ET5(et_ls)


# In[ ]:





# In[14]:


#RDD9m(1)
#xi: reg work post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
#ef RD_WL_1(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    #return(result)


# In[15]:


#1-1. Primary
def P_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[20]:


Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

pwl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary



# In[22]:


#1-2. Hsgrad
def HS_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[24]:


Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

hswl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary



# In[ ]:


#1-3. Univ
def UNI_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

uniwl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:    
#1-4. nopart, nodad
def NPND_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nodad','nopart','sepdiv','single','smom','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['nodad','nopart','sepdiv','single','smom','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

npndwl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#1-5. single,smom
def SG_WL_1(et_ls):

    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nodad','nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index=et_ls[['nodad','nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

sgwl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
  
    
# In[ ]:
#1-6. smom, nopart
def NPS_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','smom','nopart','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index=et_ls[['nodad','nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

npswl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','smom','nopart','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#1-7 nodad, smom
def NDS_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

ndswl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:
#1-8 nopart, sepdiv
def NPDIV_WL_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index=et_ls[['nodad','nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
npdivwl1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work', ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:
#RDD6m(2)xi: reg work post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-7 & m<6, robust 
#def RD_WL_2(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    #return(result)


# In[ ]:


#2-1. Primary
def P_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

pwl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:

#2-2 Hsgrad
def HS_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)

# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
hswl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:


#2-3 Univ
def UNI_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
uniwl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:


#2-4 nopart, nodad 0.99
def NPND_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','nopart','nodad']]
npndwl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:
#2-5 single, smom 
def SG_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)



# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','single','smom','iq_2', 'iq_3', 'iq_4']]
sgwl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:
#2-6 nopart, smom 
def NPS_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','nopart','smom']]
npswl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#2-7 nodad, smom 
def NDS_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','nodad','smom']]
ndswl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
 

# In[ ]:

#2-8 nopart, sepdiv 
def NPDIV_WL_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)

# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','nopart','sepdiv']]
npdivwl2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary



# In[ ]:


#RDD4m(3)xi: reg work post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-5 & m<4, robust 
#def RD_WL_3(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['sepdiv','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    #return(result)


# In[ ]:
#3-1 Primary
def P_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
pwl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
 

# In[ ]:

#3-2 Hsgrad
def HS_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)



# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

hswl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
                 


#3-3 Univ
def UNI_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
                     
uniwl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
   
                     
                     
# In[ ]:
#3-4 nopart, nodad 
def NPND_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','nopart','nodad']]
                     
npndwl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
 
                     
# In[ ]:
#3-5 single, smom 
def SG_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','single','smom']]
sgwl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

                     
# In[ ]:
#3-6 nopart, smom
def NPS_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
npswl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
     
                     
# In[ ]:
#3-7 nodad, smom 
def NDS_WL_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
ndswl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
 
                     
# In[ ]:
#3-8 nopart, sepdiv 
def NPDIV_WL_3(et_ls): 
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npdivwl3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

                     
                     
# In[ ]:
#RDD3m(4)xi: reg work post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-4 & m<3, robust
#def RD_WL_4(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['sepdiv','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    #return(result)


# In[ ]:


#4-1 Primary
def P_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
# In[ ]:

# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
pwl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
                     

                     
# In[ ]:
#4-2 Hsgrad
def HS_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
hswl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
  
# In[ ]:


#4-3 Univ
def UNI_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniwl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
 
# In[ ]:
#4-4 nopart, nodad 
def NPND_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
npndwl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:

#4-5 single, smom 
def SG_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
sgwl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary 
# In[ ]:
#4-6 nopart, smom 
def NPS_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npswl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#4-7 nodad, smom 
def NDS_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
ndswl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
  
# In[ ]:
#4-8 nopart, sepdiv 
def NPDIV_WL_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npdivwl4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work', ['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:


# In[ ]:


#RDD3m(5)xi: reg work post if m>-3 & m<2, robust 
#def RD_WL_5(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post'], addcons=True)
    #return(result)


# In[ ]:


#5-1 Primary
def P_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','primary','primary_dad'], addcons=True)
    return(result)


# In[ ]:
pwl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','primary','primary_dad'], addcons=True).summary
# In[ ]:
#5-2 Hsgrad
def HS_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad'], addcons=True)
    return(result)



# In[ ]:
hswl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad'], addcons=True).summary
 
# In[ ]:
#5-3 Univ
def UNI_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad'], addcons=True)
    return(result)


# In[ ]:
uniwl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad'], addcons=True).summary
# In[ ]:
#5-4 nopart, nodad
def NPND_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad'], addcons=True)
    return(result)


# In[ ]:
npndwl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad'], addcons=True).summary
# In[ ]:
#5-5 single, smom 
def SG_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom'], addcons=True)
    return(result)


# In[ ]:
sgwl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom'], addcons=True).summary
# In[ ]:
#5-6 nopart, smom 
def NPS_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom'], addcons=True)
    return(result)


# In[ ]:
npswl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom'], addcons=True).summary
# In[ ]:
#5-7 nopart, smom 
def NDS_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom'], addcons=True)
    return(result)


# In[ ]:
ndswl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom'], addcons=True).summary
# In[ ]:
#5-8 nopart, sepdiv 
def NPDIV_WL_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv'], addcons=True)
    return(result)

# In[ ]:
npdivwl5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv'], addcons=True).summary


# In[ ]:





# In[ ]:


#RDD2m(6)xi: reg work post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-3 & m<2, robust
##note: pleave omitted because of collinearity
#def RD_WL_6(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','iq_2', 'iq_3', 'iq_4'],addcons=True)
    #return(result)


# In[ ]:


#6-1 Primary
def P_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
pwl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-2 Hsgrad
def HS_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
hswl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-3 Univ
def UNI_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniwl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-4 nopart, nodad 
def NPND_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npndwl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','nodad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#6-5 single, smom 
def SG_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
sgwl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','single','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-6 nopart, smom
def NPS_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npswl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-7 nodad, smom 
def NDS_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nodad','smom','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
ndswl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nodad','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-8 nopart, sepdiv 
def NPDIV_WL_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npdivwl6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work', ['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4'], addcons=True).summary




# In[ ]:

# In[ ]:
#DID(7)xi: reg work post m m2 age age2 age3 immig primary hsgrad univ sib pleave i.n_month i.q, robust ""cluster(m)""

#create i_n_month: in_month_1 for January, in_month_2 for February and so on
for j in range(1,13):
    et_ls['i_n_month_'+str(j)] = 0
    for i in range(len(et_ls)):
        if et_ls.loc[i,'n_month'] == j:
            et_ls.loc[i, 'i_n_month_'+str(j)] = 1


# In[ ]:


#def DID_WL_7(et_ls):
#Y_ls_DID = ['work','work2']
#X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

#et_ls_DID = et_ls[['work','work2','post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

#mt.reg(et_ls_DID, 'work', ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    #return(result)


# In[ ]:


#7-1 primary
def P_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
Y_ls_DID = ['work','work2']
X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

pwl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-2 hsgrad
def HS_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
hswl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
# In[ ]:
#7-3 univ
def UNI_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
uniwl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
# In[ ]:
#7-4 nopart, nodad 
def NPND_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
Y_ls_DID = ['work','work2']
X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']
et_ls_DID=et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]
npndwl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
# In[ ]:
#7-5  single, smom
def SG_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
sgwl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
# In[ ]:
#7-6 nopart, smom
def NPS_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
npswl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-7 nodad, smom
def NDS_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
ndswl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
# In[ ]:
#7-8  nopart, sepdiv
def NPDIV_WL_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work', ['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
npdivwl7=mt.reg(et_ls_DID, 'work', ['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary


# In[ ]:

# In[ ]:


##*************************************5-2 Employed******************************************************/
#RDD9m(1)xi: reg work2 post i.post|m i.post|m2 age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-10 & m<9, robust 
#def RD_EP_1(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    #return(result)


# In[ ]:


#1-1.1 Primary
def P_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

pep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#1-2.1 Hsgrad
def HS_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
hsep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:

#1-3.1 Univ
def UNI_EP_1(et_ls): 
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#1-4.1 nopart, nodad
def NPND_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nodad','nopart','sepdiv','single','smom','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
Y_ls= ['work','work2']
X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
et_ls_index = et_ls[['nodad','nopart','sepdiv','single','smom','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

npndep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nodad','nopart','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#1-5.1 single,smom
def SG_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nodad','nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
sgep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','smom','single','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:

#1-6.1 nopart, smom
def NPS_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)

    return(result)


# In[ ]:
npsep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:

#1-7.1 nodad, smom
def NDS_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
ndsep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#1-8 nopart, sepdiv
def NPDIV_EP_1(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npdivep1=mt.reg(et_ls_index[(et_ls['m']> -10) & (et_ls['m']< 9)], 'work2', ['post','m','m2','ipost_1', 'ipost_2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:

# In[ ]:


#RDD6m(2): xi: reg work2 post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-7 & m<6, robust 
#def RD_EP_2(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)


# In[ ]:


#2-1.1 Primary
def P_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
pep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#2-2.1 Hsgrad
def HS_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
hsep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#2-3.1 Univ
def UNI_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#2-4.1 nopart, nodad 
def NPND_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npndep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#2-5.1 single, smom 
def SG_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)

# In[ ]:
sgep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:


#2-6.1  nopart, smom 
def NPS_EP_2(et_ls):

    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npsep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:

#2-7.1  nodad, smom 
def NDS_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
# In[ ]:
ndsep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:


#2-8.1 nopart, sepdiv 
def NPDIV_EP_2(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npdivep2=mt.reg(et_ls_index[(et_ls['m']> -7) & (et_ls['m']<6)], 'work2', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:


# In[ ]:


#RDD4m(3)xi: reg work2 post i.post|m age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-5 & m<4, robust 
#def RD_EP_3(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    #return(result)


# In[ ]:


#3-1.1 primary
def P_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
pep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#3-2.1 hsgard
def HS_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
hsep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#3-3.1 univ
def UNI_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#3-4.1
def NPND_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npndep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#3-5.1 single, smom
def SG_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)



# In[ ]:
sgep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#3-6.1  nopart, smom
def NPS_EP_3(et_ls): 
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npsep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#3-7.1 nodad, smom 
def NDS_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
ndsep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#3-8.1 nopart, sepdiv 
def NPDIV_EP_3(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:

npdivep3=mt.reg(et_ls_index[(et_ls['m']> -5) & (et_ls['m']<4)], 'work2', ['post','m','ipost_1','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary



# In[ ]:


# In[ ]:
#RDD3m(4)xi: reg work2 post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-4 & m<3, robust
#def RD_EP_4(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
#mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4'],addcons=True)
    


# In[ ]:


#4-1 Primary
def P_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','primary','primary_dad','pleave','iq_2','iq_3','iq_4'], addcons=True)
    return(result)

# In[ ]:
pep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','primary','primary_dad','pleave','iq_2','iq_3','iq_4'], addcons=True).summary

# In[ ]:


#4-2 Hsgrad
def HS_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
    


# In[ ]:
hsep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#4-3 Univ
def UNI_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#4-4 nopart, nodad 
def NPND_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npndep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#4-5 single, smom
def SG_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
sgep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','single','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary


# In[ ]:
#4-6 nopart, smom 
def NPS_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npsep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#4-7 nodad, smom 
def NDS_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
ndsep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nodad','smom','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:
#4-8 nopart, sepdiv 
def NPDIV_EP_4(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
# In[ ]:
npdivep4=mt.reg(et_ls_index[(et_ls['m']> -4) & (et_ls['m']<3)], 'work2', ['post','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:





# In[ ]:


#RDD2m(5)xi: reg work2 post if m>-3 & m<2, robust 
#def RD_EP_5(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]
#m#t.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post'],addcons=True)
    


# In[ ]:


#5-1 Primary
def P_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','primary','primary_dad'], addcons=True)
    return(result)


# In[ ]:
pep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','primary','primary_dad'], addcons=True).summary
# In[ ]:
#5-2 Hsgrad
def HS_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','hsgrad','hsgrad_dad'], addcons=True)
    return(result)


# In[ ]:
hsep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','hsgrad','hsgrad_dad'], addcons=True).summary
# In[ ]:
#5-3 Univ
def UNI_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','univ','univ_dad'], addcons=True)
    return(result)
# In[ ]:
uniep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','univ','univ_dad'], addcons=True).summary
# In[ ]:


#5-4 nopart, nodad 
def NPND_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','nodad'], addcons=True)
    return(result)
# In[ ]:
npndep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','nodad'], addcons=True).summary
# In[ ]:



#5-5 single, smom 
def SG_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','single','smom'], addcons=True)
    return(result)


# In[ ]:
sgep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','single','smom'], addcons=True).summary
# In[ ]:
#5-6 nopart, smom 
def NPS_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','smom'], addcons=True)
    return(result)


# In[ ]:
npsep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','smom'], addcons=True).summary
# In[ ]:
#5-7 nopart, smom 0.73
def NDS_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','smom'], addcons=True)
    return(result)


# In[ ]:
ndsep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','smom'], addcons=True).summary
# In[ ]:
#5-8 nopart, sepdiv 
def NPDIV_EP_5(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','sepdiv'], addcons=True)
    return(result)


# In[ ]:
npdivep5=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','sepdiv'], addcons=True).summary
# In[ ]:
#RDD2m(6)xi: reg work2 post age age2 age3 immig primary hsgrad univ sib pleave i.q if m>-3 & m<2, robust
#def RD_EP_6(et_ls):
#Y_ls= ['work','work2']
#X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
#et_ls_index = et_ls[['work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

#mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','age','age2','age3','immig','primary','hsgrad','univ','sib','iq_2', 'iq_3', 'iq_4'],addcons=True)
    #return(result)
##note: pleave omitted because of collinearity


# In[ ]:


#6-1 Primary
def P_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
# In[ ]:
pep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','primary','primary_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:


#6-2 Hsgrad
def HS_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
hsep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','hsgrad','hsgrad_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-3 Univ
def UNI_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
uniep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','univ','univ_dad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-4 nopart, nodad
def NPND_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','nodad','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:

npndep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','nodad','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:

#6-5 single, smom 
def SG_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','single','smom','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
sgep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','single','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#6-6 nopart, smom 
def NPS_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','smom','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)
    


# In[ ]:
npsep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary

# In[ ]:

#6-7 nodad, smom 
def NDS_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nodad','smom','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
ndsep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nodad','smom','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
 
# In[ ]:

#6-8 nopart, sepdiv 
def NPDIV_EP_6(et_ls):
    Y_ls= ['work','work2']
    X_ls = ['post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']
    et_ls_index = et_ls[['nopart','sepdiv','single','smom','nodad','primary_dad','univ_dad','hsgrad_dad','work','work2','post','m','m2','ipost_1', 'ipost_2', 'age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4']]

    result=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4'], addcons=True)
    return(result)


# In[ ]:
npdivep6=mt.reg(et_ls_index[(et_ls['m']> -3) & (et_ls['m']<2)], 'work2', ['post','nopart','sepdiv','iq_2', 'iq_3', 'iq_4'], addcons=True).summary
# In[ ]:
#DID(7):xi: reg work2 post m m2 age age2 age3 immig primary hsgrad univ sib pleave i.n_month i.q, robust cluster(m)
#def DID_EP_7(et_ls):
#Y_ls_DID = ['work','work2']
#X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

#et_ls_DID = et_ls[['work','work2','post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

#mt.reg(et_ls_DID, 'work2', ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    #return(result)


# In[ ]:


#7-1 primary
def P_EP_7(et_ls): 
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
Y_ls_DID = ['work','work2']
X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

pep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','primary','primary_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
 
# In[ ]:
#7-2 hsgrad
def HS_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
hsep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','hsgrad','hsgrad_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-3 univ
def UNI_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
uniep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','univ','univ_dad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-4 nopart, nodad 
def NPND_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
Y_ls_DID = ['work','work2']
X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

et_ls_DID=et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

npndep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','nopart','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-5  single, smom
def SG_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
sgep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','single','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-6 nopart, smom
def NPS_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
npsep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','nopart','smom','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:
#7-7 nodad, smom
def NDS_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
ndsep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','smom','nodad','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary
# In[ ]:

#7-8  nopart, sepdiv
def NPDIV_EP_7(et_ls):
    Y_ls_DID = ['work','work2']
    X_ls_DID = ['post','m','m2','age','age2','age3','immig','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']

    et_ls_DID = et_ls[['nopart','nodad','single','smom','sepdiv','work','work2','post','m','m2','primary_dad','hsgrad_dad','univ_dad','primary','hsgrad','univ','sib','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12']]

    result=mt.reg(et_ls_DID, 'work2', ['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True)
    return(result)


# In[ ]:
npdivep7=mt.reg(et_ls_DID, 'work2', ['post','m','m2','nopart','sepdiv','pleave','iq_2', 'iq_3', 'iq_4','i_n_month_2','i_n_month_3','i_n_month_4','i_n_month_5','i_n_month_6','i_n_month_7','i_n_month_8','i_n_month_9','i_n_month_10','i_n_month_11','i_n_month_12'], cluster='m', addcons=True).summary

# In[ ]:


# In[ ]:


##================Create Table E1
def tableET1(et_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID(7)': []})
        result = ('Working Last Week_Primary', 'Primary_pvalue', 'Primary_SE', 'Working_Highschool', 'HS_pvalue','HS_SE',
                  'Working_Uni', 'Uni_pvalue', 'Uni_SE',
                  'Employed_Primary','E_Primary_pvalue', 'E_Primary_SE','Employed_Highschool',
                  'E_HS_pvalue','E_HS_SE', 'Employed_Uni', 'E_Uni_pvalue','E_Uni_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Education Level'] = result
        table = table.set_index('Education Level')

##=================Working Last Week 
        #1. Working Last Week under Primary educated parents 
        P_RD1 = P_WL_1(et_ls)
        P_RD2 = P_WL_2(et_ls)
        P_RD3 = P_WL_3(et_ls)
        P_RD4 = P_WL_4(et_ls)
        P_RD5 = P_WL_5(et_ls)
        P_RD6 = P_WL_6(et_ls) 
        P_DID7 = P_WL_7(et_ls)
        
        pr = [P_RD1.beta['post'], P_RD2.beta['post'], P_RD3.beta['post'], P_RD4.beta['post'], P_RD5.beta['post'],
                 P_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Working Last Week_Primary"] = pr
        
        #Primary_pvalue 
        p_pv = [pwl1['p>t']['post'], pwl2['p>t']['post'],pwl3['p>t']['post'], pwl4['p>t']['post'],pwl5['p>t']['post'],
                pwl6['p>t']['post'], pwl7['p>t']['post']]
        table.loc["Primary_pvalue"]=p_pv
        
        #Primary Standard Error
        p_se = [P_RD1.se['post'],P_RD2.se['post'], P_RD3.se['post'], P_RD4.se['post'], P_RD5.se['post'],
                 P_RD6.se['post'], P_DID7.se['post']]
        table.loc["Primary_SE"] = p_se
        
        #2. HS coefficient
        HS_RD1 = HS_WL_1(et_ls)
        HS_RD2 = HS_WL_2(et_ls)
        HS_RD3 = HS_WL_3(et_ls)
        HS_RD4 = HS_WL_4(et_ls)
        HS_RD5 = HS_WL_5(et_ls)
        HS_RD6 = HS_WL_6(et_ls) 
        HS_DID7 = HS_WL_7(et_ls)
        
        hs = [HS_RD1.beta['post'], HS_RD2.beta['post'], HS_RD3.beta['post'], HS_RD4.beta['post'], HS_RD5.beta['post'],
                 HS_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Working_Highschool"] = hs
        
        #Highschool_pvalue 
        hs_pv = [hswl1['p>t']['post'], hswl2['p>t']['post'],hswl3['p>t']['post'], hswl4['p>t']['post'],hswl5['p>t']['post'],
                hswl6['p>t']['post'], hswl7['p>t']['post']]
        table.loc["HS_pvalue"]=hs_pv
        
        #Highschool Standard Error
        hs_se = [HS_RD1.se['post'],HS_RD2.se['post'], HS_RD3.se['post'], HS_RD4.se['post'], HS_RD5.se['post'],
                 HS_RD6.se['post'], P_DID7.se['post']]
        table.loc["HS_SE"] = hs_se
        
        
        #3. Uni coefficient
        UNI_RD1 = UNI_WL_1(et_ls)
        UNI_RD2 = UNI_WL_2(et_ls)
        UNI_RD3 = UNI_WL_3(et_ls)
        UNI_RD4 = UNI_WL_4(et_ls)
        UNI_RD5 = UNI_WL_5(et_ls)
        UNI_RD6 = UNI_WL_6(et_ls) 
        UNI_DID7 = UNI_WL_7(et_ls)
        
        uni = [UNI_RD1.beta['post'], UNI_RD2.beta['post'], UNI_RD3.beta['post'], UNI_RD4.beta['post'], UNI_RD5.beta['post'],
                 UNI_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Working_Uni"] = uni
        
        #Uni_pvalue 
        uni_pv = [uniwl1['p>t']['post'], uniwl2['p>t']['post'],uniwl3['p>t']['post'], uniwl4['p>t']['post'],uniwl5['p>t']['post'],
                uniwl6['p>t']['post'], uniwl7['p>t']['post']]
        table.loc["Uni_pvalue"]=uni_pv
        
        #Uni SE 
        uni_se = [UNI_RD1.se['post'],UNI_RD2.se['post'], UNI_RD3.se['post'], UNI_RD4.se['post'], UNI_RD5.se['post'],
                 UNI_RD6.se['post'], P_DID7.se['post']]
        table.loc["Uni_SE"] = uni_se

###=============Employed
        #1. Employed under Primary educated parents 
        P_RD_EP1 = P_EP_1(et_ls)
        P_RD_EP2 = P_EP_2(et_ls)
        P_RD_EP3 = P_EP_3(et_ls)
        P_RD_EP4 = P_EP_4(et_ls)
        P_RD_EP5 = P_EP_5(et_ls)
        P_RD_EP6 = P_EP_6(et_ls) 
        P_DID7 = P_EP_7(et_ls)
        
        pr_ep = [P_RD_EP1.beta['post'], P_RD_EP2.beta['post'], P_RD_EP3.beta['post'], P_RD_EP4.beta['post'], P_RD_EP5.beta['post'],
                 P_RD_EP6.beta['post'], P_DID7.beta['post']]
        table.loc["Employed_Primary"] = pr_ep
        
        #Primary_pvalue 
        p_ep_pv = [pep1['p>t']['post'], pep2['p>t']['post'],pep3['p>t']['post'], pep4['p>t']['post'],pep5['p>t']['post'],
                pep6['p>t']['post'], pep7['p>t']['post']]
        table.loc["E_Primary_pvalue"]=p_ep_pv
        
        #Primary StandaRD_EP Error
        p_ep_se = [P_RD_EP1.se['post'],P_RD_EP2.se['post'], P_RD_EP3.se['post'], P_RD_EP4.se['post'], P_RD_EP5.se['post'],
                 P_RD_EP6.se['post'], P_DID7.se['post']]
        table.loc["E_Primary_SE"] = p_ep_se
        
        #2. HS coefficient
        HS_RD_EP1 = HS_EP_1(et_ls)
        HS_RD_EP2 = HS_EP_2(et_ls)
        HS_RD_EP3 = HS_EP_3(et_ls)
        HS_RD_EP4 = HS_EP_4(et_ls)
        HS_RD_EP5 = HS_EP_5(et_ls)
        HS_RD_EP6 = HS_EP_6(et_ls) 
        HS_DID7 = HS_EP_7(et_ls)
        
        hs_ep = [HS_RD_EP1.beta['post'], HS_RD_EP2.beta['post'], HS_RD_EP3.beta['post'], HS_RD_EP4.beta['post'], HS_RD_EP5.beta['post'],
                 HS_RD_EP6.beta['post'], P_DID7.beta['post']]
        table.loc["Employed_Highschool"] = hs_ep
        
        #Highschool_pvalue (Table5-copy)
        hs_ep_pv = [hsep1['p>t']['post'], hsep2['p>t']['post'],hsep3['p>t']['post'], hsep4['p>t']['post'],hsep5['p>t']['post'],
                hsep6['p>t']['post'], hsep7['p>t']['post']]
        table.loc["E_HS_pvalue"]=hs_ep_pv
        
        #Highschool StandaRD_EP Error
        hs_ep_se = [HS_RD_EP1.se['post'],HS_RD_EP2.se['post'], HS_RD_EP3.se['post'], HS_RD_EP4.se['post'], HS_RD_EP5.se['post'],
                 HS_RD_EP6.se['post'], P_DID7.se['post']]
        table.loc["E_HS_SE"] = hs_ep_se

        #Employed coefficient under college educated paretns
        UNI_EP_RD1 = UNI_EP_1(et_ls)
        UNI_EP_RD2 = UNI_EP_2(et_ls)
        UNI_EP_RD3 = UNI_EP_3(et_ls)
        UNI_EP_RD4 = UNI_EP_4(et_ls)
        UNI_EP_RD5 = UNI_EP_5(et_ls)
        UNI_EP_RD6 = UNI_EP_6(et_ls) 
        UNI_DID7 = UNI_EP_7(et_ls)
        
        uni_ep = [UNI_EP_RD1.beta['post'], UNI_EP_RD2.beta['post'], UNI_EP_RD3.beta['post'], UNI_EP_RD4.beta['post'], UNI_EP_RD5.beta['post'],
                 UNI_EP_RD6.beta['post'], P_DID7.beta['post']]
        table.loc["Employed_Uni"] = uni_ep
        
        #Uni_pvalue 
        uni_ep_pv = [uniep1['p>t']['post'], uniep2['p>t']['post'],uniep3['p>t']['post'], uniep4['p>t']['post'],uniep5['p>t']['post'],
                     uniep6['p>t']['post'], uniep7['p>t']['post']]
        table.loc["E_Uni_pvalue"]=uni_ep_pv
        
        #Uni SE 
        uni_ep_se = [UNI_EP_RD1.se['post'],UNI_EP_RD2.se['post'], UNI_EP_RD3.se['post'], UNI_EP_RD4.se['post'], UNI_EP_RD5.se['post'],
                 UNI_EP_RD6.se['post'], P_DID7.se['post']]
        table.loc["E_Uni_SE"] = uni_ep_se

    
    
    
 #####================================================================================       
        #Observations
        table=table.astype(float).round(4)
        obs =[P_RD1.N, P_RD2.N, P_RD3.N, P_RD4.N, P_RD5.N, P_RD6.N, P_DID7.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48]
        table.loc["Number of months"] = months

        
        return(table)


# In[ ]:


#tableET1(et_ls)


# In[ ]:


##================Create Table E2
def tableET2(et_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID(7)': []})
        result = ('1.NoPartner_NoDad', 'NoPartner_NoDad_pvalue', 'NoPartner_NoDad_SE', 
                  '2.Singlemom', 'Singlemom_pvalue','Singlemom_SE',
                  '3.NoPartner_NotMarried', 'NoPartner_NotMarried_pvalue', 'NoPartner_NotMarried_SE',
                  '4.NoDad_NotMarried', 'NoDad_NotMarried_pvalue', 'NoDad_NotMarried_SE',
                  '5.Divorced_NoPartner','Divorced_NoPartner_pvalue','Divorced_NoPartner_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Marital Status_WLS'] = result
        table = table.set_index('Marital Status_WLS')

##=================Working Last Week 
        #1. Working Last Week under No Part & No Dad
        NPND_RD1 = NPND_WL_1(et_ls)
        NPND_RD2 = NPND_WL_2(et_ls)
        NPND_RD3 = NPND_WL_3(et_ls)
        NPND_RD4 = NPND_WL_4(et_ls)
        NPND_RD5 = NPND_WL_5(et_ls)
        NPND_RD6 = NPND_WL_6(et_ls) 
        NPND_DID7 = NPND_WL_7(et_ls)
        
        npnd = [NPND_RD1.beta['post'], NPND_RD2.beta['post'], NPND_RD3.beta['post'], NPND_RD4.beta['post'], NPND_RD5.beta['post'],
                 NPND_RD6.beta['post'], NPND_DID7.beta['post']]
        table.loc["1.NoPartner_NoDad"] = npnd
        
        #NoPartner_NoDad_pvalue 
        npnd_pv = [npndwl1['p>t']['post'], npndwl2['p>t']['post'],npndwl3['p>t']['post'], npndwl4['p>t']['post'],npndwl5['p>t']['post'],
                npndwl6['p>t']['post'], npndwl7['p>t']['post']]
        table.loc["NoPartner_NoDad_pvalue"]=npnd_pv
        
        #NoPartner_NoDad Standard Error
        npnd_se = [NPND_RD1.se['post'],NPND_RD2.se['post'], NPND_RD3.se['post'], NPND_RD4.se['post'], NPND_RD5.se['post'],
                 NPND_RD6.se['post'], NPND_DID7.se['post']]
        table.loc["NoPartner_NoDad_SE"] = npnd_se
        
        #1-2 Working Last Week under Singlemom
        SG_RD1 = SG_WL_1(et_ls)
        SG_RD2 = SG_WL_2(et_ls)
        SG_RD3 = SG_WL_3(et_ls)
        SG_RD4 = SG_WL_4(et_ls)
        SG_RD5 = SG_WL_5(et_ls)
        SG_RD6 = SG_WL_6(et_ls) 
        SG_DID7 = SG_WL_7(et_ls)
        
        
        SG = [SG_RD1.beta['post'], SG_RD2.beta['post'], SG_RD3.beta['post'], SG_RD4.beta['post'], SG_RD5.beta['post'],
                 SG_RD6.beta['post'], SG_DID7.beta['post']]
        table.loc["2.Singlemom"] = SG
        
        #Singlemom_pvalue 
        SG_pv = [sgwl1['p>t']['post'], sgwl2['p>t']['post'],sgwl3['p>t']['post'], sgwl4['p>t']['post'],sgwl5['p>t']['post'],
                sgwl6['p>t']['post'], sgwl7['p>t']['post']]

        table.loc["Singlemom_pvalue"]=SG_pv
        
        #Singlemom Standard Error
        SG_se = [SG_RD1.se['post'],SG_RD2.se['post'], SG_RD3.se['post'], SG_RD4.se['post'], SG_RD5.se['post'],
                 SG_RD6.se['post'], SG_DID7.se['post']]
        table.loc["Singlemom_SE"] = SG_se
        
        #1-3 Working Last Week under NoPartner_NotMarried
        NPS_RD1 = NPS_WL_1(et_ls)
        NPS_RD2 = NPS_WL_2(et_ls)
        NPS_RD3 = NPS_WL_3(et_ls)
        NPS_RD4 = NPS_WL_4(et_ls)
        NPS_RD5 = NPS_WL_5(et_ls)
        NPS_RD6 = NPS_WL_6(et_ls) 
        NPS_DID7 = NPS_WL_7(et_ls)
        
        NPS = [NPS_RD1.beta['post'], NPS_RD2.beta['post'], NPS_RD3.beta['post'], NPS_RD4.beta['post'], NPS_RD5.beta['post'],
                 NPS_RD6.beta['post'], NPS_DID7.beta['post']]
        table.loc["3.NoPartner_NotMarried"] = NPS
        
        #NoPartner_NotMarriedmom_pvalue 
        NPS_pv = [npswl1['p>t']['post'], npswl2['p>t']['post'],npswl3['p>t']['post'], npswl4['p>t']['post'],npswl5['p>t']['post'],
                npswl6['p>t']['post'], npswl7['p>t']['post']]

        table.loc["NoPartner_NotMarried_pvalue"]=NPS_pv
        
        #NoPartner_NotMarriedmom Standard Error
        NPS_se = [NPS_RD1.se['post'],NPS_RD2.se['post'], NPS_RD3.se['post'], NPS_RD4.se['post'], NPS_RD5.se['post'],
                 NPS_RD6.se['post'], NPS_DID7.se['post']]
        table.loc["NoPartner_NotMarried_SE"] = NPS_se
        
        
        #1-4 Working Last Week under Nodad_NotMarried
        NDS_RD1 = NDS_WL_1(et_ls)
        NDS_RD2 = NDS_WL_2(et_ls)
        NDS_RD3 = NDS_WL_3(et_ls)
        NDS_RD4 = NDS_WL_4(et_ls)
        NDS_RD5 = NDS_WL_5(et_ls)
        NDS_RD6 = NDS_WL_6(et_ls) 
        NDS_DID7 = NDS_WL_7(et_ls)
        
        NDS = [NDS_RD1.beta['post'], NDS_RD2.beta['post'], NDS_RD3.beta['post'], NDS_RD4.beta['post'], NDS_RD5.beta['post'],
                 NDS_RD6.beta['post'], NDS_DID7.beta['post']]
        table.loc["4.NoDad_NotMarried"] = NDS
        
        #NoDad_NotMarried_pvalue 
        NDS_pv = [ndswl1['p>t']['post'], ndswl2['p>t']['post'],ndswl3['p>t']['post'], ndswl4['p>t']['post'],ndswl5['p>t']['post'],
                ndswl6['p>t']['post'], ndswl7['p>t']['post']]

        table.loc["NoDad_NotMarried_pvalue"]=NDS_pv
        
        #NoDad_NotMarried Standard Error
        NDS_se = [NDS_RD1.se['post'],NDS_RD2.se['post'], NDS_RD3.se['post'], NDS_RD4.se['post'], NDS_RD5.se['post'],
                 NDS_RD6.se['post'], NDS_DID7.se['post']]
        table.loc["NoDad_NotMarried_SE"] = NDS_se
        
        #1-5 Working Last Week under Divorced_NoPartner
        NPDIV_RD1 = NPDIV_WL_1(et_ls)
        NPDIV_RD2 = NPDIV_WL_2(et_ls)
        NPDIV_RD3 = NPDIV_WL_3(et_ls)
        NPDIV_RD4 = NPDIV_WL_4(et_ls)
        NPDIV_RD5 = NPDIV_WL_5(et_ls)
        NPDIV_RD6 = NPDIV_WL_6(et_ls) 
        NPDIV_DID7 = NPDIV_WL_7(et_ls)
        
        NPDIV = [NPDIV_RD1.beta['post'], NPDIV_RD2.beta['post'], NPDIV_RD3.beta['post'], NPDIV_RD4.beta['post'], NPDIV_RD5.beta['post'],
                 NPDIV_RD6.beta['post'], NPDIV_DID7.beta['post']]
        table.loc["5.Divorced_NoPartner"] = NPDIV
        
        #Divorced_NoPartnermom_pvalue 
        NPDIV_pv = [npdivwl1['p>t']['post'], npdivwl2['p>t']['post'],npdivwl3['p>t']['post'], npdivwl4['p>t']['post'],npdivwl5['p>t']['post'],
                npdivwl6['p>t']['post'], npdivwl7['p>t']['post']]

        table.loc["Divorced_NoPartner_pvalue"]=NPDIV_pv
        
        #Divorced_NoPartnermom Standard Error
        NPDIV_se = [NPDIV_RD1.se['post'],NPDIV_RD2.se['post'], NPDIV_RD3.se['post'], NPDIV_RD4.se['post'], NPDIV_RD5.se['post'],
                 NPDIV_RD6.se['post'], NPDIV_DID7.se['post']]
        table.loc["Divorced_NoPartner_SE"] = NPDIV_se




##=============================================================================    
        #Observations
        table=table.astype(float).round(4)
        obs =[NPND_RD1.N, NPND_RD2.N, NPND_RD3.N, NPND_RD4.N, NPND_RD5.N, NPND_RD6.N, NPND_DID7.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48]
        table.loc["Number of months"] = months

        
        return(table)


# In[ ]:


#tableET2(et_ls)


# In[ ]:


##================Create Table E2
def tableET3(et_ls):
        table = pd.DataFrame({'RDD_9m(1)': [], 'RDD_6m(2)': [], 'RDD_4m(3)': [],
                          'RDD_3m(4)': [], 'RDD_2m(5)': [], 'RDD_2m(6)': [],
                          'DID(7)': []})
        result = ( '1.NoPartner_NoDad', 'NoPartner_NoDad_pvalue', 'NoPartner_NoDad_SE', 
                  '2.Singlemom', 'Singlemom_pvalue','Singlemom_SE',
                  '3.NoPartner_NotMarried', 'NoPartner_NotMarried_pvalue', 'NoPartner_NotMarried_SE',
                  '4.NoDad_NotMarried', 'NoDad_NotMarried_pvalue', 'NoDad_NotMarried_SE',
                  '5.Divorced_NoPartner','Divorced_NoPartner_pvalue','Divorced_NoPartner_SE',
                  'Observations','Linear trend in m','Quadric trend in m','Calendar month of birth dummies',
                  'Controls','Number of months')
        table['Marital Status_Employed'] = result
        table = table.set_index('Marital Status_Employed')

##=================Employed 
        #1. Employed under No Part & No Dad
        NPND_RD_EP1 = NPND_EP_1(et_ls)
        NPND_RD_EP2 = NPND_EP_2(et_ls)
        NPND_RD_EP3 = NPND_EP_3(et_ls)
        NPND_RD_EP4 = NPND_EP_4(et_ls)
        NPND_RD_EP5 = NPND_EP_5(et_ls)
        NPND_RD_EP6 = NPND_EP_6(et_ls) 
        NPND_DID_EP7 = NPND_EP_7(et_ls)
        
        ep_npnd = [NPND_RD_EP1.beta['post'], NPND_RD_EP2.beta['post'], NPND_RD_EP3.beta['post'], NPND_RD_EP4.beta['post'], NPND_RD_EP5.beta['post'],
                 NPND_RD_EP6.beta['post'], NPND_DID_EP7.beta['post']]
        table.loc["1.NoPartner_NoDad"] = ep_npnd
        
        #EP_NoPartner_NoDad_pvalue 
        ep_npnd_pv = [npndep1['p>t']['post'], npndep2['p>t']['post'],npndep3['p>t']['post'], npndep4['p>t']['post'],npndep5['p>t']['post'],
                npndep6['p>t']['post'], npndep7['p>t']['post']]

        table.loc["NoPartner_NoDad_pvalue"]=ep_npnd_pv
        
        #EP_NoPartner_NoDad Standard Error
        ep_npnd_se = [NPND_RD_EP1.se['post'],NPND_RD_EP2.se['post'], NPND_RD_EP3.se['post'], NPND_RD_EP4.se['post'], NPND_RD_EP5.se['post'],
                 NPND_RD_EP6.se['post'], NPND_DID_EP7.se['post']]
        table.loc["NoPartner_NoDad_SE"] = ep_npnd_se

        
        #1-2 Employed under EP_Singlemom
        SG_RD_EP1 = SG_EP_1(et_ls)
        SG_RD_EP2 = SG_EP_2(et_ls)
        SG_RD_EP3 = SG_EP_3(et_ls)
        SG_RD_EP4 = SG_EP_4(et_ls)
        SG_RD_EP5 = SG_EP_5(et_ls)
        SG_RD_EP6 = SG_EP_6(et_ls) 
        SG_DID_EP7 = SG_EP_7(et_ls)
        
        
        EP_SG = [SG_RD_EP1.beta['post'], SG_RD_EP2.beta['post'], SG_RD_EP3.beta['post'], SG_RD_EP4.beta['post'], SG_RD_EP5.beta['post'],
                 SG_RD_EP6.beta['post'], SG_DID_EP7.beta['post']]
        table.loc["2.Singlemom"] = EP_SG
        
        #EP_Singlemom_pvalue 
        EP_SG_pv = [sgep1['p>t']['post'], sgep2['p>t']['post'],sgep3['p>t']['post'], sgep4['p>t']['post'],sgep5['p>t']['post'],
                sgep6['p>t']['post'], sgep7['p>t']['post']]

        table.loc["Singlemom_pvalue"]=EP_SG_pv
        
        #EP_Singlemom StandaRD_EP Error
        EP_SG_se = [SG_RD_EP1.se['post'],SG_RD_EP2.se['post'], SG_RD_EP3.se['post'], SG_RD_EP4.se['post'], SG_RD_EP5.se['post'],
                 SG_RD_EP6.se['post'], SG_DID_EP7.se['post']]
        table.loc["Singlemom_SE"] = EP_SG_se

        
        #1-3 Employed under EP_NoPartner_NotMarried
        NPS_RD_EP1 = NPS_EP_1(et_ls)
        NPS_RD_EP2 = NPS_EP_2(et_ls)
        NPS_RD_EP3 = NPS_EP_3(et_ls)
        NPS_RD_EP4 = NPS_EP_4(et_ls)
        NPS_RD_EP5 = NPS_EP_5(et_ls)
        NPS_RD_EP6 = NPS_EP_6(et_ls) 
        NPS_DID_EP7 = NPS_EP_7(et_ls)
        
        EP_NPS = [NPS_RD_EP1.beta['post'], NPS_RD_EP2.beta['post'], NPS_RD_EP3.beta['post'], NPS_RD_EP4.beta['post'], NPS_RD_EP5.beta['post'],
                 NPS_RD_EP6.beta['post'], NPS_DID_EP7.beta['post']]
        table.loc["3.NoPartner_NotMarried"] = EP_NPS
        
        #EP_NoPartner_NotMarriedmom_pvalue 
        EP_NPS_pv = [npsep1['p>t']['post'], npsep2['p>t']['post'],npsep3['p>t']['post'], npsep4['p>t']['post'],npsep5['p>t']['post'],
                npsep6['p>t']['post'], npsep7['p>t']['post']]

        table.loc["NoPartner_NotMarried_pvalue"]= EP_NPS_pv
        
        #EP_NoPartner_NotMarriedmom Standard Error
        EP_NPS_se = [NPS_RD_EP1.se['post'],NPS_RD_EP2.se['post'], NPS_RD_EP3.se['post'], NPS_RD_EP4.se['post'], NPS_RD_EP5.se['post'],
                 NPS_RD_EP6.se['post'], NPS_DID_EP7.se['post']]
        table.loc["NoPartner_NotMarried_SE"] = EP_NPS_se

        
       #1-4 Employed Last Week under EP_NoDad_NotMarried
        NDS_RD_EP1 = NDS_EP_1(et_ls)
        NDS_RD_EP2 = NDS_EP_2(et_ls)
        NDS_RD_EP3 = NDS_EP_3(et_ls)
        NDS_RD_EP4 = NDS_EP_4(et_ls)
        NDS_RD_EP5 = NDS_EP_5(et_ls)
        NDS_RD_EP6 = NDS_EP_6(et_ls) 
        NDS_DID_EP7 = NDS_EP_7(et_ls)
        
        EP_NDS = [NDS_RD_EP1.beta['post'], NDS_RD_EP2.beta['post'], NDS_RD_EP3.beta['post'], NDS_RD_EP4.beta['post'], NDS_RD_EP5.beta['post'],
                 NDS_RD_EP6.beta['post'], NDS_DID_EP7.beta['post']]
        table.loc["4.NoDad_NotMarried"] = EP_NDS
        
        #EP_NoDad_NotMarried_pvalue 
        EP_NDS_pv = [ndsep1['p>t']['post'], ndsep2['p>t']['post'],ndsep3['p>t']['post'], ndsep4['p>t']['post'],ndsep5['p>t']['post'],
                ndsep6['p>t']['post'], ndsep7['p>t']['post']]

        table.loc["NoDad_NotMarried_pvalue"]=EP_NDS_pv
        
        #EP_NoDad_NotMarried Standard Error
        EP_NDS_se = [NDS_RD_EP1.se['post'],NDS_RD_EP2.se['post'], NDS_RD_EP3.se['post'], NDS_RD_EP4.se['post'], NDS_RD_EP5.se['post'],
                 NDS_RD_EP6.se['post'], NDS_DID_EP7.se['post']]
        table.loc["NoDad_NotMarried_SE"] = EP_NDS_se

        #1-5 Employed under EP_Divorced_NoPartner
        NPDIV_RD_EP1 = NPDIV_EP_1(et_ls)
        NPDIV_RD_EP2 = NPDIV_EP_2(et_ls)
        NPDIV_RD_EP3 = NPDIV_EP_3(et_ls)
        NPDIV_RD_EP4 = NPDIV_EP_4(et_ls)
        NPDIV_RD_EP5 = NPDIV_EP_5(et_ls)
        NPDIV_RD_EP6 = NPDIV_EP_6(et_ls) 
        NPDIV_DID_EP7 = NPDIV_EP_7(et_ls)
        
        EP_NPDIV = [NPDIV_RD_EP1.beta['post'], NPDIV_RD_EP2.beta['post'], NPDIV_RD_EP3.beta['post'], NPDIV_RD_EP4.beta['post'], NPDIV_RD_EP5.beta['post'],
                 NPDIV_RD_EP6.beta['post'], NPDIV_DID_EP7.beta['post']]
        table.loc["5.Divorced_NoPartner"] = EP_NPDIV
        
        #EP_Divorced_NoPartnermom_pvalue 
        EP_NPDIV_pv = [npdivep1['p>t']['post'], npdivep2['p>t']['post'],npdivep3['p>t']['post'], npdivep4['p>t']['post'],npdivep5['p>t']['post'],
                npdivep6['p>t']['post'], npdivep7['p>t']['post']]

        table.loc["Divorced_NoPartner_pvalue"]=EP_NPDIV_pv
        
        #EP_Divorced_NoPartnermom Standard Error
        EP_NPDIV_se = [NPDIV_RD_EP1.se['post'],NPDIV_RD_EP2.se['post'], NPDIV_RD_EP3.se['post'], NPDIV_RD_EP4.se['post'], NPDIV_RD_EP5.se['post'],
                 NPDIV_RD_EP6.se['post'], NPDIV_DID_EP7.se['post']]
        table.loc["Divorced_NoPartner_SE"] = EP_NPDIV_se





##=============================================================================    
        #Observations
        table=table.astype(float).round(4)
        obs =[NPND_RD_EP1.N, NPND_RD_EP2.N, NPND_RD_EP3.N, NPND_RD_EP4.N, NPND_RD_EP5.N, NPND_RD_EP6.N, NPND_DID_EP7.N]
        table.loc["Observations"] = obs        
               
        #Linar trend in m
        linear = ["Y","Y","Y","N","N","N","Y"]
        table.loc["Linear trend in m"] = linear
        
        #Quadric trend in m
        quadric = ["Y","N","N","N","N","N","Y"]
        table.loc["Quadric trend in m"] = quadric
      
        #Calendar month of birth dummies
        dummies = ["N","N","N","N","N","N","Y"]
        table.loc["Calendar month of birth dummies"] = dummies

        #Controls
        controls = ["Y","Y","Y","Y","N","Y","Y"]
        table.loc["Controls"] = controls
      
        
        #Number of months
        months = [18,12,8,6,4,4,48]
        table.loc["Number of months"] = months

        
        return(table)


# In[ ]:


#tableET3(et_ls)


# In[ ]:




