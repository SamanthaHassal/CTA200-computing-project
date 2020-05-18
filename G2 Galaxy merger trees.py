#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import scipy
import matplotlib.pyplot as plt 
import pandas as pd


# In[3]:


#read G2_merger_tree working file.csv using pandas
dt=pd.read_csv('G2_merger_tree working file.csv')
# store galaxy id, redshift, virial mass, stellar mass, and stellar formation rate in variables
Z, g_id, sfr, vir_mass, stel_mass = dt['redshift'], dt['galaxyID'], dt['sfr'], dt['mvir'], dt['stellarMass']
# create arrays to be graphed and store them in a dictionary
items = {'redshift':Z, 
         'stellar formation rate per redshift':sfr/Z, 
         'virial mass per redshift':vir_mass/Z, 
         'stellar mass per redshift':stel_mass/Z}


# In[4]:


#plot data in above dictionary
for key in items:
    plt.figure(figsize=(10,8))
    plt.scatter(items[key], g_id, c='blue', marker='>')
    plt.ylabel('Galaxy ID')
    plt.xlabel(key)
    plt.savefig(key+'.pdf')
    plt.show()
    


# In[5]:


#build scatter plot of redshift vs stellar mass
#the size of the figures represents virial mass
#the colour represents the stellar formation rate
plt.figure(figsize=(10,8))
plt.scatter(Z, stel_mass,  
            marker='*',
            s=vir_mass/10,
            c=sfr,
           cmap='plasma',
           alpha = 0.75, zorder=2)
#draw lines connecting halos on the graph based on changes in virial mass
for i in range (0, len(stel_mass)-1):
    if vir_mass[i] > vir_mass[i+1]:
        plt.plot(Z[i:i+2], stel_mass[i:i+2], c='violet',
                    zorder=1, alpha=0.4)

plt.title('Scatter plot of stellar mass vs redshift')
plt.xlabel('redshift')
plt.ylabel('stellar mass')
plt.savefig('first merger tree.pdf')
plt.show()


# In[ ]:





# In[ ]:




