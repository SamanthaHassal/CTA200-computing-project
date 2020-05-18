#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt


# In[2]:


#list of sigmas
sigma = [0.5, 0.5]

#make an array with a given size
arr_make = lambda s: np.linspace(-1, 1, s)

def Gauss_2d(size1, size2):
    '''builds arrays of size1 and size2 and uses them to generate a 2D
    Gaussian distribution
    
    Parameters
    ----------
    size1 : int
        number of samples in the x direction.
    size2 : int
        number of samples in the y direction.
    
    Returns
    -------
    err_code : numpy.ndarray
        array representing the 2D Gaussian distribution
    '''
    #make x and y arrays with sizes size1 and size 2
    x = arr_make(size1)
    y = arr_make(size2)
    #generate a meshgrid
    x, y = np.meshgrid(x, y)
    #generate and plot 2D gaussian
    z = (1/(2*np.pi*sigma[0]*sigma[1]) * 
         np.exp(-(x**2/(2*sigma[0]**2) + 
                  y**2/(2*sigma[1]**2))))
    return z

def Gauss_2d_messy(size1, size2):
    '''builds noisy arrays of size1 and size2 and uses them to generate a 2D
    Gaussian distribution
    
    Parameters
    ----------
    size1 : int
        number of samples in the x direction.
    size2 : int
        number of samples in the y direction.
    
    Returns
    -------
    err_code : numpy.ndarray
        array representing the 2D Gaussian distribution with noise
    '''
    #seed the random number generator to make the results reproducible
    np.random.seed(seed=10)
    #make x and y arrays with sizes size1 and size 2
    x = arr_make(size1) + np.random.normal(loc=0, scale=0.2, size=size1)
    y = arr_make(size2) + np.random.normal(loc=0, scale=0.2, size=size2)
    #generate a meshgrid
    x, y = np.meshgrid(x, y)
    #generate and plot 2D gaussian
    z = (1/(2*np.pi*sigma[0]*sigma[1]) * 
         np.exp(-(x**2/(2*sigma[0]**2) + 
                  y**2/(2*sigma[1]**2))))
    return z


# In[3]:


#make an image of the 2D Gaussian
T = Gauss_2d(50, 100)
#make an image of the 2D Gaussian using the messy function
U = Gauss_2d_messy(50,100)

#create uniform distribution samples
UNIF = [np.random.uniform(low=-1., high=1.0, size=(2,5)), 
        np.random.uniform(low=-1., high=1.0, size=(2,5)), 
        np.random.uniform(low=-1., high=1.0, size=(2,5))]

#graph settings
fig = plt.figure(figsize=(8,6))

#add axes to plot T on the main axis
ax1 = fig.add_axes([0.1, 0.1, 1, 1])
A = ax1.imshow(T, cmap='viridis', extent=[-1,1,-1,1])

#scatter plot of uniform distribution on the main axis
ax1.scatter(UNIF[0][0,:], UNIF[0][1,:], c='red', marker='$A$')
ax1.scatter(UNIF[2][0,:], UNIF[0][1,:], c='yellow', marker='$C$')
fig.colorbar(A)

#add axes to plot the new image in the upper left hand corner
ax2 = fig.add_axes([1/6,0.75, 1/3, 1/3])
B = ax2.imshow(U, cmap='jet', extent=[-1,1,-1,1])

#scatter plot of uniform distribution on the inset axis
ax2.scatter(UNIF[0][0,:], UNIF[0][1,:], c='blue', marker='$B$')
ax2.scatter(UNIF[2][0,:], UNIF[0][1,:], c='yellow', marker='$C$')
fig.colorbar(B)

#create arrow pointing from centre of the inset to the centre of the main axis
ax2.arrow(0,0,1.45,-1.75,fc='white',ec='white',clip_on=False, width=0.01)

#save as PDF file 
plt.savefig('SamanthaHassal-graphing-demo.pdf')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




