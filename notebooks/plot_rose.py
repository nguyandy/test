
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from cdftools.plot_tools import OceanPlots
import numpy as np


# In[2]:

N = 500
ws = np.random.random(N)*9.5
wd = np.random.random(N)*360

ws2 = np.random.random(N)*12
wd2 = np.random.random(N)*360
ocean_plots = OceanPlots()
fig, ax = ocean_plots.plot_rose(ws, wd,
                             nsector=16,
                             title='Wind Rose',
                             legend_title='Wind Speed (m/s)')


# In[4]:

N = 500
ws = np.random.random(N)*9.5
wd = np.random.random(N)*360

N = 200
ws2 = np.random.random(N)*14
wd2 = np.random.random(N)*360
ocean_plots = OceanPlots()
fig, ax, ax2 = ocean_plots.plot_rose_comparison(ws, wd, ws2, wd2,
                                            nsector=16,
                                            title1='Wind Rose1', title2='Wind Rose2',
                                            legend_title='Wind Speed (m/s)')


# In[ ]:



