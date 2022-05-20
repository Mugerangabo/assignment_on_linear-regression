#!/usr/bin/env python
# coding: utf-8

# ## I.     IMPORTING LIBRARIES 

# ### I.1     Installation of scikit-learn

# In[56]:


get_ipython().system('pip install scikit-learn')


# ### I.2     Importing the libraries and tools needed for this project

# In[57]:


#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')

#Importing tools 

from sklearn.metrics import r2_score
from scipy.optimize import curve_fit


# ## II. DATA VISUALIZATION 

# ### II.1  Importing the historical internet traffic dateset and visualization of the first few rows using Pandas

# In[58]:


df_hist = pd.read_csv("internet_traffic_hist(1).csv")
df_hist.head(11)


# ### II.2 Using matplotlib functions to get an idea on the trend of  the internet traffic volume

# In[59]:


plt.figure(figsize = (20,10))

x = df_hist.year
y = df_hist.traffic

plt.plot(x,y, label = '', linewidth = 5)
plt.plot(x, y, '*k', markersize = 15, label = '')
plt.axis([x.iloc[0]-1, x.iloc[-1]+1, y.iloc[0]*-0.1, y.iloc[-1]*1.1])

plt.xlabel('year')
plt.ylabel('Fixed Internet Traffic Volume')
plt.rcParams.update ({'font.size' : 26})


# ## III. SOLVING THE FIRST ODER POLYNOMIAL USING SIMPLE LINEAR REGRESSION

# ### III.1 Overlay a simple linear regression model over the Internet historical data

# In[60]:


plt.figure(figsize = (20,10))

order = 1

x = df_hist.year
y = df_hist.traffic

m, b = np.polyfit(x,y,order)

plt.plot(x, y, label = 'Historical Internet Traffic', linewidth = 7)
plt.plot(x, y,'*k', markersize = 15, label ='')
plt.plot(x, m*x + b, '-', label = 'Simple Linear Regression Line', linewidth = 6)

print ('The slope of line is {}.'.format(m))
print ('The y intercept is {}.'.format(b))
print ('The best fit simple linear regression line is {}x + {}.'.format(m,b))


plt.axis([x.iloc[0]-1, x.iloc[-1]+1, y.iloc[0]*-0.1, y.iloc[-1]*1.1])


plt.xlabel('Year')
plt.ylabel('Fixed Internet Traffic Volume')
plt.legend(loc = 'upper left')


plt.rcParams.update({'font.size': 26})
plt.show()


# ## IV. SOLVING THE HIGHER ODER POLYNOMIALS USING LINEAR REGRESSION

# In[61]:


models = []      
errors_hist = []  
mse_hist = []   

for order in range(1,4):
   
    p = (np.poly1d(np.polyfit(x, y, order)))
    models.append(p)
    
plt.figure(figsize = (20,10))


for model in models[0:3]:
    plt.plot(x, model(x), label = 'order ' + str(len(model)), linewidth = 7)

plt.plot(x, y, '*k', markersize = 14, label = 'Historical Internet Traffic', linewidth = 7)
plt.legend(loc = 'upper left')


plt.xlabel('Year')
plt.ylabel('Fixed Internet Traffic Volume')

plt.show()


# ## V. ERRORS CALCULATION 

# In[62]:


models = []       
errors_hist = []  
mse_hist = []    


for order in range(1,4):
    
    p = (np.poly1d(np.polyfit(x, y, order)))
    models.append(p)
    
    e = np.abs(y-p(x))       
    mse = np.sum(e**2)/len(df_hist) # mse
    
    errors_hist.append(e)   
    mse_hist.append(mse) 


# In[63]:


x = df_hist.year
width = 0.2   

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

p1 = ax.bar( x, errors_hist[0], width, color = 'b', label = 'Abs. error order 1 fit')
p2 = ax.bar( x + width, errors_hist[1], width, color = 'r', label = 'Abs. error order 2 fit')
p3 = ax.bar( x + 2*width, errors_hist[2], width, color = 'y', label = 'Abs. error order 3 fit')


ax.set_xticks(x+2*width)
ax.set_xticklabels(x)
plt.legend(loc = 'upper left', fontsize =16)
plt.show()


fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

x = np.array([1,2,3])
width = .6   

p1 = ax.bar( x[0], mse_hist[0], width, color = 'b', label = 'pred. error order 1 fit')
p2 = ax.bar( x[1], mse_hist[1], width, color = 'r', label = 'pred. error order 2 fit')
p3 = ax.bar( x[2], mse_hist[2], width, color = 'y', label = 'pred. error order 3 fit')

ax.set_xticks(x+width/2)
ax.set_xticklabels(['Poly. order 1', 'Poly. order 2', 'Poly. order 3'], rotation=90)
plt.show()


# In[64]:


order = 3

x = df_hist.year.values      
y = df_hist.traffic.values

p_array = np.polyfit(x,y,order)

print(type(p_array), p_array)


p = np.poly1d(p_array)

print(type(p), p)


print('The value of the polynomial for x = 2020 is : {} '.format(p(2020)))


e = np.abs(y-p(x))
mse = np.sum(e**2)/len(x)

print('The estimated polynomial parameters are: {}'.format(p))
print('The errors for each value of x, given the estimated polynomial parameters are: \n {}'.format(e))
print('The MSE is :{}'.format(mse))


# ## VI. EXPONENTIAL GROUWTH USING NON-LINEAR REGRESSION 

# In[65]:


def my_exp_func(x, a, b):
    return a * (b ** x) 

x = np.arange(2016-2005)  
y = df_hist.traffic.values 


p, cov = curve_fit(my_exp_func, x, y)
e = np.abs(y - my_exp_func(x, *p))
mse = np.sum(e**2)/len(df_hist)

print('The estimated exponential parameters are: {}'.format(p))
print('The errors for each value of x, given the estimated exponential parameters are: \n {}'.format(e))
print('The MSE is :{}'.format(mse))

models.append(p)

errors_hist.append(e) 
mse_hist.append(mse)


# ## VII.  COMPARING MODELS

# In[66]:


plt.figure(figsize = (20,10))


for model in models[0:3]:
    
    x = df_hist.year.values      
    y = df_hist.traffic.values   
    plt.plot(x, model(x), label = 'order ' + str(len(model)), linewidth = 7)

x = np.arange(2016-2005)   
plt.plot(df_hist.year.values, my_exp_func(x, *models[-1]), label = 'Exp. non-linear regression', linewidth = 7)

plt.plot(df_hist.year, df_hist.traffic, '*k', markersize = 14, label='Historical Internet Traffic')
plt.legend(loc = 'upper left')

plt.xlabel('Year')
plt.ylabel('Fixed Internet Traffic Volume')
plt.show()


# In[67]:


x = df_hist.year
width = 0.2  

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

p1 = ax.bar( x, errors_hist[0], width, color = 'b', label = 'Abs. error order 1 fit')
p2 = ax.bar( x + width, errors_hist[1], width, color = 'r', label = 'Abs. error order 2 fit')
p3 = ax.bar( x + 2*width, errors_hist[2], width, color = 'y', label = 'Abs. error order 3 fit')
p4 = ax.bar( x + 3*width, errors_hist[3], width, color = 'g', label = 'Abs. exponential fit')


ax.set_xticks(x+2*width)
ax.set_xticklabels(x)
plt.legend(loc = 'upper left', fontsize =16)
plt.show()

fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)

x = np.array([0,1,2,3])
width = .6   

p1 = ax.bar( x[0], mse_hist[0], width, color = 'b', label = 'pred. error order 1 fit')
p2 = ax.bar( x[1], mse_hist[1], width, color = 'r', label = 'pred. error order 2 fit')
p3 = ax.bar( x[2], mse_hist[2], width, color = 'y', label = 'pred. error order 3 fit')
p4 = ax.bar( x[3], mse_hist[3], width, color = 'g', label = 'pred. exponential fit')

ax.set_xticks(x+width/2)
ax.set_xticklabels(['Poly. order 1', 'Poly. order 2', 'Poly. order 3', 'Exp. model' ], rotation=90)
plt.show()


# ## VIII. PREDICT THE FUTURE VALUES USING THE REGRESSION MODEL 

# ### VIII.1 Load and visualize projected Internet traffic volume

# In[68]:


df_proj = pd.read_csv("internet_traffic_proj.csv") 
df_proj.head()


# ### VIII.2 Compare linear and nonlinear model prediction errors

# #### VIII.2.1 Visualinzing data

# In[69]:


df = pd.concat([df_hist, df_proj]).reset_index()
df.drop('index', axis=1, inplace=True)
df = df.drop_duplicates() 
df.head(20)


# #### VIII.2.1 Visualinzing the graph

# In[70]:


plt.figure(figsize = (20,10))

errors_all = []
mse_all = []

for model in models[0:3]:
    
    x = df.year.values      
    y = df.traffic.values   
    plt.plot(x, model(x), label = 'order ' + str(len(model)), linewidth = 7)
    
    pred_y = model(x)
    e = np.abs(y - pred_y)
    errors_all.append(e)   
    mse_all.append(np.sum(e**2)/len(df)) 
    
x = np.arange(2021-2005)  
pred_y = my_exp_func(x, *models[-1])
plt.plot(df.year.values, pred_y, label = 'Exp. non-linear regression', linewidth = 7)

e = np.abs(y - pred_y)
errors_all.append(e)  
mse_all.append(np.sum(e**2)/len(df)) 

plt.plot(df.year, df.traffic, '*k', markersize = 14, label='Projected Internet Traffic')
plt.legend(loc = 'upper left')

plt.xlabel('Year')
plt.ylabel('Fixed Internet Traffic Volume')
plt.axis([2004,2021, -300, 3500])
plt.show()

