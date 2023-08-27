#!/usr/bin/env python
# coding: utf-8

# # Ford GoBike System Data

# 
# ### Dataset Description 
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > **Tip**: information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area.
# ● Note that this dataset will require some data wrangling in order to make it tidy for analysis. There are multiple cities covered by the linked system, and multiple data files will need to be joined together if a full year’s coverage is desired.
# Here are the datasets in CSV format. You can fit your model using the train data, then predict using the test data and submit your predictions in the format of the sample submission.
# Your goal is to predict the rotor bearing temperature, which is the Target column in the datasets. [here](https://www.google.com/url?q=https://video.udacity-data.com/topher/2020/October/5f91cf38_201902-fordgobike-tripdata/201902-fordgobike-tripdata.csv&sa=D&source=editors&ust=1669750197727856&usg=AOvVaw0RJVqpWyfu7RoKaPH1gynL). Files

# ### What is/are the main feature(s) of interest in your dataset?
# 
# I'm most interested in figuring out .

# what features are best for predicting most trips taken in terms of time of day, day of the week, or month of the year?

# How long does the average trip take?

# Does the above depend on if a user is a subscriber or customer?.

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

from pathlib import Path
from warnings import simplefilter

import requests

simplefilter("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# load in the dataset into a pandas dataframe
df_go_bike = pd.read_csv("new data/Go_Bike.csv")


# ### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?
# 

# In[3]:


def bar_plot (col_name):   
    # Return the Series having unique values
    x = df_go_bike[col_name].unique()

    # Return the Series having frequency count of each unique value
    y = df_go_bike[col_name].value_counts(sort=False)
    
    plt.subplots(figsize=(18,5))
    
    plt.bar(x, y)

    # Labeling the axes
    plt.xlabel(col_name)
    plt.ylabel('count')
    
    # Dsiplay the plot
    plt.show()
    #return df_go_bike.col_name.unique() , df_go_bike.col_name.value_counts()


# In[4]:


def pie_chart (col_name):     
    sorted_counts = df_go_bike[col_name].value_counts()
    plt.figure(figsize=(10,5))
    
    plt.pie(sorted_counts,
            labels = sorted_counts.index, 
            autopct = "%1.1f%%", 
            startangle = 150,
            counterclock = False, 
            wedgeprops = {'width' : 0.4},
            shadow = True);
    plt.title ("Percentage of "+ col_name)
    plt.axis('square')


# ## Distribution of duration trip
# 

# In[5]:


# duration_sec has a long-tailed distribution the duration_sec distribution looks roughly bimodal, with one peak between 250 and 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.
bar_plot('duration_sec')


# In[6]:


# duration_minu has a long-tailed distribution the duration_minu distribution looks roughly bimodal, with one peak between more 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.
bar_plot('duration_minu')


# In[7]:


# duration_hr has a long-tailed distribution the duration_hr distribution looks roughly bimodal, with one peak between more 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.
bar_plot('duration_hr')


# ## Distribution of member_gender
# 
# in the figure , there are bar plot / pie_chart to showup and compare between in number of users whether male or female 

# In[8]:


bar_plot('member_gender')

pie_chart('member_gender')


# ## Distribution of user_type
# 
# in the figure , there are bar plot / pie_chart to showup and compare between in user_type whether customer or subscribe 

# In[9]:


bar_plot('user_type')

pie_chart('user_type')


# ## Distribution of member_birth_year
# 
# in the figure , there are bar plot / pie_chart to showup in member_birth_year 

# In[10]:


bar_plot('member_birth_year')


# 
# ### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?
# 
# When investigating the x, y, and z size variables, a number of outlier points were identified. Overall, these points can be characterized by an inconsistency between the recorded value of depth, and the value that would be derived from using x, y, and z. For safety, all of these points were removed from the dataset to move forwards.

# ## Bivariate Exploration
# 
# To start off with, I want to look at the pairwise correlations present between features in the data.

# In[11]:


def scatterplots(x,y):    
    # scatter plot of price vs. carat, with log transform on price axis

    plt.figure(figsize = [8, 6])
    plt.scatter(data = df_go_bike, x = x, y = y, alpha = 1/10)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


# ## member_birth_year  vs  duration_hr
# 
# Plotting member_birth_year linear relationship. For duration_hr above 0.5h , there appears to be a member_birth_year: based on the trend below 20h duration_hr , we might expect member_birth_year to take duration_hr between 0 and 5

# In[12]:


scatterplots('member_birth_year','duration_hr')


# In[13]:


scatterplots('member_gender' , 'member_birth_year')


# In[14]:


scatterplots('bike_id' , 'member_birth_year')


# In[15]:


scatterplots('start_station_id' , 'end_station_id')


# In[16]:


def subplots(x,y):
    # since there's only three subplots to create, using the full data should be fine.
    plt.figure(figsize = [35 , 35])

    # subplot 1: color vs cut
    ax = plt.subplot(3, 1, 1)
    sns.countplot(data = df_go_bike, x = x, hue = y, palette = 'Blues')
    ax.legend(ncol = 2) # re-arrange legend to reduce overlapping


# ## member_birth_year  vs  member_gender
# 
# Plotting member_birth_year linear relationship. For member_gender , there appears to be a member_birth_year: based on the trend member_gender , we might expect member_birth_year to take member_gender.

# In[17]:


subplots('member_gender','member_birth_year')


# In[18]:


def bivar (x1 , y1 , x2 , y2):
    plt.figure(figsize = [18, 6])

    # PLOT ON LEFT
    plt.subplot(1, 2, 1)
    sns.regplot(data = df_go_bike , x = x1 , y = y1 , x_jitter=0.04, scatter_kws={'alpha':1/10}, fit_reg=False)
    plt.xlabel(x1)
    plt.ylabel(y1);

    # PLOT ON RIGHT
    plt.subplot(1, 2, 2)
    plt.hist2d(data = df_go_bike , x = x2 , y = y2)
    plt.colorbar()
    plt.xlabel(x2)
    plt.ylabel(y2);


# In[19]:


bivar ('member_birth_year','member_gender','member_birth_year','duration_hr')


# ## Multivariate Exploration
# any interesting relationships between the other features not the main feature(s) of interest

# #### Faceting for Multivariate Data
# you saw how FacetGrid could be used to subset your dataset across levels of a categorical variable, and then create one plot for each subset. Where the faceted plots demonstrated were univariate before, you can actually use any plot type, allowing you to facet bivariate plots to create a multivariate visualization.

# using hist() function to draw Histogram for each column and use figsize(,) parameter to show it obviously.
# looking at the distribution of the main variable of interest: using hist() function to draw Histogram for each column 

# In[20]:


# Use this, and more code cells, to explore your data. Don't forget to add
# Markdown cells to document your observations and findings.
df_go_bike.hist(figsize = (20,20));


# using plotting.scatter_matrix() function to draw Histogramsfor each column and scatter plotting between numerical columns 
# 

# In[21]:


pd.plotting.scatter_matrix(df_go_bike, figsize = (30,30));


# The faceted box plot suggests a slight interaction between the two categorical variables, where, in level B of "member_gender", the level of "user_type" seems to be have a larger effect on the value of "bike_id", compared to the trend within "member_gender" level A.

# In[22]:


g = sns.FacetGrid(data = df_go_bike, col = 'member_gender', size = 6)
g.map(sns.boxplot, 'user_type', 'bike_id')


# Setting margin_titles = True means that instead of each facet being labeled with the combination of row and column variable, labels are placed separately on the top and right margins of the facet grid. This is a boon, since the default plot titles are usually too long.

# FacetGrid also allows for faceting a variable not just by columns, but also by rows. We can set one categorical variable on each of the two facet axes for one additional method of depicting multivariate trends.
# Setting margin_titles = True means that instead of each facet being labeled with the combination of row and column variable, labels are placed separately on the top and right margins of the facet grid. This is a boon, since the default plot titles are usually too long.

# In[23]:


g = sns.FacetGrid(data = df_go_bike, col = 'user_type', row = 'member_birth_year', size = 2.5,margin_titles = True)
g.map(plt.scatter, 'bike_id', 'member_gender')


# The code for the 2-d bar chart doesn't actually change much. The actual heatmap call is still the same, only the aggregation of values changes. Instead of taking size after the groupby operation, we compute the mean across dataframe columns and isolate the column of interest.

# In[24]:


cat_means = df_go_bike.groupby(['member_gender', 'user_type']).mean()['duration_hr']
cat_means = cat_means.reset_index(name = 'duration_hr_avg')
cat_means = cat_means.pivot(index = 'user_type', columns = 'member_gender',
                            values = 'duration_hr_avg')
sns.heatmap(cat_means, annot = True, fmt = '.3f',
           cbar_kws = {'label' : 'mean(duration_hr)'})


# In[ ]:


get_ipython().system('jupyter nbconvert --to html --no-input Ford_Go_Bike_Part2.ipynb  ')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Ford_Go_Bike_Part2.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to latex Ford_Go_Bike_Part2.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to slides Ford_Go_Bike_Part2.ipynb')

