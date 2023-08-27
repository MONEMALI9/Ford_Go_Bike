#!/usr/bin/env python
# coding: utf-8

# # Project : Ford GoBike
# ## Table of Contents
# <ul>
#     <li><a href="#intro">Introduction</a></li>
#     <li><a href="#wrangling">Data Wrangling</a></li>
#     <li><a href="#eda">Exploratory Data Analysis</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# ### Dataset Description 
# 
# ### What features in the dataset do you think will help support your investigation into your feature(s) of interest?
# 
# > **Tip**: information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area.
# ● Note that this dataset will require some data wrangling in order to make it tidy for analysis. There are multiple cities covered by the linked system, and multiple data files will need to be joined together if a full year’s coverage is desired.
# Here are the datasets in CSV format. You can fit your model using the train data, then predict using the test data and submit your predictions in the format of the sample submission.
# Your goal is to predict the rotor bearing temperature, which is the Target column in the datasets. [here](https://www.google.com/url?q=https://video.udacity-data.com/topher/2020/October/5f91cf38_201902-fordgobike-tripdata/201902-fordgobike-tripdata.csv&sa=D&source=editors&ust=1669750197727856&usg=AOvVaw0RJVqpWyfu7RoKaPH1gynL). Files
# 
# ### What is the structure of your dataset?
# 
# #### Data Dictionary
# 
# 01 - duration_sec 
# 
# 02 - start_time 
# 
# 03 - end_time 
# 
# 04 - start_station_id 
# 
# 05 - start_station_name 
# 
# 06 - start_station_latitude 
#    
# 07 - start_station_longitude
# 
# 08 - end_station_id  
# 
# 09 - end_station_name 
# 
# 10 - end_station_latitude  
# 
# 11 - end_station_longitude  
# 
# 12 - bike_id	
# 
# 13 - user_type
# 
# 14 - member_birth_year
# 
# 15 - member_gender	
# 
# 16 - bike_share_for_all_trip
# 
# ### What is/are the main feature(s) of interest in your dataset?
# 
# I'm most interested in figuring out what features are best for predicting most trips taken in terms of time of day, day of the week, or month of the year? and How long does the average trip take? Does the above depend on if a user is a subscriber or customer?.

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


# <a id='wrangling'></a>
# ## Data Wrangling
# 
# > **Tip**: In this section of the report, you will load in the data, check for cleanliness, and then trim and clean your dataset for analysis. Make sure that you **document your data cleaning steps in mark-down cells precisely and justify your cleaning decisions.**
# 

# Data Wrangling which include :
#     1.Gathering Data
#     2.Assessing Data
#     3.cleaning Data

# ### Gathering Data

# In[2]:


# Load your data and print out a few lines. Perform operations to inspect data
#   types and look for instances of missing or possibly errant data.
df = pd.read_csv('data/201902-fordgobike-tripdata.csv')


# we aqucistion data from dataset like : csv file in our example

# In[3]:


df.head()


# we use head() or tail() function to display a sample of data 

# In[4]:


df.tail()


# ### Assessing Data 

# We assessing our data using some function like : shape , ndim , dtypes , size  , info() , nunique() , isnull()

# In[5]:


# return number of columns and number of row
df.shape


# the shape function get number of rows and number of columns in tuple

# In[6]:


#return number of dimensions of data
df.ndim


# size function show us the result of multiplication of number of rows and number of columns

# In[7]:


# return size of Dataset which is a multiplication of number of rows and number of columns
df.size


# dtypes show us data type of each column (features)

# In[8]:


#return types of each column
df.dtypes


# info() function show us number of non_null_value in each column and datatype
# 
# it has two features(no of non_null_value,datatype)

# In[9]:


#return number of non-null-value and datatype of each column
df.info()


# nunique() show us number of unique values in each column

# In[10]:


#return number of unique value
df.nunique()


# isnull() function show us boolean value for each element (each cell) it is null or not
# 
# if it(element) null return True
# 
# else  return False

# In[11]:


# return which value is nul or not for each element in DataSet 
df.isnull()


# In[12]:


# return which value is nul or not for each columns in DataSet 
df.isnull().any()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return True
# 
# else  return False

# In[13]:


#return number of columns has a null value
df.isnull().any().sum()


# In[14]:


#return number of null value for each column
df.isnull().sum()


# isnull().any() function show us boolean value for each column it is null or not
# 
# if column null return 1
# 
# else  return 0

# In[15]:


#return a number of cell has a null value
df.isnull().sum().sum()


# In[16]:


#return statistical descriptive of dataset for each column
df.describe()


# describe() function show us descriptive statistical value for each column
# 
# in 8 value such as: count element in each column
#     
#                     mean : getting average in each column
#                     
#                     std : standarded deviation
#                         
#                     min : minimum value in each column
#                         
#                     max : maximum value in each column
#                         
#                     50% : median of value for each column

# 
# ### Data Cleaning
# > **Tip**: Make sure that you keep your reader informed on the steps that you are taking in your investigation. Follow every code cell, or every set of related code cells, with a markdown cell to describe to the reader what was found in the preceding cell(s). Try to make it so that the reader can then understand what they will be seeing in the following cell(s).
#  

# 1.duplicated data
# 
# 2.missing value
# 
# 3.incorrect datatype

# In[17]:


df_clean = df.copy()


# from assessing no NULL value
# 
# we will check for missing value and incorrect datatype

# In[18]:


##df_clean.drop(['PatientId','AppointmentID'],axis=1,inplace = True)


# In[19]:


df_clean.drop(df_clean[ df_clean['member_gender'] == 'Other'].index,axis = 0,inplace = True)


# In[20]:


df_clean.head(1)


# In[21]:


df_clean.dtypes


# After discussing the structure of the data and any problems that need to be
# 
# cleaned, perform those cleaning steps in the second part of this section.

# #### duplicicated data

# In[22]:


#check for duplicated data
sum(df_clean.duplicated())


# In[23]:


# if we have duplicated data we remove it
df_clean.drop_duplicates(inplace = True)


# In[24]:


sum(df_clean.duplicated())


# #### Missing value

# ### checking

# In[25]:


df_clean.head(1)


# In[26]:


df_clean.info()


# In[27]:


df_clean.isnull().sum()


# In[28]:


df_clean.isnull().sum().sum()


# In[29]:


df_clean.dropna(inplace = True)


# In[30]:


df_clean.isnull().sum()


# In[31]:


df_clean.isnull().sum().sum()


# ##### incorrect datatype

# In[32]:


df_clean.dtypes


# In[33]:


df['start_time'] = pd.to_datetime(df['start_time'])


# In[34]:


df['end_time'] = pd.to_datetime(df['end_time'])


# In[35]:


df[['start_station_id','end_station_id','bike_id','member_birth_year']]= df[['start_station_id','end_station_id','bike_id','member_birth_year']].astype(str)


# ### checking

# In[36]:


df_clean.dtypes


# In[37]:


df_clean.head(1)


# ###### convert sec to minute

# In[38]:


duration_minu = df_clean.duration_sec/60


# In[39]:


duration_minu


# In[40]:


df_clean['duration_minu'] = duration_minu


# ###### checking

# In[41]:


df_clean.head(1)


# ###### convert minute to hours

# In[42]:


duration_hr = (df_clean.duration_sec/60)/60


# In[43]:


duration_hr


# In[44]:


df_clean['duration_hr'] = duration_hr


# In[45]:


df_clean.head(1)


# ##### convert hours to days

# In[46]:


duration_days = ((df_clean.duration_sec/60)/60)/24


# In[47]:


duration_days


# In[48]:


df_clean['duration_days'] = duration_days


# In[49]:


df_clean.head(1)


# ##### convert  days to weeks

# In[50]:


duration_weeks = (((df_clean.duration_sec/60)/60)/24)/7


# In[51]:


df_clean['duration_weeks'] = duration_weeks


# In[52]:


df_clean.head(1)


# ##### convert  weeks  to month

# In[53]:


duration_months = ((((df_clean.duration_sec/60)/60)/24)/7)/4


# In[54]:


df_clean['duration_months'] = duration_months


# In[55]:


df_clean.head(1)


# In[56]:


df_clean.to_csv("new data/Go_Bike.csv");


# #### DONE
# #### non_dublicated data....non_missing value....non incorrect datatype

# <a id='eda'></a>
# ## Exploratory Data Analysis
# 
# > **Tip**: Now that you've trimmed and cleaned your data, you're ready to move on to exploration. **Compute statistics** and **create visualizations** with the goal of addressing the research questions that you posed in the Introduction section. You should compute the relevant statistics throughout the analysis when an inference is made about the data. Note that at least two or more kinds of plots should be created as part of the exploration, and you must  compare and show trends in the varied visualizations. 
# 
# 
# 
# > **Tip**: - Investigate the stated question(s) from multiple angles. It is recommended that you be systematic with your approach. Look at one variable at a time, and then follow it up by looking at relationships between variables. You should explore at least three variables in relation to the primary question. This can be an exploratory relationship between three variables of interest, or looking at how two independent variables relate to a single dependent variable of interest. Lastly, you  should perform both single-variable (1d) and multiple-variable (2d) explorations.
# 
# 
# ### Research Question:
# 
# 1) When are most trips taken in terms of time of day, day of the week, or month of the year?
# 
# 2) How long does the average trip take?
# 
# 3) Does the above depend on if a user is a subscriber or customer?

# In[57]:


df_go_bike = pd.read_csv("new data/Go_Bike.csv")


# In[58]:


df_go_bike.info()


# In[59]:


df_go_bike.isnull().sum().sum()


# ## Univariate Exploration
# 
# I'll start by looking at the distribution of the main variable of interest:

# ### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?
# using hist() function to draw Histogram for each column and use figsize(,) parameter to show it obviously.

# In[60]:


df_go_bike.info()


# In[61]:


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


# In[62]:


def pie_chart (col_name):     
    sorted_counts = df_go_bike[col_name].value_counts()
    plt.figure(figsize=(10,5))
    
    plt.pie(sorted_counts, labels = sorted_counts.index, startangle = 150,
            counterclock = False, wedgeprops = {'width' : 0.4});
    plt.axis('square')


# In[63]:


df_go_bike.duration_sec.unique()


# In[64]:


df_go_bike.duration_sec.value_counts()


# ## Distribution of duration trip
# duration_sec has a long-tailed distribution the duration_sec distribution looks roughly bimodal, with one peak between 250 and 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.

# In[65]:


bar_plot('duration_sec')


# duration_sec has a long-tailed distribution the duration_sec distribution looks roughly bimodal, with one peak between 250 and 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.

# In[66]:


bar_plot('duration_minu')


# duration_minu has a long-tailed distribution the duration_minu distribution looks roughly bimodal, with one peak between more 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.

# In[67]:


bar_plot('duration_hr')


# duration_hr has a long-tailed distribution the duration_hr distribution looks roughly bimodal, with one peak between more 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.

# In[68]:


bar_plot('duration_days')


# duration_days has a long-tailed distribution the duration_days distribution looks roughly bimodal, with one peak between more 300. Interestingly, there's a steep jump in frequency right before 200, rather than a smooth ramp up.

# In[69]:


df_go_bike.member_gender.unique()


# In[70]:


df_go_bike.member_gender.value_counts()


# ## Distribution of member_gender
# 
# in the figure , there are bar plot / pie_chart to showup and compare between in number of users whether male or female 

# In[71]:


bar_plot('member_gender')


# in the figure , there are bar plot to showup and compare between in number of users whether male or female 

# In[72]:


pie_chart('member_gender')


# in the figure , there are pie chart to showup and compare between in number of users whether male or female 

# In[73]:


df_go_bike.user_type.unique()


# In[74]:


df_go_bike.user_type.value_counts()


# ## Distribution of user_type
# 
# in the figure , there are bar plot / pie_chart to showup and compare between in user_type whether customer or subscribe 

# In[75]:


bar_plot('user_type')


# in the figure , there are bar plot to showup and compare between in number of users whether customer or subscriber 

# In[76]:


pie_chart('user_type')


# in the figure , there are pie chart to showup and compare between in number of users whether customer or subscriber 

# In[77]:


df_go_bike.info()


# In[78]:


df_go_bike.member_birth_year.value_counts()


# In[79]:


df_go_bike.member_birth_year.unique()


# ## Distribution of member_birth_year
# 
# in the figure , there are bar plot / pie_chart to showup in member_birth_year 

# In[80]:


bar_plot('member_birth_year')


# member_birth_year has a long-tailed distribution the member_birth_year distribution looks roughly bimodal, with one peak between 8000 and 10000. Interestingly, there's a steep jump in frequency right before 6000, rather than a smooth ramp up.

# In[81]:


df_go_bike.info()


# ### Discuss the distribution(s) of your variable(s) of interest. Were there any unusual points? Did you need to perform any transformations?
# 
# 
# ### Of the features you investigated, were there any unusual distributions? Did you perform any operations on the data to tidy, adjust, or change the form of the data? If so, why did you do this?
# 
# When investigating the x, y, and z size variables, a number of outlier points were identified. Overall, these points can be characterized by an inconsistency between the recorded value of depth, and the value that would be derived from using x, y, and z. For safety, all of these points were removed from the dataset to move forwards.
# 
# ## Bivariate Exploration
# 
# To start off with, I want to look at the pairwise correlations present between features in the data.

# In[82]:


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


# In[83]:


bivar ('member_birth_year','member_gender','member_birth_year','duration_hr')


# In[84]:


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

# In[85]:


subplots('member_gender','member_birth_year')


# In[86]:


def scatterplots(x,y):    
    # scatter plot of price vs. carat, with log transform on price axis

    plt.figure(figsize = [8, 6])
    plt.scatter(data = df_go_bike, x = x, y = y, alpha = 1/10)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


# using plotting.scatter_matrix() function to draw Histogramsfor each column and scatter plotting between numerical columns and use figsize(,) parameter to show it obviously

# ## member_birth_year  vs  duration_hr
# 
# Plotting member_birth_year linear relationship. For duration_hr above 0.5h , there appears to be a member_birth_year: based on the trend below 20h duration_hr , we might expect member_birth_year to take duration_hr between 0 and 5

# In[87]:


scatterplots('member_birth_year','duration_hr')


# In[ ]:





# ### Talk about some of the relationships you observed in this part of the investigation. How did the feature(s) of interest vary with other features in the dataset?
# 
# Price had a surprisingly high amount of correlation with the diamond size, even before transforming the features. An approximately linear relationship was observed when price was plotted on a log scale and carat was plotted with a cube-root transform. The scatterplot that came out of this also suggested that there was an upper bound on the diamond prices available in the dataset, since the range of prices for the largest diamonds was much narrower than would have been expected, based on the price ranges of smaller diamonds.
# 
# There was also an interesting relationship observed between price and the categorical features. For all of cut, color, and clarity, lower prices were associated with increasing quality. One of the potentially major interacting factors is the fact that improved quality levels were also associated with smaller diamonds. This will have to be explored further in the next section.
# 
# ### Did you observe any interesting relationships between the other features (not the main feature(s) of interest)?
# 
# Expected relationships were found in the association between the 'x', 'y', and 'z' measurements of diamonds to the other linear dimensions as well as to the 'carat' variable. A small negative correlation was observed between table size and depth, but neither of these variables show a strong correlation with price, so they won't be explored further. There was also a small interaction in the categorical quality features. Diamonds of lower clarity appear to have slightly better cut and color grades.
# 
# 
# ## Multivariate Exploration
# 
# The main thing I want to explore in this part of the analysis is how the three categorical measures of quality play into the relationship between price and carat.

# #### Faceting for Multivariate Data
# you saw how FacetGrid could be used to subset your dataset across levels of a categorical variable, and then create one plot for each subset. Where the faceted plots demonstrated were univariate before, you can actually use any plot type, allowing you to facet bivariate plots to create a multivariate visualization.

# ###### using hist() function to draw Histogram for each column and use figsize(,) parameter to show it obviously.

# In[88]:


# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df_go_bike.hist(figsize = (20,20));


# ##### using plotting.scatter_matrix() function to draw Histogramsfor each column and scatter plotting between numerical columns and use figsize(,) parameter to show it obviously

# In[ ]:


pd.plotting.scatter_matrix(df_go_bike, figsize = (30,30));


# The faceted box plot suggests a slight interaction between the two categorical variables, where, in level B of "member_gender", the level of "user_type" seems to be have a larger effect on the value of "bike_id", compared to the trend within "member_gender" level A.

# In[ ]:


g = sns.FacetGrid(data = df_go_bike, col = 'member_gender', size = 6)
g.map(sns.boxplot, 'user_type', 'bike_id')


# The faceted box plot suggests a slight interaction between the two categorical variables, where, in level B of "member_birth_year", the level of "user_type" seems to be have a larger effect on the value of "member_gender", compared to the trend within "member_birth_year" level A.
# 
# FacetGrid also allows for faceting a variable not just by columns, but also by rows. We can set one categorical variable on each of the two facet axes for one additional method of depicting multivariate trends.

# Setting margin_titles = True means that instead of each facet being labeled with the combination of row and column variable, labels are placed separately on the top and right margins of the facet grid. This is a boon, since the default plot titles are usually too long.

# In[ ]:


g = sns.FacetGrid(data = df_go_bike, col = 'user_type', row = 'member_birth_year', size = 2.5,margin_titles = True)
g.map(plt.scatter, 'bike_id', 'member_gender')


# The code for the 2-d bar chart doesn't actually change much. The actual heatmap call is still the same, only the aggregation of values changes. Instead of taking size after the groupby operation, we compute the mean across dataframe columns and isolate the column of interest.

# In[ ]:


cat_means = df_go_bike.groupby(['member_gender', 'user_type']).mean()['duration_hr']
cat_means = cat_means.reset_index(name = 'duration_hr_avg')
cat_means = cat_means.pivot(index = 'user_type', columns = 'member_gender',
                            values = 'duration_hr_avg')
sns.heatmap(cat_means, annot = True, fmt = '.3f',
           cbar_kws = {'label' : 'mean(duration_hr)'})


# ### Talk about some of the relationships you observed in this part of the investigation. Were there features that strengthened each other in terms of looking at your feature(s) of interest?
# 
# I extended my investigation of price against diamond size in this section by looking at the impact of the three categorical quality features. The multivariate exploration here showed that there indeed is a positive effect of increased quality grade on diamond price, but in the dataset, this is initially hidden by the fact that higher grades were more prevalent in smaller diamonds, which fetch lower prices overall. Controlling for the carat weight of a diamond shows the effect of the other C's of diamonds more clearly. This effect was clearest for the color and clarity variables, with less systematic trends for cut.
# 
# ### Were there any interesting or surprising interactions between features?
# 
# Looking back on the point plots, it doesn't seem like there's a systematic interaction effect between the three categorical features. However, the features also aren't fully independent. But it is interesting in something like the 1-carat plot for prices against cut and clarity, the shape of the 'cut' dots is fairly similar for the SI2 through VVS2 clarity levels.

# In[ ]:


get_ipython().system('jupyter nbconvert --to html --no-input Ford_Go_Bike_Part1.ipynb  ')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script Ford_Go_Bike_Part1.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to latex Ford_Go_Bike_Part1.ipynb')


# In[ ]:


get_ipython().system('jupyter nbconvert --to slides Ford_Go_Bike_Part1.ipynb')

