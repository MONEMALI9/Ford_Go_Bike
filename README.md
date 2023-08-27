# Ford GoBike System Data


### Dataset Description 

### What features in the dataset do you think will help support your investigation into your feature(s) of interest?

> **Tip**: information about individual rides made in a bike-sharing system covering the greater San Francisco Bay area.
● Note that this dataset will require some data wrangling in order to make it tidy for analysis. There are multiple cities covered by the linked system, and multiple data files will need to be joined together if a full year’s coverage is desired.
Here are the datasets in CSV format. You can fit your model using the train data, then predict using the test data and submit your predictions in the format of the sample submission.
Your goal is to predict the rotor bearing temperature, which is the Target column in the datasets. 
[here](https://www.google.com/url?q=https://video.udacity-data.com/topher/2020/October/5f91cf38_201902-fordgobike-tripdata/201902-fordgobike-tripdata.csv&sa=D&source=editors&ust=1669750197727856&usg=AOvVaw0RJVqpWyfu7RoKaPH1gynL). Files


## Summary of Findings

### What is the structure of your dataset?

#### Data Dictionary

01 - duration_sec 

02 - start_time 

03 - end_time 

04 - start_station_id 

05 - start_station_name 

06 - start_station_latitude 
   
07 - start_station_longitude

08 - end_station_id  

09 - end_station_name 

10 - end_station_latitude  

11 - end_station_longitude  

12 - bike_id	

13 - user_type

14 - member_birth_year

15 - member_gender	

16 - bike_share_for_all_trip

### What is/are the main feature(s) of interest in your dataset?

I'm most interested in figuring out what features are best for predicting most trips taken in terms of time of day,
 day of the week, or month of the year? and How long does the average trip take? Does the above depend on 
 if a user is a subscriber or customer?.

## Key Insights for Presentation

For the presentation, I focus on just the influence of the four Cs of diamonds
and leave out most of the intermediate derivations. I start by introducing the
price variable, followed by the pattern in carat distribution, then plot the
transformed scatterplot.

Afterwards, I introduce each of the categorical variables one by one. To start,
I use the violin plots of price and carat across clarity. I'm only looking at
the clarity grade plot here since it's the clearest example of how the
categorical quality grades affect diamond pricing. The other two categorical
variables, cut and color, are covered afterwards, using point plots. I've made
sure to use different color palettes for each quality variable to make sure it
is clear that they're different between plots.
