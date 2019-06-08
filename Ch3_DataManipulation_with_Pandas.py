#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 18:23:27 2019

@author: vivabrian
"""

#Data Manipulation Tutorials following
#Jake Vanderplas's PythonDataScienceHandbook
#His Github repo: https://github.com/jakevdp/PythonDataScienceHandbook
#Data found at: https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data


#################################################################
### Chapter 3, Data Manipulation with Pandas ####################
#################################################################

### <Introducing Pandas objects> ###

#Pandas - built on top of NumPy, provides efficient implementation of a DataFrame
#can be seen as enhanced versions of NumPy arrays 
#but with labels rather than simple integer indices
import numpy as np
import pandas as pd

# <Pandas Series Object>: 1-Dim array of indexed data.
#much more general/flexible thna the 1-Dim NumPy ARRAY.
data = pd.Series([0.25,0.5,0.75,1.0])
#index can be strings
data = pd.Series([0.25,0.5,0.75,1.0],
                 index=['a','b','c','d'])
data
#noncontiguous/nonsequential indices
data = pd.Series([0.25,0.5,0.75,1.0],
                 index=[2,5,3,7])
data

## Create Series as specialized dictionary ##
population_dict = {'Cali': 38332521,
                   'TX': 26448193,
                   'NY': 19651127,
                   'FL': 19552860,
                   'IL': 12882135}
population = pd.Series(population_dict)
population['Cali']

#Unlike a dictionary, the Series also supports array-style operations such as slicing.
population['Cali':'IL']

# <Pandas DataFrame Object>:
#1. a generalization of a NumPy array OR
#2. a specialization of a Python dictionary.

area_dict = {'Cali':43242,'TX':3245,'NY':2342,'FL':2323,'IL':2594}
area = pd.Series(area_dict)

states = pd.DataFrame({'population':population,
                       'area':area})
states
states.index
states.columns
#DataFrame: generalization of a 2D NumPy Array (#1) where both rows/columns 
#have a generalized index for accesing the data

#Different Ways of Constructing DataFrame Objects

#1. From a single Series Object
pd.DataFrame(population, columns=['population'])

#2. From a list of dicts
data =[{'a':i,'b':2*i} for i in range(3)]
pd.DataFrame(data)
#missing values -> Pandas will fill them in with NaN "not a number"
pd.DataFrame([{'a':1,'b':2},{'b':3,'c':4}])

#3. From a dictionary of Series Objects
pd.DataFrame({'population': population,
              'area': area})

#4. From a 2D NumPy Array
pd.DataFrame(np.random.rand(3, 2),
             columns=['hi','hello'],
             index=['a','b','c'])

#5. From a NumPy structured array
A = np.zeros(3, dtype=[('A','i8'),('B','f8')])
A
pd.DataFrame(A)

# <Pandas Index Object>:
ind = pd.Index([2,3,5,7,11])
ind[::2]
#Index Objects: indices are immutable (unlike NumPy arrays).
ind[1]=0 #Index does not support mutable operations

## Index as ordered set
# Pandas objects can perform JOINS
indA = pd.Index([1, 3, 5, 7, 9])
indB = pd.Index([2, 3, 5, 7, 11])
indA & indB #intersection
indA | indB #Union
indA ^ indB #Symmetric difference. Complement of the intersection

## Data Indexing and Selction
#indexers: loc, iloc, and ix
data = pd.Series(['a','b','c'], index=[1,3,5])
data
#explicit indexing when indexing
data[1]
#implicit index when slicing
data[1:3]
#loc => attribute allows indexing and slicing that always references the explicit index.
data.loc[1]
data.loc[1:3]
#iloc => allows indexing and slicing that always references the implicit Python-style index
data.iloc[1]
data.iloc[1:3]
#ix => hybrid of the two.


### <Operating on Data in Pandas> ###

## Ufuncs: Index Preservation
rng = np.random.RandomState(42)
ser = pd.Series(rng.randint(0, 10, 4))
ser

df = pd.DataFrame(rng.randint(0, 10, (3,4)),
                  columns=['A','B','C','D'])
df

#applying NumPy ufunc will result in another Pandas object w/ the indices preserved.
np.exp(ser)
np.sin(df * np.pi / 4)

## Ufuncs: Index Alignment
#Pandas auto align indices (convenient when working with incomplete data)

#Ex)
area = pd.Series({'Alaska':1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
#population density: resulting array = union of the indices.
population / area

##Index Alignment in DataFrame
A = pd.DataFrame(rng.randint(0, 20, (2, 2)),
                 columns=list('AB'))
A
B = pd.DataFrame(rng.randint(0, 10, (3, 3)),
                 columns=list('BAC'))
B
#notice the NaNs & Correct index alignment if we do A+B
A+B

#We can use fill_value to fill in the missing values
fill = A.stack().mean() #mean of all values in A
A.add(B, fill_value = fill)

##Ufuncs: Operations Between DataFrame and Series

A = rng.randint(10, size=(3,4))
A
A - A[0]
#You can see here that the subtraction operates row-wise.
df = pd.DataFrame(A, columns=list('QRST'))
df - df.iloc[0]
#If you want column-wise operation instead, specify the 'axis' keyword.
df.subtract(df['R'], axis=0)


### <Handling Missing Data> ###

#Interesting Fact:
#R - contains 4 basic data types
#NumPy - far more than this.

#Pandas Missing Data
#1. the Python None object.
#2. the special floating-point NaN value

#First the "None" object
vals1 = np.array([1, None, 3, 4])
vals1
#performing aggregations like sum() or min() across an array w/ a None value, will result in error
vals1.sum()

#NaN: Not a Number = specifically a floating-point value
vals2 = np.array([1, np.nan, 3, 4])
vals2
vals2.dtype

#NaN is like a data virus - infects any other object it touches.
1 + np.nan
vals2.sum(), vals2.min(), vals2.max() #no error but not useful

#**Special Aggregations in NumPy to ignore the missing values: "nan___"
np.nansum(vals2), np.nanmin(vals2), np.nanmax(vals2)

## Operating on Null Values
#Methods for detecting, removing, and replacing null values in pd data structures.

#isnull() : generate a boolean mask indicating missing values
#notnull() : opposite of isnull()
#dropna() : return a filtered version of the data
#fillna() : return a copy of the data with missing values filled or imputed.

#Detecting Null Values
data = pd.Series([1, np.nan, 'hello', None])
data.isnull()
data[data.notnull()]

#Dropping Null Values
#For a series: straightforward, no problem.
data.dropna() 
#For a dataframe: cannot drop single values. Can only drop full rows or full columns.
#ex)
df = pd.DataFrame([[1, np.nan, 2],
                   [2, 3, 5],
                   [np.nan, 4, 6]])
df
df.dropna() #drop all rows in which ANY null value is present
df.dropna(axis='columns') #drop all columns in which ANY null value is present
#This drops some good data tho..

#Idea: Dropping rows or columns with 'all' NA values, or a majority of NA values. 
df[3] = np.nan
df
df.dropna(axis = 'columns', how = 'all') #drops that third column 
#'thresh' parameter lets you specify a min number of non-null values for the row/column to be kept.
df.dropna(axis = 'rows', thresh = 3) 
#this drops 1st and 3rd row b/c they only contain TWO non-null values.


#Filling Null Values
#ex)
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
data
data.fillna(0)
#Forward fill (propagate the previous value forward)
data.fillna(method='ffill')
#Backward fill
data.fillna(method='bfill')

#For data frame: can specify an axis.
df.fillna(method='ffill', axis=1) #If no prev value, the NA value remains null.


### <Hierarchical Indexing> ### aka multi-indexing

#High-dim data can be compactly represented.
#Example) Representing 2D data w/in a 1D Series.

## ** The Bad Way ** ##
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
pop
#Straightforwardly index or slice the series based on this multiple index
pop[('California', 2010): ('Texas', 2000)]
#But if you want to select all values from 2010, gets messy, not as efficient for large datasets.
pop[[i for i in pop.index if i[1] == 2010]]

## ** The BETTER Way: Pandas MultiIndex ** ##
#Create multiindex from the tuples:
index = pd.MultiIndex.from_tuples(index)
index
#reindex the series with this MultiIndex
pop = pop.reindex(index)
pop
#To access the data for which the second index is 2010.
pop[:, 2010]

## MultiIndex as extra dimension
#Use the unstack() method to quickly convert the multi-indexed series into a DataFrame:
pop_df = pop.unstack()
pop_df
#Adding another column of data (say, population under 18)
pop_df = pd.DataFrame({'total': pop,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014]})
pop_df
#do some operations (fraction of people under 18)
f_u18 = pop_df['under18']/pop_df['total']
f_u18
f_u18.unstack()

## Naming MultiIndex level names
pop.index.names = ['state','year']
pop

#Indexing and slicing a multiindex
pop['California', 2000]
pop[:,2000]
pop[pop > 22000000]

#Slicing operations fail if index is not sorted.
index = pd.MultiIndex.from_product([['a','c','b'],[1,2]])
data = pd.Series(np.random.rand(6), index=index)
data.index.names = ['Char', 'int']
data['a':'b'] #results in error

#solution: sort_index() method
data = data.sort_index()
data

#stacking and unstacking
pop.unstack(level=0)
pop.unstack(level=1)
#stack it again
pop.unstack().stack()

#Index setting and resetting: reset_index()
#Turning index labels into columns
pop_flat = pop.reset_index(name='population')
pop_flat #often times, raw input data will look like this in the real world.
pop_flat.set_index(['state','year'])

## Data Aggregations on Multi-indices
#set up medical data:
index = pd.MultiIndex.from_product([[2013, 2014],[1,2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37
health_data = pd.DataFrame(data, index=index, columns=columns)
health_data

#avg out the measurements in the two visits each year.
data_mean = health_data.mean(level='year')
data_mean
#Using the "axis=1" keyword Take mean along levels on the COLUMNS as well ('shortcut' to GroupBy)
data_mean.mean(axis=1, level='type')


### <Combining Datasets: Concat and Append> ###

#use a convenient function to create a particular type of dataframe for these exercises
def make_df(cols, ind):
    data = {c: [str(c) + str(i) for i in ind] for c in cols}
    return pd.DataFrame(data, ind)

make_df('ABC', range(3))

#Concatenation of Series and DataFrame objects: very similar to concatenation of NumPy arrays 
#using "np.concatenate"

## Simple Concatenation with pd.concat
#1 dim series
ser1 = pd.Series(['A','B','C'], index=[1,2,3])
ser2 = pd.Series(['D','E','F'], index=[4,5,6])
pd.concat([ser1,ser2])
#also works on higher-dim objects, like DataFrames.
df1 = make_df('AB', [1,2])
df2 = make_df('AB', [3,4])
print(df1); print(df2); print(pd.concat([df1, df2]))

#default: concatenation is row-wise within the DataFrame (axis=0)
#try column-wise concatenation with axis=1
df3 = make_df('AB', [0,1])
df4 = make_df('CD', [0,1])
print(df3); print(df4); print(pd.concat([df3, df4], axis=1))

#Duplicate indices? - That's ok. Pandas preserves the indices anyway (unlike NumPy.concatenate!)
x = make_df('AB', [0,1])
y = make_df('AB', [2,3])
y.index = x.index
print(x); print(y); print(pd.concat([x,y]))

#If you want to verify that the indices in the pd.concat() do not overlap: use the 'verify_integrity' flag.
try:
    pd.concat([x,y], verify_integrity=True)
except ValueError as e:
    print("ValueError:", e)

#Ignoring the Index
#Sometimes, index doesn't matter, you prefer to ignore it.
#Use the 'ignore_index' flag.
print(pd.concat([x,y], ignore_index=True))

#Add multiindex keys (hierarchically indexed series)
print(pd.concat([x,y], keys=['x','y']))

#Concatenation with Joins
df5 = make_df('ABC',[1, 2])
df6 = make_df('BCD',[3, 4])
print(df5); print(df6); print(pd.concat([df5, df6], sort=True))
#By default, the join is a union of the input columns (join='outer')
#Change this to an intersection of the columns using join='inner'
print(pd.concat([df5,df6], join='inner'))

#another option: directly specifying the index of the columns using the 'join_axes' argument
print(pd.concat([df5,df6], join_axes=[df5.columns]))

#Sometimes, the Append() method - easier...
print(df1.append(df2))
print(df5.append(df6, sort=True))


### <Combining Datasets: Merge and Join> ###
#Essential features of Pandas: High-performance, in-memory join and merge operations.

#pd.merge() function - types of joins: one-to-one, many-to-one, many-to-many
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group':['Accounting', 'Engineering', 'Engineering', 'HR']})
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date':[2004, 2008, 2012, 2014]})
print(df1); print(df2)
#Combine this into a single DataFrame
df3 = pd.merge(df1, df2)
df3 #recognizes the common "employee" column, joins using this column as a key.

#Many-to-one joins: Two key columns contains duplicate entries. Preserves those duplicate entries.
#Ex)
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3); print(df4); print(pd.merge(df3, df4))

#Many-to-Many joins: key column in both L and R array contains duplicates, result: many-to-many merge.
#Ex)
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(df1); print(df5); print(pd.merge(df1, df5))

## Specifying the Merge Key

#The 'on' keyword
print(pd.merge(df1, df2, on='employee'))

#The 'left_on' and 'right_on' keywords to specify the two column names.
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df1); print(df3)
print(pd.merge(df1, df3, left_on='employee', right_on='name'))
#drop one of key columns to avoid redundancy.
pd.merge(df1, df3, left_on='employee', right_on='name').drop('name', axis=1)

#Merging on an index, rather than a column.
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a); print(df2a);
#specify the left_index and/or right_index flags
print(pd.merge(df1a, df2a, left_index=True, right_index=True))
#The join() method can perform a merge that defaults to joining on indices.
print(df1a.join(df2a))
#Mix indices and columns
print(pd.merge(df1a, df3, left_index=True, right_on='name'))

## Specifying Set Arithmetic for Joins
df6 = pd.DataFrame({'name': ['Peter', 'Paul', 'Mary'],
                    'food': ['fish', 'beans', 'bread']},
                    columns = ['name', 'food'])
df7 = pd.DataFrame({'name': ['Mary', 'Joseph'],
                    'drink': ['wine', 'beer']},
                    columns = ['name', 'drink'])
print(df6); print(df7); print(pd.merge(df6, df7)) #by default, inner join
#Using the "how" keyword.
pd.merge(df6, df7, how='inner')
pd.merge(df6, df7, how='outer')
pd.merge(df6, df7, how='left') #left join, and right join: how = 'right'

## Overlapping Column Names: The Suffixes Keyword
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8); print(df9); print(pd.merge(df8, df9, on='name'))
#automatically appends _x or _y to make the output columns unique.
#You can specify a custom suffix using the suffixes keyword:
print(pd.merge(df8, df9, on='name', suffixes=['_L', '_R']))


#EXAMPLE WITH DATA: US States Data

pop = pd.read_csv('Data/state-population.csv')
areas = pd.read_csv('Data/state-areas.csv')
abbrevs = pd.read_csv('Data/state-abbrevs.csv')

print(pop.head()); print(areas.head()); print(abbrevs.head())

#say we want to
#1. Rank US states and territories by their 2010 population density.

#First do a many to one merge based on the state/region column of pop 
#and the abbrev column of abbrevs.
merged = pd.merge(pop, abbrevs, how='outer',
                  left_on='state/region', right_on='abbreviation')
merged = merged.drop('abbreviation', 1) #drop column 'abbreviation' to avoid duplicate info
merged.head()

#Double check for mismatches/look for rows with nulls
merged.isnull().any()
merged[merged['population'].isnull()].head()
merged.loc[merged['state'].isnull(),'state/region'].unique()
#here we can see that our population data includes entries for PR and USA 
#but these entries do not appear in the state abbreviation key.
#fill in appropriate entries:
merged.loc[merged['state/region'] == 'PR', 'state'] = 'Puerto Rico'
merged.loc[merged['state/region'] == 'USA', 'state'] = 'United States'
merged.isnull().any()
#No more nulls in the state column!

final = pd.merge(merged, areas, on='state', how='left')
final.head()

#check for nulls
final.isnull().any()
#nulls in the area column. See which regions are ignored.
final['state'][final['area (sq. mi)'].isnull()].unique()
#The dataframe does not contain the area of the US as a whole.
#Since the pop density of the entire US is not relevant here, drop the null values.
final.dropna(inplace=True)
final.head()

#Select the portion of data in year 2000, and the total population
data2010 = final.query("year == 2010& ages == 'total'")
data2010.head()

#Compute the population density and display it in order
#reindex our data on state then compute
data2010.set_index('state', inplace=True)
density = data2010['population'] / data2010['area (sq. mi)']
#sort
density.sort_values(ascending=False, inplace=True)
density.head()
#check the end of the list
density.tail()


### <Aggregation and Grouping> ###

#Example) planets data
import seaborn as sns
planets = sns.load_dataset('planets')
planets.shape

#simple aggregation on pandas dataframe
#by default, return results within each column
df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})
df
df.mean()
df.mean(axis='columns') #aggregate within each row

#describe() computes several common aggregates
planets.dropna().describe()

#<<List of some Pandas aggregation methods>>
#can be used for both DataFrame and Series objects
#count() - total number of items
#first(), last() - first and last item
#mean(), median() - mean and median
#min(), max() - minimum and maximum
#std(), var() - std dev and variance
#mad() - mean absolute deviation
#prod() - product of all items
#sum() - sum of all items

## GroupBy: Split, Apply, Combine (coined by Hadley Wickham)
#1. Split - breaking up and grouping a DataFrame depending on the value of the specified key.
#2. Apply - computing some function, usually an aggregate, transformation, or filtering within the individual groups.
#3. Combine - merging the results of these operations into an output array.

#ex)
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data': range(6)}, columns=['key','data'])
df
df.groupby('key') #returns a DataFrameGroupBy object.
df.groupby('key').sum()

#Column indexing
planets.groupby('method') #returns DataFrameGroupBy object
planets.groupby('method')['orbital_period'] #SeriesGroupBy object
planets.groupby('method')['orbital_period'].median()
#shows general scale of orbital periods that each method is sensitive to.

#Iteration over Groups
#GroupBy object supports direct iteration over the groups

for (method, group) in planets.groupby('method'):
    print("{0:30s} shape = {1}".format(method, group.shape))

#Dispatch methods
planets.groupby('method')['year'].describe().unstack()

## Aggregate, filter, transform, and apply.
rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A','B','C','A','B','C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                    columns = ['key', 'data1', 'data2'])
df

# 1. Aggregation
df.groupby('key').aggregate(['min', np.median, max])
#pass a dictionary mapping column names to operations
#applied on that column. (e.g. min, max)
df.groupby('key').aggregate({'data1': 'min',
                             'data2': 'max'})
# 2. Filtering
def filter_func(x):
    return x['data2'].std() > 4
print(df); print(df.groupby('key').std());
print(df.groupby('key').filter(filter_func))
#Group A dropped since it does not have a std dev > 4.

# 3. Transformation
#common ex: centering and subtracting group-wise mean
df.groupby('key').transform(lambda x: x- x.mean())

# 4. The apply() method (take in a DataFrame, and return a Pandas object or scalar)
#an apply fn that normalizes the first column by the sum of the second
def norm_by_data2(x):
    x['data1'] /= x['data2'].sum()
    return x
print(df);print(df.groupby('key').apply(norm_by_data2))

## Specifying the split key.
#Key as series or list
L = [0, 1, 0, 1, 2, 0]
print(df); print(df.groupby(L).sum())
#Dictionary or series mapping Index to group
df2 = df.set_index('key')
mapping = {'A': 'vowel', 'B': 'consonant', 'C': 'consonant'}
print(df2); print(df2.groupby(mapping).sum())

#Any Python function
print(df2); print(df2.groupby(str.lower).mean())
#?str.lower

#A list of valid keys
df2.groupby([str.lower, mapping]).mean()

#Grouping example
decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'
planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)


### <Pivot Tables> ###
# 'multidimensional' version of GroupBy aggregation.
titanic = sns.load_dataset('titanic')
titanic.head()

## Pivot Tables by Hand
titanic.groupby('sex')[['survived']].mean()

#say we want to look at survival by both sex and class
#1. group by class and gender
#2. select survival
#3. apply a mean aggregate
#4. combine the resulting groups
#5. unstack the hierarchical index to reveal the hidden multidimensionality.
titanic.groupby(['sex', 'class'])['survived'].aggregate('mean').unstack()
#but code looks a bit garbled.

## Pivot Table Syntax equivalent
titanic.pivot_table('survived', index='sex', columns='class')
#Survival favors both women and higher classes.

# Multilevel pivot tables
#look at age as a third dimension.
#bin the age using pd.cut 
age = pd.cut(titanic['age'],[0, 18, 80])
titanic.pivot_table('survived',['sex',age],'class')

# Apply same strategy when working with columns.
#use pd.qcut to automatically compute quantiles
fare = pd.qcut(titanic['fare'],2)
titanic.pivot_table('survived',['sex',age],[fare,'class'])
# ==> 4 dimensional aggregation with hierarchical indices...

## Additional pivot table options

titanic.pivot_table(index='sex', columns='class',
                    aggfunc={'survived':sum, 'fare':'mean'})
#aggregation specification: ('sum','mean','count','min','max',etc.)

#computing totals along each grouping via the 'margins' keyword.
titanic.pivot_table('survived',index='sex', columns='class', margins=True)
#class-agnostic survival rate by gender, the gender-agnostic survival rate by class, 
#and the overall survival rate of 38%.


## Example) Birthrate Data
births = pd.read_csv('Data/births.csv')
births.head() #number of births grouped by date & gender

#using pivot_table
births['decade'] = 10 * (births['year'] //10)
births.pivot_table('births', index='decade', columns='gender', aggfunc='sum')

#every decade: male > female births.. -_- #GCDA!!!

#visualize the total number of births by year
%matplotlib inline
import matplotlib.pyplot as plt
sns.set()
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot()
plt.ylabel('total births per year')

#Further data exploration
quartiles = np.percentile(births['births'],[25,50,75])
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[0])
#robust estimate of the sample mean
#0.74 comes from the IQR of a Gaussian distribution

#use query() method to filter out rows with births outside these values
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')

#set day column to integers (originally was a string due to nulls)
births['day'] = births['day'].astype(int)

#create a datetime index from year, month, and day
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
births['dayofweek'] = births.index.dayofweek

#plot births by weekdays for several decades
import matplotlib as mpl
births.pivot_table('births', index='dayofweek',
                   columns='decade', aggfunc='mean').plot()
plt.gca().set_xticklabels(['Mon','Tues','Wed','Thurs','Fri','Sat','Sun'])
plt.ylabel('mean births by day');
#births are slighly less common on weekends than on weekdays.
#1990s and 2000s data are missing b/c the CDC data contains only the month of birth starting in 1989.


#plot the mean number of births by the day of the YEAR
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.head()
births_by_date.tail()
#result: multi-index over months and days
#turn these months and days into a date by associating them with a dummy variable

births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]
births_by_date.head()

#Plot the results (focusing on the month and day only)
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax);

### <Vectorized String Operations> ###

#ex)
## Doing a loop on array of strings
data = ['peter','Paul',None,'Mary','gUIDO']
[s.capitalize() for s in data]
#does not work due to the missing value

#Pandas correctly handles missing data via the "str" attribute of Pandas Series 
#and Index objects containing strings.
names = pd.Series(data)
names
names.str.capitalize()

## Table of Pandas String Methods
monte = pd.Series(['Graham Chapman','John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
monte.str.lower() #lowercase (strings)
monte.str.len() #return length (numbers)
monte.str.startswith('T') #return Boolean values
monte.str.split() #return lists or other compound values for each elem.

#Methods using Regular Expressions
monte.str.extract('([A-Za-z]+)') #extract the first name from each 
monte.str.findall(r'^[^AEIOU].*[^aetou]$')
#find all names that start and end with a consonant.
#start-of-string(^) , end-of-string ($)

#Vectorized item access and slicing
monte.str[0:3]
monte.str.split().str.get(-1)

#Indicator Variables
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C', 'B|D', 'B|C',
                                    'B|C|D']})
full_monte
full_monte['info'].str.get_dummies('|') #quickly split out indicator variables into a DataFrame.


##Dates and Times

#manually build a date using the 'datetime' type:
from datetime import datetime
datetime(year=2015, month=7, day=4)

#parse dates
from dateutil import parser
date = parser.parse("4th of July, 2015")
date

#Print day of the week.
date.strftime('%A') #std. string format code

#Typed arrays of times: NumPy's datetime64
import numpy as np
date = np.array('2015-07-04', dtype=np.datetime64)
#Quickly apply vectorized operations
date + np.arange(12)

#Pandas - Timestamp object
import pandas as pd
date = pd.to_datetime("4th of July, 2015")
date
date + pd.to_timedelta(np.arange(12),'D')


##Pandas Time Series: Indexing by Time
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
data
#Slicing
data['2014-07-04':'2015-07-04']
#Subsetting data by year
data['2015']

##Pandas Time Series Data Structures

#pd.date_range
pd.date_range('2015-07-03', periods=8)
#alter freq='D' (default) to 'H' (hourly timestamps)
pd.date_range('2015-07-03', periods=8, freq='H')
#Creating sequences of period or time delta values
pd.period_range('2015-07', periods=8, freq='M') #monthly periods
pd.timedelta_range(0, periods=10, freq='H') #hourly increase.


## Resampling, Shifting, and Windowing
#Pandas - developed largely in a financial context
#EX) "conda install pandas-datareader"
#this package can import financial data from a number of available sources.
from pandas_datareader import data
goog = data.DataReader('GOOG', start='2004', end='2016',
                       data_source='google')
#Google finance : deprecated.
#Good replacement for financial data as of 2019 - Quandl
#As a result, this section is skipped.


##Example: Visualizing Seattle Bicycle Counts
#download dataset using the link in the pdf.
data = pd.read_csv('Fremont_Bridge.csv', index_col='Date', parse_dates = True)
data.head()

#Shorten column names, and add a "Total" column
data.columns=['West','East']
data['Total'] = data.eval('West + East')
data.dropna().describe()

#Plot raw data
%matplotlib inline
import seaborn; seaborn.set()
import matplotlib.pyplot as plt
data.plot()
plt.ylabel('Hourly Bicycle Count');

#Resample data
weekly = data.resample('W').sum()
weekly.plot(style=[':','--','-'])
plt.ylabel('Weekly bicycle count')

#Use a rolling mean
#30-day rolling mean
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':','--','-'])
plt.ylabel('mean hourly count')

#obtain a smoother vers. of a rolling mean
#using a window function.
#ex) Gaussian window
daily.rolling(50, center=True,
              win_type='gaussian').sum(std=10).plot(style=[':','--','-']);
#specifies both the width of the window and 
#the width of the Gaussian within the window (we chose 10 days)

#Digging deeper into the data
#Avg traffic as a function of the time of day
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4*60*60*np.arange(6)
by_time.plot(xticks=hourly_ticks, style=[':','--','-']);
#Hourly traffic: strongly bimodal

#Based on day of week
by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon','Tues',
                    'Wed','Thurs',
                    'Fri','Sat',
                    'Sun']
by_weekday.plot(style=[':','--','-'])
#strong distinction between weekday and weekends

#A compound groupby look at the hourly trend on weekdays vs weekends
weekend = np.where(data.index.weekday <5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend,data.index.time]).mean()
#Multiple subplots (Matplotlib tool)
fig, ax = plt.subplots(1, 2, figsize=(14,5))
by_time.ix['Weekday'].plot(ax=ax[0], title='Weekdays',
          xticks=hourly_ticks, style=[':','--','-'])
by_time.ix['Weekend'].plot(ax=ax[1], title='Weekends',
          xticks=hourly_ticks, style=[':','--','-']);
#Bimodal commute during work
#Unimodal recreational pattern during weekends.

#Future topics of interest: examine the effect of weather, temperature, time of year, etc.

          
## High-Performance Pandas: eval() and query()

#Motivating query() and eval(): Compound Expressions;
rng = np.random.RandomState(42)
x = rng.rand(int(1E6))
y = rng.rand(int(1E6))
%timeit x+y

mask = (x > 0.5) & (y < 0.5) #better than
tmp1 = (x > 0.5)
tmp2 = (y < 0.5)
mask = tmp1 & tmp2 #this can lead to significant memory & computational overhead

#numexpr: fast numerical expression evaluator
import numexpr
mask_numexpr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
np.allclose(mask, mask_numexpr)

# pandas.eval() for Efficient Operations
nrows, ncols = 100000, 100
rng = np.random.RandomState(42)
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                      for i in range(4))
%timeit df1 + df2 + df3 + df4
#compute the same result via pd.eval 
#by constructing the exression as a string
%timeit pd.eval('df1 + df2 + df3 + df4') #about 50% faster

#pd.eval() supports all comparison operators
result1 = (df1 < df2) & (df2 <= df3) & (df3 != df4)
result2 = pd.eval('df1 < df2 <= df3 != df4')
np.allclose(result1, result2)

# DataFrame.eval() for Column-Wise Operations
df = pd.DataFrame(rng.rand(1000,3), columns=['A','B','C'])
df.head()

#create a new column and assign a computed value
df.eval('D = (A+B)/C', inplace=True)
df.head()
#you can also apply this to modify existing column(s).


##DataFrame.Query() Method
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = df.query('A < 0.5 and B < 0.5')
np.allclose(result1, result2)


##When to Use These Functions
#1. Computation Time
#2. Memory Use