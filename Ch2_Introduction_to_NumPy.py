#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 21:02:52 2019

@author: vivabrian
"""

#Data Manipulation Tutorials following
#Jake Vanderplas's PythonDataScienceHandbook
#His Github repo: https://github.com/jakevdp/PythonDataScienceHandbook
#Data found at: https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data


#################################################################
### Chapter 2, The Basics of NumPy  (Numerical Python) Arrays ###
#################################################################


#Goals:
#1. Attributes (size, shape, memory consumption, data types)
#2. Indexing (get and set value of individual array elements)
#3. Slicing arrays (get and set smaller subarrays)
#4. Reshaping (change shape of array)
#5. Combine multiple arrays into one, split one array into many

import numpy as np
np.random.seed(0) #to run the code: "fn + f9"

### Data Attributes ###

x1 = np.random.randint(10, size=6) #1-dim array
x2 = np.random.randint(10, size=(3, 4)) #2-dim array
x3 = np.random.randint(10, size=(3, 4, 5)) #3-dim array

print("x3 ndim:", x3.ndim)
print("x3 shape:", x3.shape)
print("x3 size:", x3.size)
print("x3 dtype:", x3.dtype)

print("itemsize:", x3.itemsize, "bytes") #size in bytes per element
print("nbytes:", x3.nbytes, "bytes") #total size in bytes

### Accessing Subarrays ###
x = np.arange(10)
#first five
x[:5]
#last five
x[5:]
#4th, 5th, and 6th
x[4:7]
#every other element
x[::2]
#every other element starting at index 1 (so odd numbers here)
x[1::2]
#reverse order with negative signs
x[::-1]
#x[-5::2] & x[5::2] are the same. Weird.

#multidimensional arrays
x2
x2[:2,:3] #two rows, three columns..
x2[:3,::2] #all rows, every other column

#reverse subarray dimensions
x2[::-1,::-1]

### Reshape Arrays ###


#putnumbers  1 through 9 in a 3x3 grid using .reshape()
grid = x[1:10].reshape((3,3))
grid
#another way to reshape.
x = np.array([1,2,3])
x.reshape((1,3)) #row vector via reshape
x[np.newaxis,:] #row vector via newaxis.
x.reshape((3,1)) #column vector via reshape
x[:,np.newaxis] #column vector via newaxis.

## Array Concatenation ##
x = np.array([1,2,3])
y = np.array([3,2,1])
np.concatenate([x,y])
z = np.array([4,5,6]) #do it with three arrays
np.concatenate([x,y,z])

#concatenate for 2-D arrays
grid = np.array([[1,2,3],
                 [4,5,6]])
np.concatenate([grid,grid])
#concatenate along second axis (zero-indexed)
np.concatenate([grid,grid], axis=1)

#when working with arrays of mixed dimensions
#you can use np.vstack, np.hstack
np.vstack([x,grid])
y = np.array([[7],
              [8]])
np.hstack([grid,y])

### Splitting arrays ###

x = [1,2,3, 22, 23, 7, 8, 9]
x1, x2, x3 = np.split(x, [3, 5]) #split x with split points at indices 3 & 5 or ('4').
print(x1, x2, x3) #N splits => N+1 subarrays

#splits using np.hsplit and np.vsplit
grid = np.arange(16).reshape((4,4))
print(grid)
grid.shape
upper, lower = np.vsplit(grid,[2])
print(upper)
print(lower)
left, right = np.hsplit(grid,[2])
print(left)
print(right)

## Computation on NumPy Arrays: Universal Functions ##

# Key to making NumPy computations fast? Use VECTORIZED OPERATIONS. 
# (generally through Python's universal functions aka "ufuncs")

#ex)
np.arange(5)/np.arange(1,6)
x = np.arange(9).reshape((3,3))
x
2**x

# operator |  equivalent u func
#    +     |  np.add
#    -     |  np.subtract
#    -     |  np.negative
#    *     |  np.multiply
#    /     |  np.divide
#    //    |  np.floor_divide   (3 // 2 = 1)
#    **    |  np.power
#    %     |  np.mod  -> modulus/remainder (9 % 4 = 1) 

#Absolute value
x = np.array([-2,-1,0,1,2])
abs(x) #equivalent NumPy ufunc => np.abs() or np.absolute()

#Trig functions
theta = np.linspace(0,np.pi, 3)
print("theta = ", theta)
print("sin(theta) = ", np.sin(theta))
print("cos(theta) = ", np.cos(theta))
print("tan(theta) = ", np.tan(theta))

#Inverse Trig
#arcsin()
#arccos()
#arctan()

#Exponents and Logarithms
x = [1,2,3,4]
print("e^x = ", np.exp(x))
print("2^x = ", np.exp2(x))
print("ln(x) = ", np.log(x))
print("log2(x) = ", np.log2(x))
print("log10(x) = ", np.log10(x))

#Specialized ufuncs
#submodule "scipy.special"  
#often for computing obscure mathematical function on data
from scipy import special

#Gamma functions
x = [1,5,10]
print("gamma(x) =", special.gamma(x))
print("ln|gamma(x)|=", special.gammaln(x))
print("beta(x,2) =", special.beta(x, 2))

## Aggregations ##

x = np.arange(1,6)
np.add.reduce(x) #gives the sum of all elements in the array.
#1+2+3+4+5 = 15
np.multiply.reduce(x) #product of all array elements
#store all the intermediate results using accumulate(x).
np.add.accumulate(x) 
np.multiply.accumulate(x)

#outer product
x = np.arange(1, 6)
np.multiply.outer(x,x)

#Summing values n an array
L = np.random.random(100)
sum(L)
np.sum(L)
%timeit sum(L)
%timeit np.sum(L) #much faster. plus np.sum is aware of multiple array dim.

#Min, Max
min(L), max(L) #vs.
np.min(L), max(L) #much faster.

#Multidim. aggregates
M = np.random.random((3,4))
print(M)

M.min(axis=0) #compute MIN across each COLUMN
M.max(axis=1) #compute MAX across each ROW

#Other aggregation functions

# np.sum
# np.prod
# np.mean
# np.std
# np.var
# np.min
# np.max
# np.argmin ==> Find index of minimum value
# np.argmax ==> Find index of maximum value
# np.median
# np.percentile
# np.any ==> evaluate whether any elements are true
# np.all ==> evaluate whether all elements are true


#Example - avg height of US Presidents
import pandas as pd
data = pd.read_csv('Data/president_heights.csv')
heights = np.array(data['height(cm)'])
print(heights)

#compute a variety of summary statistics

print("Mean height:", heights.mean())
print("Std dev height:", heights.std())
print("Min height:", heights.min())
print("Max height:", heights.max())

#compute quantiles

print("25th percentile:", np.percentile(heights,25))
print("Median:", np.percentile(heights,50))
print("75th percentile:", np.percentile(heights,75))

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.hist(heights)
plt.title('Height Distribution of US Prez')
plt.xlabel('Height (cm)')
plt.ylabel('number');


## Computation on Arrays: Broadcasting ##
#Broadcasting: set of rules for applying binary ufuncs (addition, subtraction, etc.)
#on arrays of different sizes.

a = np.array([0,1,2])
a + 5
#higher dimensions
a = np.arange(3)
b = np.arange(3)[:, np.newaxis]
a+b

#Broadcasting Example 1
M = np.ones((2,3))
M.shape
a.shape
#Array "a" has fewer dimensions, so pad it on the left with ones
#M.shape -> (2,3)
#a.shape -> (1,3)
#First dimension disagrees, so stretch this dimension to match
#M.shape -> (2,3)
#a.shape -> (2,3)
M+a
#final shape -> (2,3)


## Comparisons, Marks, and Boolean Logic ##

#Example: Counting Rainy Days - Seattle!!
#365 values of rainfall from Jan 1 to Dec 31, 2014
rainfall = pd.read_csv('Data/Seattle2014.csv')['PRCP'].values
inches = rainfall/254 #1/10mm -> inches
inches.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()

plt.hist(inches, 40);

#efficient to use ufuncs to do element-wise comparisons over arrays.

#Comparison Operators as ufuncs
x = np.array([1,2,3,4,5])
x > 3
x != 3

(2 * x) == (x ** 2)

# operator |  equivalent u func
#    ==    |  np.equal
#    !=    |  np.not_equal
#    <     |  np.less
#    <=    |  np.less_equal
#    >     |  np.greater
#    >=    |  np.greater_equal

#Working with Boolean Arrays
rng = np.random.RandomState(0)
x = rng.randint(10, size=(3,4))

#how many values < 6 ?
np.count_nonzero(x < 6)
np.sum(x < 6) #false interpreted as 0, true: 1

#how many values less than 6 in each ROW?
np.sum(x < 6, axis=1)

#are there ANY values greater than 8?
np.any(x > 8)

#are ALL values in each ROW less than 8?
np.all(x < 8, axis = 1)

#back to rainy days dataset
np.sum((inches >0.5) & (inches <1))

# operator |  equivalent u func
#    &     |  np.bitwise_and
#    |     |  np.bitwise_or
#    ^     |  np.bitwise_xor
#    ~     |  np.bitwise_not

print("Number days without rain:", np.sum(inches ==0))
print("Number days with rain:", np.sum(inches !=0))
print("Days with more than 0.5 inches:", np.sum(inches >0.5))

#Boolean Arrays as Masks
x < 5
#To SELECT these vallues from the array, index on the array -> known as a MASKING operation
x[x < 5]

#Back to Rain Data Example

#Construct a mask of all rainy days
rainy = (inches > 0)
#Construct a mask of all summer days (June 21st: 172nd day)
summer = (np.arange(365) - 172 < 90) & (np.arange(365) - 172 > 0)

print("Median precip on rainy days in 2014 (in.):", np.median(inches[rainy]))
print("Median precip on summer days in 2014 (in.):", np.median(inches[summer]))
print("Max precip on summer days in 2014(in.):", np.max(inches[summer]))
print("Median precip on non-summer rainy days (inches):", np.median(inches[rainy & ~summer]))

## Fancy Indexing - pass arrays of indices in place of single scalars. ##
#Example
rand = np.random.RandomState(42)
x = rand.randint(100, size=10)
print(x)
#access three different elements
[x[3],x[7],x[2]]
#or
x[[3,7,4]]

#fancy indexing in multi dimensions
X = np.arange(12).reshape((3,4))
X

row = np.array([0,1,2])
col = np.array([2,1,3])
X[row,col]
#break it down
#first value: X[0,2]
#second value: X[1,1]
#third value: X[2,3]

#Combined Indexing
print(X)
X[2,[2,0,1]]
# essentially [2,2], [2,0], [2,1]
X[1:, [2,0,1]]
#X[1:,] = 2nd to every row after that
#so... X[1:, [2,0,1] = X[1,[2,0,1]] and X[2,[2,0,1]]

#Example: Setlecting Random Points
mean = [0,0]
cov = [[1,2],
       [2,5]]
X = rand.multivariate_normal(mean, cov, 100)
#help(rand.multivariate_normal)
X.shape

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set() #for plot styling

plt.scatter(X[:,0], X[:,1])

#Use fancy indexing to select 20 random points
indices = np.random.choice(X.shape[0], 20, replace=False) #no repeats
indices

selection = X[indices]
selection.shape

#which points were selected? check 
plt.scatter(X[:,0],X[:,1], alpha=0.3)
plt.scatter(selection[:,0], selection[:, 1],
            facecolor='none',s=200)

#Modifying Values with Fancy Indexing.
x = np.arange(10)
i = np.array([2,1,8,4])
x[i] = 99 #modify the indices 1, 2, 4, 8 to value 99
print(x)
#now subtract 10 on all those modified values (for whatever the reason)
x[i] -= 10
print(x)

#Example - Binning Data (for creating a histogram by hand)
np.random.seed(42)
x = np.random.randn(100)

#compute a histogram by hand
bins = np.linspace(-5,5,20)
counts = np.zeros_like(bins)

#find appropriate bins for each x
i = np.searchsorted(bins,x)

#add 1 to each of these bins
np.add.at(counts, i, 1)

#plot histogram
plt.plot(bins, counts, linestyle = 'steps')

#a faster, more practical way using plt.hist()
plt.hist(x, bins, histtype='step')



## Sorting Arrays ##


#Fast Sorting: np.sort and np.argsort

x = np.array([2,1,4,3,5])
np.sort(x)

#another way
x.sort()
print(x)

#np.argsort : returns the INDICES of the sorted elements
x = np.array([2,1,4,3,5])
i = np.argsort(x)
print(i)

#Sorting along rows or columns
rand = np.random.RandomState(42)
X = rand.randint(0, 10, (4, 6))
print(X)

#sort each column of X
np.sort(X, axis=0) #axis=1 for sorting each row

#Partial Sorts: Partitioning
x = np.array([7,2,3,1,6,5,4])
np.partition(x, 3) #puts the smallest 3 valuesto the left, rest to the right, in arbitrary order.

#partition along an arbitrary axis of a multidim array
np.partition(X, 2, axis=1)
#along each row, first two slots = smallest values, rest arbitrary.

#Example: K-Nearest Neighbors
X = rand.rand(10,2)

%matplotlib inline
import matplotlib.pyplot as plt
import seaborn; seaborn.set()
plt.scatter(X[:,0],X[:,1],s=100)

#compute matrix of square distances
dist_sq = np.sum((X[:, np.newaxis,:] - X[np.newaxis,:,:]) ** 2, axis =-1)

#confusing?
#break it down:
#step 1: for each pair of points, compute differences/distances
differences = X[:, np.newaxis, :] - X[np.newaxis, :, :]
differences.shape

#square it
sq_differences = differences **2
sq_differences.shape

#sum the coordinate differences to get the squared distance
dist_sq = sq_differences.sum(-1)
dist_sq.shape

#diagonals (dist between each point and itself) should all be zero.
dist_sq.diagonal()

#use np.argsort to sort along each row
nearest = np.argsort(dist_sq, axis=1)
print(nearest)
#First column: 0 thru 9 b/c each point's closest neighbor is itself

#Interested in the nearest k neighbors.
#Partition each row so that the smallest k+1 squared distances comes first
K = 2
nearest_partition = np.argpartition(dist_sq, K+1, axis=1)

#visualize it
plt.scatter(X[:, 0], X[:, 1], s=100)

#draw lines from each point to its two nearest neighbors
K = 2
for i in range(X.shape[0]):
    for j in nearest_partition[i, :K+1]:
        #plot line from X[i] to X[j]
        plt.plot(*zip(X[j],X[i]),color='black')


## Structured Data: NumPy's Structured Arrays ##

#Structured Arrays

#create a structured array using compound, specified data types
data = np.zeros(4, dtype={'names':('name','age','weight'),
                          'formats':('U10','i4','f8')})
print(data.dtype)

#U10 - Unicode string of max length 10
#i4 - 4-byte (32bit) integer
#f8 - 8-byte (64bit) float

#now fill the array with values
name = ['Alice', 'Bob', 'Cathy', 'Doug']
age = [25, 45, 37, 19]
weight = [55.0, 85.5, 68.0, 61.5]

data['name'] = name
data['age'] = age
data['weight'] = weight
print(data)

#EX: filtering on age
#get names where age is < 30
data[data['age'] < 30]['name']

#Record Arrays
#fields can be accessed as attributes rather than as dictionary keys

data_rec = data.view(np.recarray)
data_rec.age
#easier, but downside: can be a little bit slower.

#On to Pandas (See Ch3_notes.py)