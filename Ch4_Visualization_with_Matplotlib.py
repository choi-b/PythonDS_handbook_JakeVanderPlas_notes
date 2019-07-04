# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 07:48:55 2019

@author: 604906
"""

#Data Manipulation Tutorials following
#Jake Vanderplas's PythonDataScienceHandbook
#His Github repo: https://github.com/jakevdp/PythonDataScienceHandbook
#Data found at: https://github.com/jakevdp/PythonDataScienceHandbook/tree/master/notebooks/data


#################################################################
### Chapter 4, Visualization with Matplotlb #####################
#################################################################

#Matplotlib: designed to work with SciPy stack.
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

#set the 'classic' style
plt.style.use('classic')

#show() or No show()
# %matplotlib notebook -> interactive plots embedded within the notebook
# %matplotlib inline -> static images of your plot embedded in the notebook
# (done only once per kernel/session)

import numpy as np
x = np.linspace(0, 10, 100)

fig = plt.figure()
plt.plot(x, np.sin(x), '-')
plt.plot(x, np.cos(x), '--')

#Saving Figures to File
fig.savefig('my_figure.png')

#Use IPython Image object to display contents of this file.
from IPython.display import Image
Image('my_figure.png')

#See what other file formats are supported.
fig.canvas.get_supported_filetypes()

## Object-oriented Interface
fig, ax = plt.subplots(2)
ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.cos(x))

## Simple Line Plots
plt.style.use('seaborn-whitegrid')
fig = plt.figure()
ax= plt.axes()
x = np.linspace(0, 10, 1000)
ax.plot(x, np.sin(x))

#Single Figure with Multiple Lines
#use the 'plot' function multiple times
plt.plot(x, np.sin(x))
plt.plot(x, np.cos(x))

## Adjusting the Plot: Line Colors and Styles
plt.plot(x, np.sin(x - 0), color='blue') #color by name
plt.plot(x, np.sin(x - 1), color='g') #color code
plt.plot(x, np.sin(x - 2), color='0.75') #grayscale betw 0 and 1
plt.plot(x, np.sin(x - 3), color='#FFDD44') #hex code
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3)) #RGB tuple, values betw 0 and 1
plt.plot(x, np.sin(x - 5), color='chartreuse') #HTML color names

#line styles
#linestyle = 'solid' <==> linestyle = '-'
#linestyle = 'dashed' <==> linestyle = '--'
#linestyle = 'dashdot' <==> linestyle = '-.'
#linestyle = 'dotted' <==> linestyle = ':'

#line styles + color codes
#plt.plot(x, x+0, '-g') ==> solid green
#plt.plot(x, x+1, '--c') ==> dashed cyan
#plt.plot(x, x+2, '-.k') ==> dashdot black
#plt.plot(x, x+3, ':r') ==> dotted red

#RGB = Red/Green/Blue
#CMYK = Cyan/Magenta/Yellow/blacK


## Adjusting the Plot: Axes Limits
plt.plot(x, np.sin(x))
plt.xlim(-1, 11)
plt.ylim(-1.5, 1.5)

#plt.axis() - set the x and y limits with a single call
plt.plot(x, np.sin(x))
plt.axis([-1, 11, -1.5, 1.5])

#automatically tighten the bounds around the plot
plt.plot(x, np.sin(x))
plt.axis('tight')

#equal aspect ratio (one x unit = one y unit)
plt.plot(x, np.sin(x))
plt.axis('equal')


## Labeling Plots ##

#Titles & Labels
plt.plot(x, np.sin(x))
plt.title('A Sine Curve')
plt.xlabel("x")
plt.ylabel("sin(x)")

#Legends/Labels
plt.plot(x, np.sin(x), '-g', label='sin(x)')
plt.plot(x, np.cos(x), ':b', label='cos(x)')
plt.axis('equal')
plt.legend()

#plt functions to ax methods
#plt.plot() -> ax.plot()
#plt.legend() -> ax.legend()

#plt.-----() -> ax.set_-----()
#with ----- = #[xlabel,ylabel,xlim,ylim,title]


## Simple Scatter Plots ##

# 1) using plt.plot
x = np.linspace(0, 10, 30)
y = np.sin(x)
plt.plot(x, y, 'o', color='black')

#line (-), circle marker (o), black (k)
plt.plot(x, y, '-ok')

#specify lines and markers
plt.plot(x, y, '-p', color='gray',
         markersize=15, linewidth=4,
         markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.ylim(-1.2,1.2)

# 2) using plt.scatter [More Powerful]
plt.scatter(x, y, marker='o')

#properties of each indiv point can be indiv controlled or mapped to data
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3,
            cmap='viridis')
plt.colorbar() #show color scale

#Example with Iris data
from sklearn.datasets import load_iris
iris = load_iris()
features = iris.data.T
plt.scatter(features[0],features[1],alpha=0.2,
            s=100*features[3],c=iris.target,
            cmap='viridis')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
#size of the point = petal width
#color = particular species of flowers
#multicolor/multifeature scatterplots
#=> good for both exploration and presentation of data.

#plt.plot > plt.scatter for larger data
#more efficient/preferred.


## Visualizing Errors ##
x = np.linspace(0, 10, 50)
dy = 0.8
y = np.sin(x) + dy * np.random.randn(50)
plt.errorbar(x, y, yerr=dy, fmt='.k')

## Density & Contour Plots

#Visualize a 3-D function
#z = f(x, y)
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10+ y*x) * np.cos(x)

#contour plot -> plt.contour
x = np.linspace(0,5,50)
y = np.linspace(0,5,40)
X, Y = np.meshgrid(x, y) #meshgrid builds 2-D grids from 1-D arrays
Z = f(X,Y)
plt.contour(X, Y, Z, colors='black')
#By default: negative values = dashed lines
#color-code using the "cmap" argument
plt.contour(X, Y, Z, 20, cmap='RdGy') #RdGy = red-gray colormap.
#also specify more lines be drawn
#20 equally spaced intervals within the data range

#Switch to a filled contour plot using plt.contourf()
plt.contourf(X, Y, Z, 20, cmap='RdGy')
plt.colorbar()


#Histograms, Binnings, and Density
plt.style.use('seaborn-white')
data = np.random.randn(1000)
plt.hist(data)
#more customization
plt.hist(data, bins=30, normed=True, alpha=0.5,
         histtype='stepfilled', color='steelblue',
         edgecolor='none')

#Comparing histograms of several distributions
x1 = np.random.normal(0, 0.8, 1000)
x2 = np.random.normal(-2, 1, 1000)
x3 = np.random.normal(3, 2, 1000)

kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)

plt.hist(x1, **kwargs)
plt.hist(x2, **kwargs)
plt.hist(x3, **kwargs)


## Two-Dimensional Histograms and Binnings
#define x and y drawn from a multiv. Gaussian distrib.
mean = [0,0]
cov = [[1,1],[1,2]]
x, y = np.random.multivariate_normal(mean, cov, 10000).T

#plt.hist2d: 2-D histogram
#2D histograms: show the relationship of intensities at the exact position between two images.
#more helpful info on understanding 2D hist: https://svi.nl/TwoChannelHistogram
plt.hist2d(x, y, bins=30, cmap='Blues')
cb = plt.colorbar()
cb.set_label('counts in bin')

#plt.hexbin: hexagonal binnings
plt.hexbin(x, y, gridsize=30, cmap='Blues')
cb = plt.colorbar(label='count in bin')

#Kernel Density Estimation (KDE)
# "smearing out" the points to make a smooth function.
#EX)
from scipy.stats import gaussian_kde
#[ndim, nsamples]
data = np.vstack([x,y])
kde = gaussian_kde(data)
#Evaluate on a regular grid
xgrid = np.linspace(-3.5,3.5,40)
ygrid = np.linspace(-6,6,40)
Xgrid, Ygrid = np.meshgrid(xgrid,ygrid)
Z = kde.evaluate(np.vstack([Xgrid.ravel(),
                            Ygrid.ravel()]))
#Plot image
plt.imshow(Z.reshape(Xgrid.shape),
           origin='lower', aspect='auto',
           extent=[-3.5,3.5,-6,6],
           cmap='Blues')
cb = plt.colorbar()
cb.set_label("density")

#Customizing Plot Legends
plt.style.use('classic')

x=np.linspace(0,10,1000)
fig, ax = plt.subplots()
ax.plot(x, np.sin(x), '-b', label='Sine')
ax.plot(x, np.cos(x), '--r', label='Cosine')
ax.axis('equal')
leg = ax.legend()

ax.legend(loc='upper left', frameon=False)
fig

ax.legend(frameon=False, loc='lowr center', ncol=2)
fig

#legend in a box/shadow
ax.legend(fancybox=True, framealpha=1, shadow=True, borderpad=1)
fig

#applying labels to the plot elements
y = np.sin(x[:, np.newaxis] + np.pi * np.arange(0,2,0.5))
plt.plot(x, y[:,0], label='first')
plt.plot(x, y[:,1], label='second')
plt.plot(x, y[:,2:])
plt.legend(framealpha=1, frameon=True)

#Multiple legends? - use ax.add_artist()

## Example ##
# MNIST Digits Data
from sklearn.datasets import load_digits
digits = load_digits(n_class=6)

fig, ax = plt.subplots(8,8, figsize=(6,6))
for i, axi in enumerate(ax.flat):
    axi.imshow(digits.images[i], cmap='binary')
    axi.set(xticks=[], yticks=[])

#Project the digits into 2 dim using IsoMap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
projection = iso.fit_transform(digits.data)

#plot
plt.scatter(projection[:,0], projection[:,1],lw=0.1,
            c=digits.target, cmap=plt.cm.get_cmap('cubehelix',6))
plt.colorbar(ticks=range(6), label='digit value')
plt.clim(-0.5,5.5)

#Observations
#Ranges of 5 and 3 nearly overlap.
#0 and 1 instantly separated.

## Multiple Subplots
plt.style.use('seaborn-white')

#Simple Grids of Subplots
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4) #adjust the spacing between the plots.
for i in range (1,7):
    plt.subplot(2, 3, i)
    plt.text(0.5, 0.5, str((2,3, i)),
             fontsize=18, ha='center')

#Example of grid argument
#with normally distributed data
mean = [0, 0]
cov = [[1,1],[1,2]]
x, y = np.random.multivariate_normal(mean, cov, 3000).T

#set up axes with GridSpec
fig = plt.figure(figsize=(6,6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1,1:])
y_hist = fig.add_subplot(grid[:-1,0],xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1,1:],yticklabels=[],sharex=main_ax)

#scatter points on the main axis
main_ax.plot(x,y,'ok',markersize=3,alpha=0.2)

#histogram on the side axes
x_hist.hist(x, 40, histtype='stepfilled',
            orientation='vertical', color='gray')
x_hist.invert_yaxis()
y_hist.hist(y, 40, histtype='stepfilled',
            orientation='horizontal', color='gray')
y_hist.invert_xaxis()

## Text and Annotation ##

#Example: Effect of Holidays on US Births
import pandas as pd
births = pd.read_csv('data/births.csv')

quartiles = np.percentile(births['births'],[25, 50, 75])
quartiles #25th, 50th, 75th percentiles
mu, sig = quartiles[1], 0.74* (quartiles[2] - quartiles[0])
#sigma ~ (q3-q1)/1.35 or the formula above.

#cut out outliers/filter data using '.query' 
births = births.query('(births > @mu - 5 * @sig) & (births < @mu + 5 * @sig)')
births.info() #check data types
births['day'] = births['day'].astype(int)
births.index = pd.to_datetime(10000 * births.year +
                              100 * births.month +
                              births.day, format='%Y%m%d')
#Group the births by month and day.
births_by_date = births.pivot_table('births',
                                    [births.index.month, births.index.day])
births_by_date.index = [pd.datetime(2012, month, day) for (month, day) in births_by_date.index]
#chose a leap year : 2012 so Feb 29th is represented.
fig, ax = plt.subplots(figsize=(12,4))
births_by_date.plot(ax=ax)

#plt.text/ax.text command to add labels to plot
style = dict(size=10, color='gray')

ax.text('2012-1-1', 3950, "New Year's Day", **style)
ax.text('2012-7-4', 4250, "Independence Day", ha='center', **style) #ha: horizontal alignment
ax.text('2012-9-4', 4850, "Labor Day", ha='center', **style)
ax.text('2012-10-31', 4600, "Halloween Day", ha='right', **style)
ax.text('2012-11-25', 4450, "Thanksgiving", ha='center', **style)
ax.text('2012-12-25', 3850, "Christmas", ha='right', **style)

#add axis label/title
ax.set(title='US Births by Day of Year (1969-1988)',
       ylabel='average daily births')

#Format the x axis with centered month labels
ax.xaxis.set_major_locator(mpl.dates.MonthLocator())
ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))
ax.xaxis.set_major_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))

# < more text editing options >
#ax.transData: Transform associated with data coordinates
#ax.transAxes: Transform associated with axes
#fig.transFigure: Transform associated with figure

#Arrows and Annotation
#using ax.annotate & the 'arrowprops' dictionary

## Customizing Ticks ##

#Matplotlib object hierarchy.
# Figure -> could contain 1+ axes objects -> 
# each of which in turn contain other objects representing plot contents

#Hide ticks/labels
#ax.xaxis.set_major_locator(plt.NullLocator())
#ax.yaxis.set_major_locator(plt.NullLocator())

#Reduce/Increase # of Ticks
# .xaxis.set_major_locator(plt.MaxNLocator(3))


## Three-Dimensional Plotting in Matplotlib
# use the keyword projection='3d'

from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection='3d')

#3D points and lines
zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
ax.plot3D(xline, yline, zline, 'black')

#3D scattered points
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')

## 3D Contour Plots ##
#3D contour diagram of a 3D sinusoidal function
def f(x,y):
    return np.sin(np.sqrt(x**2 + y**2))

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)

X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='binary')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#Change viewing angle
#elevation: 60 degrees (60 degrees above the x-y plane)
#azimuth: 35 degrees (rotated 35 degrees counter-clockwise about the z-axis)
ax.view_init(60, 35)
fig


## Wireframes and Surface Plots ##
#wireframe
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color = 'gray')
ax.set_title('wireframe')

#surface plot : like wireframe but filled with polygon
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface')

#Partial polar grid: gives us a slice into the function
r = np.linspace(0, 6, 20)
theta = np.linspace(-0.9 * np.pi, 0.8*np.pi, 40)
r, theta = np.meshgrid(r, theta)

X = r * np.sin(theta)
Y = r * np.cos(theta)
Z = f(X,Y)

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')


## Geographic Data with Basemap ##
#Basemap toolkit: one of the toolkits under mpl_toolkits
#leaflet & Google Maps API may be a better choice for intensive map visualizations
from mpl_toolkits.basemap import Basemap
#Check below for help with Basemap install/import for Python 3 / Windows Users.
#https://stackoverflow.com/questions/35716830/basemap-with-python-3-5-anaconda-on-windows
#AND
#https://github.com/matplotlib/basemap/issues/419
#Install pillow package: pip install pillow
plt.figure(figsize=(8,8))
m = Basemap(projection='ortho', resolution=None,
            lat_0=50, lon_0=-100)
m.bluemarble(scale=0.5)

#use an etopo image (shows topographical features both on land and under the ocean)
fig = plt.figure(figsize=(8,8))
m = Basemap(projection='lcc', resolution=None,
            width=8E6, height=8E6,
            lat_0=45, lon_0=-100,)
m.etopo(scale=0.5, alpha=0.5)

#Map (long, lat) to (x, y) for plotting
x, y = m(-122.3, 47.6)
plt.plot(x, y, 'ok', markersize=5)
plt.text(x, y, ' Seattle', fontsize=12) # the "space" addresses a small overlap.

# Map Projections
from itertools import chain

def draw_map(m, scale=0.2):
    #draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    #lats and longs returned as a dict.
    lats = m.drawparallels(np.linspace(-90,90,13))
    lons = m.drawmeridians(np.linspace(-180,180,13))
    
    #keys contain the plt.line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    #cycle thru the lines and set desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

# Cylindrical Projections 
# (constant latitude and longitude mapped to horizontal and vertical lines)
fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-90, urcrnrlat=90,
            llcrnrlon=-180, urcrnrlon=180, )
draw_map(m)
#llcrn: lower-left corner  & urcrn: upper-right corner
#other cylindrical projections: Mercator (projection='merc')
#and the cylindrical equal-area (projection='cea') projections.

#Pseudo-cylindrical projections
#relax that the meridians (lines of constant longitude) remain vertical.
#give better properties near the poles.
#Common example: The Mollweide projection (projection= "moll")
#Others: sinusoidal ("sinu") and Robinson ("robin")
fig = plt.figure(figsize=(8, 6), edgecolor='w')
m = Basemap(projection='moll', resolution=None,
            lat_0=0, lon_0=0)
draw_map(m)

#Perspective Projections
#useful for showing small portions of the map.
#like photographing Earth from a particular point in space
#common ex: orthographic projection ('ortho')
#other: gnomonic projection ('gnom') & stereographic projection ('stere')
#ex)
fig = plt.figure(figsize=(8,8))
m = Basemap(projection='ortho', resolution=None,
            lat_0=50, lon_0=0)
draw_map(m)

#Conic Projections
#map onto a single cone, which is unrolled
#good local properties.
#common ex: Lambert conformal conic projection ('lcc')
#other: equidistant conic ('eqdc') & Albers equal-area ('aea')
#like perspective projections, good for small to medium patches of the globe.
fig = plt.figure(figsize=(8,8))
m = Basemap(projection='lcc', resolution=None,
            lon_0=0, lat_0=50, lat_1=45, lat_2=55,
            width=1.6E7, height=1.2E7)
draw_map(m)

## Drawing a Map Background
#check book for some of the drawing functions
#Example: Drawing land/sea boundaries
#create both low & high resolution map of Scotland's Isle of Skyre
#You might also need to: conda install basemap-data-hires

fig, ax = plt.subplots(1, 2, figsize=(12,8))
for i, res in enumerate(['l', 'h']):
    m = Basemap(projection='gnom', lat_0=57.3, lon_0=-6.2,
                width=90000, height=120000, resolution=res, ax=ax[i])
    m.fillcontinents(color="#FFDDCC", lake_color="#DDEEFF")
    m.drawmapboundary(fill_color="#DDEEFF")
    m.drawcoastlines()
    ax[i].set_title("resolution='{0}'".format(res))
#low-resolution coastlines: not suitable
#but will work fine for a global view, and MUCH faster than using the high resolution.


## Plotting Data on Maps
# like text, or any plt function.

# Example: California Cities
cities = pd.read_csv('Data/california_cities.csv')

#Filter out desired data
lat = cities['latd'].values
lon = cities['longd'].values
population = cities['population_total'].values
area = cities['area_total_km2'].values

#1. Draw map background
fig = plt.figure(figsize=(8,8))
m = Basemap(projection='lcc', resolution='h',
            lat_0=37.5, lon_0=-119,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

#2. Scatter city data, with color reflection population
#& size reflecting area
m.scatter(lon, lat, latlon=True,
          c=np.log10(population), s=area,
          cmap='Reds', alpha=0.5)

#3. Colorbar and Legend
plt.colorbar(label=r'$\log_{10}({\rm population})$')
plt.clim(3,7)

#legend with dummy points
for a in [100, 300, 500]:
    plt.scatter([], [], c='k', alpha=0.5, s=a,
                label=str(a) + ' km$^2$')
plt.legend(scatterpoints=1, frameon=False,
           labelspacing=1, loc='lower left');
# => Dense areas: near SF and LA.

# Example: Surface Temperature Data (skipped)


## Visualization with Seaborn ##

#Complaints about Matplotlib
#1. Way predated Pandas - not designed for use with Pandas df
#2. Relatively low level API - hard to do sophisticated stat visualization

## Answer: Seaborn
# API on top of Matplotlib
# High-level plotting. Statistical data exploration
# Can do some statistical model fitting too.

#Histograms, KDE, and Densities
data = np.random.multivariate_normal([0,0],[[5,2],[2,2]],size=2000)
data = pd.DataFrame(data, columns=['x','y'])

for col in 'xy':
    plt.hist(data[col], normed=True, alpha=0.5)

import seaborn as sns
#KDE
for col in 'xy':
    sns.kdeplot(data[col], shade=True)
           
#Histograms + KDE using distplot
sns.distplot(data['x'])
sns.distplot(data['y'])         
           
#Kdeplot : 2 dim kernel density plot
sns.kdeplot(data)

#Joint distribubtion and marginal distributions using sns.jointplot
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='kde')
           
#Same thing but with hexagons
with sns.axes_style('white'):
    sns.jointplot("x", "y", data, kind='hex')           
           
#Pairplots
iris = sns.load_dataset("iris")           
iris.head()

sns.pairplot(iris, hue='species', height=2.5)

#Faceted Histograms
#Histograms of subsets
#tip amount received at restaurants
tips = sns.load_dataset('tips')
tips.head()
tips['tip_pct'] = 100 * tips['tip'] / tips['total_bill'] #create a new tip_pct col

grid = sns.FacetGrid(tips, row="sex", col="time", margin_titles=True)
grid.map(plt.hist, "tip_pct", bins=np.linspace(0,40, 15))

#Factor Plot (factorplot renamed to "catplot")
#compares distributions given various discrete factors.
with sns.axes_style(style='ticks'):
    g = sns.catplot("day", "total_bill", "sex", data=tips, kind="box")
    g.set_axis_labels("Day", "Total Bill")

#Joint Distributions
with sns.axes_style('white'):
    sns.jointplot('total_bill', 'tip', data=tips, kind='hex')
#add Kernel density estimate and regression
sns.jointplot("total_bill", "tip", data=tips, kind='reg')

#Bar plots
planets = sns.load_dataset('planets')
planets.head()
#Histogram as a special case of a factor plot.
with sns.axes_style('white'):
    g = sns.catplot('year', data=planets, aspect=2,
                       kind='count', color='steelblue')
    g.set_xticklabels(step=5)
#Separated by method of discovery for each of the planets
with sns.axes_style('white'):
    g = sns.catplot("year", data=planets, aspect=4.0, kind='count',
                    hue='method', order=range(2001,2015))
    g.set_ylabels('Number of Planets Discovered')


# Example: Exploring Marathon Finishing Times (Skipped)


