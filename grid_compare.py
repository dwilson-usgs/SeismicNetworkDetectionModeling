#!/usr/bin/env python

from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as ml
from obspy.core import UTCDateTime
from scipy.interpolate import griddata
from obspy.geodetics import gps2dist_azimuth
import matplotlib
from obspy.taup import TauPyModel
from scipy import signal as scisig
from obspy.signal import spectral_estimation as spec
from matplotlib import cm
import pickle
import cartopy.crs as ccrs
import cartopy
import cartopy.feature as cfeature
from matplotlib.patches import Circle
import matplotlib.ticker as mticker
import csv
from matplotlib import colors

import thresholdmodeling as tm       ### this is the threshold modeling module

model = TauPyModel(model="iasp91")
client = Client("IRIS")

##########################################################
################# User input Section #####################

f1='DetectGridnewmadB2019013105'
f2='DetectGridnewmad2019013105'

##########################################################

with open('%s.pickle'%f1,'rb') as f:
    xi,yi,zi,Sdict = pickle.load(f)
f.close()
with open('%s.pickle'%f2,'rb') as f:
    xi2,yi2,zi2,Sdict2 = pickle.load(f)
f.close()

central_lon=np.mean(xi)
lnmin=np.min(xi)
lnmax=np.max(xi)
ltmin=np.min(yi)
ltmax=np.max(yi)
extent=[lnmin,lnmax,ltmin,ltmax]
    
if 1:
    plt.figure(1, figsize=(10,6))
    c1=np.floor(np.min(zi)*10)/10
    c2=np.ceil(np.max(zi)*10)/10
    #c1=1.0
    #c2=2.9
    #ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax = plt.axes(projection=ccrs.Mercator(central_lon))
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    plt.contourf(xi, yi, zi.reshape(xi.shape), np.arange(c1, c2+.01, 0.1), cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
    gridlines=ax.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))

    # Add color bar
    plt.clim(c1,c2)
    cbar=plt.colorbar()
    cbar.set_label('Detection Threshold (M)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f1)
    for sta in Sdict:
        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())

if 1:
    plt.figure(2, figsize=(10,6))
    #c1=np.floor(min(zi)*10)/10
    #c2=np.ceil(max(zi)*10)/10
    #c1=1.0
    #c2=2.9
    #ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax = plt.axes(projection=ccrs.Mercator(central_lon))
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    plt.contourf(xi2, yi2, zi2.reshape(xi.shape), np.arange(c1, c2+.01, 0.1), cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
    gridlines=ax.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))

    # Add color bar
    plt.clim(c1,c2)
    cbar=plt.colorbar()
    cbar.set_label('Detection Threshold (M)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f2)
    for sta in Sdict2:
        plt.plot(Sdict2[sta]['lon'],Sdict2[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())

if 1:
    plt.figure(3, figsize=(10,6))
    c1=np.floor(np.min(zi2-zi)*20)/20
    c2=np.ceil(np.max(zi2-zi)*20)/20
    c1=np.min([c1,-.05])
    c2=np.max([c2,.05])
    c3=np.max([np.abs(c1),np.abs(c2)])
    norm = colors.DivergingNorm(vmin=np.min([c1,-.001]), vcenter=0, vmax=np.max([c2,.001]))
    contours=np.arange(c1, c2+.05, 0.05)
    indx=np.where(np.abs(contours)>0.001)
    contours=contours[indx[0]]
    #c1=1.0
    #c2=2.9
    #ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax = plt.axes(projection=ccrs.Mercator(central_lon))
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    zi3=zi2-zi
    
    plt.contourf(xi2, yi2, zi3.reshape(xi.shape),  contours, vmin=np.min([c1,-.001]), norm=norm, vmax=np.max([c2,.001]), cmap=plt.cm.seismic, transform=ccrs.PlateCarree() )
    plt.clim(c1,c2)
    
        
    gridlines=ax.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))

    # Add color bar
    
    cbar=plt.colorbar()
    cbar.set_label('Detection Threshold difference (M)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('%s-%s'%(f2,f1))
    for sta in Sdict:
        if Sdict[sta]['chans']['V'] or Sdict[sta]['chans']['H']:
            plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kx', markersize=4.5, transform=ccrs.PlateCarree())
    for sta in Sdict2:
        if Sdict2[sta]['chans']['V'] or Sdict2[sta]['chans']['H']:
            plt.plot(Sdict2[sta]['lon'],Sdict2[sta]['lat'], 'ko', markersize=5.5, fillstyle='none',transform=ccrs.PlateCarree())


if 1:
    plt.figure(4, figsize=(10,6))
    c1=np.floor(np.min(zi2-zi)*10)/10
    c2=np.ceil(np.max(zi2-zi)*10)/10
    c3=np.max([np.abs(c1),np.abs(c2)])
    #c1=1.0
    #c2=2.9
    #ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax = plt.axes(projection=ccrs.Mercator(central_lon))
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    zi3=zi2-zi
    #plt.contourf(xi2, yi2, zi3.reshape(xi.shape), np.arange(-c3, c3+.05, 0.05), cmap=plt.cm.seismic, transform=ccrs.PlateCarree() )
    gridlines=ax.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))

    # Add color bar
    #plt.clim(-c3,c3)
    #cbar=plt.colorbar()
    #cbar.set_label('Detection Threshold difference (M)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('%s-%s'%(f2,f1))
    for sta in Sdict:
        if Sdict[sta]['chans']['V'] or Sdict[sta]['chans']['H']:
            plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kx', markersize=4.5, transform=ccrs.PlateCarree())
    for sta in Sdict2:
        if Sdict2[sta]['chans']['V'] or Sdict2[sta]['chans']['H']:
            plt.plot(Sdict2[sta]['lon'],Sdict2[sta]['lat'], 'ko', markersize=5.5, fillstyle='none',transform=ccrs.PlateCarree())

    for sta in Sdict:
        if len(Sdict[sta])>0 and (Sdict[sta]['chans']['V'] or Sdict[sta]['chans']['H']) and len(Sdict2[sta])>0:
            if (Sdict2[sta]['chans']['V'] or Sdict2[sta]['chans']['H']):
                if  ((Sdict[sta]['chans']['V'] and Sdict2[sta]['chans']['V']) and
                         Sdict[sta]['lon'] > lnmin and Sdict[sta]['lon'] < lnmax and
                         Sdict[sta]['lat'] > ltmin and Sdict[sta]['lat'] < ltmax):
                    dbdiff=20*(np.log10(Sdict2[sta]['chans']['V'])-np.log10(Sdict[sta]['chans']['V']))
                    if dbdiff>0:
                        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'ro', alpha=.5,markersize=dbdiff*5.5*.5, transform=ccrs.PlateCarree())
                    else:
                        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'bo', alpha=.5,markersize=-dbdiff*5.5*.5, transform=ccrs.PlateCarree())
    
if 1:
    plt.figure(5, figsize=(10,6))
    c1=np.floor(np.min(zi2-zi)*10)/10
    c2=np.ceil(np.max(zi2-zi)*10)/10
    c3=np.max([np.abs(c1),np.abs(c2)])
    #c1=1.0
    #c2=2.9
    #ax = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax = plt.axes(projection=ccrs.Mercator(central_lon))
    ax.set_extent(extent)

    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax.add_feature(cartopy.feature.STATES, edgecolor='black')

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'
    zi3=zi2-zi
    #plt.contourf(xi2, yi2, zi3.reshape(xi.shape), np.arange(-c3, c3+.05, 0.05), cmap=plt.cm.seismic, transform=ccrs.PlateCarree() )
    gridlines=ax.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))

    # Add color bar
    #plt.clim(-c3,c3)
    #cbar=plt.colorbar()
    #cbar.set_label('Detection Threshold difference (M)')

    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('%s-%s'%(f2,f1))
    for sta in Sdict:
        if len(Sdict[sta])>0 and (Sdict[sta]['chans']['V'] or Sdict[sta]['chans']['H']):
            plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kx', markersize=4.5, transform=ccrs.PlateCarree())
    for sta in Sdict2:
        if len(Sdict2[sta])>0 and (Sdict2[sta]['chans']['V'] or Sdict2[sta]['chans']['H']):
            plt.plot(Sdict2[sta]['lon'],Sdict2[sta]['lat'], 'ko', markersize=5.5, fillstyle='none',transform=ccrs.PlateCarree())

    for sta in Sdict:
        if len(Sdict[sta])>0 and (Sdict[sta]['chans']['V'] or Sdict[sta]['chans']['H']) and len(Sdict2[sta])>0:
            if (Sdict2[sta]['chans']['V'] or Sdict2[sta]['chans']['H']):
                if  ((Sdict[sta]['chans']['H'] and Sdict2[sta]['chans']['H']) and
                         Sdict[sta]['lon'] > lnmin and Sdict[sta]['lon'] < lnmax and
                         Sdict[sta]['lat'] > ltmin and Sdict[sta]['lat'] < ltmax):
                    dbdiff=20*(np.log10(Sdict2[sta]['chans']['H'])-np.log10(Sdict[sta]['chans']['H']))
                    if dbdiff>0:
                        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'ro', alpha=.5,markersize=dbdiff*5.5*.5, transform=ccrs.PlateCarree())
                    else:
                        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'bo', alpha=.5,markersize=-dbdiff*5.5*.5, transform=ccrs.PlateCarree())
    

plt.show()
    
