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

import thresholdmodeling as tm       ### this is the threshold modeling module

model = TauPyModel(model="iasp91")
client = Client("IRIS")

##########################################################
################# User input Section #####################
# time for station noise analysis
#starttime = UTCDateTime("2020-08-01 05:00:00")
#endtime = UTCDateTime("2020-08-01 05:10:00")
starttime = UTCDateTime("2020-01-01 00:00:00")
#endtime = UTCDateTime("2021-01-01 00:00:00")
endtime = UTCDateTime("2020-01-01 12:00:00")

# coordinates of study area
#boxcoords=[38.0, -81.0, 48.0, -66] # new england
#boxcoords=[34.0, -94.0, 41.0, -83] # new madrid
#boxcoords=[33.5, -100.1, 37.5, -94.4] # oklahoma
#boxcoords=[31.0, -91.0, 37.0, -83] # for csv test
boxcoords=[31.0, -91.0, 48.0, -66] # for csv test

# in degrees, box coords buffer to select station out side of box
bb=2 
# number of stations required for event association
nsta=5 
# number of phase picks required for event association
npick=12 
# error level for estimating
velerr=.05
# magnitude to use for plotting detection distance circles
cm=2.0
# Calculate noise levels, or load them from a pickle or csv file?
#calc=False
calc = True
#pickleorcsv='csv'      #this should be 'csv' or 'pickle', only used if calc=False
pickleorcsv='pickle'      #this should be 'csv' or 'pickle', only used if calc=False

# title to be used for saving files
titl="csvtest" # title to be used for saving and loading files

debug = True

if calc:
    # build an inventory
# does Obspy have a service that will return stns based on geographic boundaries? (box/radius?)
    stas= "*"
    nets="IU,US,N4,NE,TA,PE,CN,NM,ET,AG,AO"
#    nets="IU,US,N4,NE"
    chans="HH*,BH*"
    #nets="O2,OK,US,N4,TA,GS"





    inventory = client.get_stations(network=nets,station=stas,channel=chans,starttime=starttime, endtime=endtime, minlatitude=boxcoords[0]-bb,
                                        minlongitude=boxcoords[1]-bb, maxlatitude=boxcoords[2]+bb, maxlongitude=boxcoords[3]+bb, level='response')


    # if you have other sites to add that can't be easily wildcarded, you can add them like this
    othernets="LD"
    othersites="NCB,FLET,KSCT,MCVT,HCNY,NPNY,PAL,ODNJ,WCNY,WVNY,SDMD,GCMD,GEDE,WADE"
    #othersites="BLO,BVIL,CBMO,CGM3,CGMO,EDIL,EVIN,FFIL,FVM,GBIN,HAIL,JCMO,MCIL,MGMO,MPH,MVKY,OHIN,OLIL,PBMO,PIOH,PLAL,PVMO,SCIN,SCMO,SIUC,STIL,TCIN,TYMO,UALR,USIN,UTMT,WVIL"

    inventory += client.get_stations(network=othernets,station=othersites,channel=chans,starttime=starttime,
                                        endtime=endtime,  level='response')

####     end of user input   ####################
#############################################################################   
######### parameters below are default filter parameters for P and S ########
# defaults
fmin=1.25
fmax=20.
fminS=0.8
fmaxS=12.5
#fmin=1.25
#fmax=15.
#fminS=0.8
#fmaxS=12.5

# Median P and S values from noise study
PdBval=-114.4
SdBval=-114.6

Pstd=10**(PdBval/20)
Sstd=10**(SdBval/20)

if calc==True:
    #Sdict=tm.calc_noise(inventory,starttime, endtime, fmin, fmax, fminS,fmaxS)
    #Sdict=tm.get_noise_MUSTANG(inventory,starttime, endtime, fmin, fmax, fminS,fmaxS, use_profile=True, profile_stat='50')
    Sdict=tm.get_noise_MUSTANG(inventory,starttime, endtime, fmin, fmax, fminS,fmaxS)
                            
    with open('NoiseVals%s%s.pickle'%(titl,starttime.strftime('%Y%m%d%H')),'wb') as f:
        pickle.dump([Sdict, boxcoords],f)
    f.close()
else:
    if pickleorcsv=='pickle':
        with open('NoiseVals%s%s.pickle'%(titl,starttime.strftime('%Y%m%d%H')),'rb') as f:
            Sdict, boxcoords = pickle.load(f)
        f.close()
    elif pickleorcsv=='csv':
        Sdict=tm.calc_noise_csv('NoiseVals%s%s.csv'%(titl,starttime.strftime('%Y%m%d%H')))

    else:
        print("pickleorcsv must be set to either 'pickle' or 'csv'. ")
        
ltmin=np.floor(boxcoords[0]/2)*2
ltmax=np.ceil(boxcoords[2]/2)*2 +2
lnmin=np.floor(boxcoords[1]/2)*2
lnmax=np.ceil(boxcoords[3]/2)*2 +2

for sta in Sdict:
    Sdict[sta]['skip']=0
    Sdict[sta]['hit']=0

# set up a grid for modelling
#x=np.arange(boxcoords[1],boxcoords[3]+.25,.25)
#y=np.arange(boxcoords[0],boxcoords[2]+.25,.25)
x=np.arange(boxcoords[1],boxcoords[3]+1,1)
y=np.arange(boxcoords[0],boxcoords[2]+1,1)

results, Sdict = tm.model_thresh(Sdict,x,y,npick,velerr,dist_cut=250)

x=np.asarray(results[:,0])
y=np.asarray(results[:,1])
z=np.asarray(results[:,2])
zd=np.asarray(results[:,4])
zd2=np.asarray(results[:,6])
ze=np.asarray(results[:,7])

extent=[boxcoords[1], boxcoords[3], boxcoords[0], boxcoords[2]]
central_lon = np.mean(extent[:2])
central_lat = np.mean(extent[2:])
nbins=300
xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]

zi = griddata( (x,y), z, (xi, yi), method='cubic')
zdi = griddata( (x,y), zd, (xi, yi), method='cubic')
zd2i = griddata( (x,y), zd2, (xi, yi), method='linear')
zei = griddata( (x,y), ze, (xi, yi), method='cubic')

#with open('DetectGrid%s%s.pickle'%(titl,starttime.strftime('%Y%m%d%H')),'wb') as f:
#    pickle.dump([xi,yi,zi],f)
#f.close()
    
if 1:
    plt.figure(7, figsize=(10,6))
    c1=np.floor(min(z)*10)/10
    c2=np.ceil(max(z)*10)/10
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

#    plt.contourf(xi, yi, zi.reshape(xi.shape), np.arange(c1, c2+.01, 0.1), cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
    plt.contourf(xi, yi, zi.reshape(xi.shape), np.arange(c1, c2+.01, 0.25), cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
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
    #plt.title("%s - %s "% (letter.upper(), content.upper()))
    for sta in Sdict:
        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())

if 1:
    plt.figure(2, figsize=(10,7))
    #ax3 = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax3 = plt.axes(projection=ccrs.Mercator(central_lon))
    ax3.set_extent(extent)

    ax3.add_feature(cartopy.feature.OCEAN)
    ax3.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax3.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax3.add_feature(cartopy.feature.STATES, edgecolor='black')
    plt.contourf(xi, yi, zdi.reshape(xi.shape), cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
    # Add color bar
    gridlines=ax3.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))
    #plt.clim(0.0,300)
    cbar=plt.colorbar()
    cbar.set_label('Distance to %ith nearest pick'%(npick))

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.title("%s - %s "% (letter.upper(), content.upper()))
    for sta in Sdict:
        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())

if 1:
    plt.figure(3, figsize=(10,7))
    #ax3 = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax3 = plt.axes(projection=ccrs.Mercator(central_lon))
    ax3.set_extent(extent)

    ax3.add_feature(cartopy.feature.OCEAN)
    ax3.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax3.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax3.add_feature(cartopy.feature.STATES, edgecolor='black')
    zd2i=zd2i+.0001
    plt.contourf(xi, yi, zd2i.reshape(xi.shape), np.arange(1.0, 3.2, 0.2),cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
    # Add color bar
    gridlines=ax3.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))
    #plt.clim(0.0,2)
    cbar=plt.colorbar()
    #plt.clim(1,2.5)
    cbar.set_label('Distance to %ith nearest pick (actual / possible)'%(npick))

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.title("%s - %s "% (letter.upper(), content.upper()))
    for sta in Sdict:
        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())

#lfeat=cartopy.feature.NaturalEarthFeature(category='physical',name='land',scale='50m',facecolor='none')
#ofeat=cartopy.feature.NaturalEarthFeature(category='physical',name='ocean',scale='50m',facecolor='none')
if 1:
    f=open('NoiseCircleVals%s%s.csv'%(titl,starttime.strftime('%Y%m%d%H')),"w+")
    f.write("station, p-dist, s-dist , effectiveness (%), used, total \n")
    plt.figure(1, figsize=(10,8))
    #ax2 = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax2 = plt.axes(projection=ccrs.Mercator(central_lon))
    ax2.set_extent(extent)
    ax2.add_feature(cartopy.feature.OCEAN)
    ax2.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax2.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax2.add_feature(cartopy.feature.STATES, edgecolor='black')
    for sta in Sdict:
        tot=Sdict[sta]['hit']+Sdict[sta]['skip']
        if tot >0.1:
            eff=100*Sdict[sta]['hit'] / (tot)
        else:
            eff=0
        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())
        if Sdict[sta]['chans']['V']:
            PdB=20*np.log10(Sdict[sta]['chans']['V'])
            #m25d=np.exp((2.2-aP[0]-aP[1]*PdB)/aP[2])
            m25d=tm.calc_magdist(PdB,cm,phase='P')
            circle=Circle((Sdict[sta]['lon'],Sdict[sta]['lat']),(m25d)/111.1949,color='blue',alpha=.2,transform=ccrs.Geodetic())
            ax2.add_patch(circle)
        if Sdict[sta]['chans']['H']:
            SdB=20*np.log10(Sdict[sta]['chans']['H'])
            #m25dS=np.exp((2.2-aS[0]-aS[1]*SdB)/aS[2])
            m25dS=tm.calc_magdist(SdB,cm,phase='S')
            circleS=Circle((Sdict[sta]['lon'],Sdict[sta]['lat']),(m25dS)/111.1949,color='green',alpha=.2,transform=ccrs.Geodetic())
            ax2.add_patch(circleS)
        if Sdict[sta]['lon'] > extent[0] and Sdict[sta]['lon'] < extent[1] and Sdict[sta]['lat'] > extent[2] and Sdict[sta]['lat'] < extent[3]:
            plt.text(Sdict[sta]['lon']+.05,Sdict[sta]['lat']+.05, sta,transform=ccrs.Geodetic())
            if Sdict[sta]['chans']['V'] and Sdict[sta]['chans']['H']:
                f.write("%s, %4.1f, %4.1f , %4.1f, %i, %i \n"%(sta,m25d,m25dS,eff, Sdict[sta]['hit'],tot))
            elif Sdict[sta]['chans']['V']:
                f.write("%s, %4.1f, X, %4.1f, %i, %i \n"%(sta,m25d,eff, Sdict[sta]['hit'],tot))
    #plt.plot(slons,slats, 'kd', markersize=4.5, transform=ccrs.PlateCarree())
    gridlines=ax2.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))
    f.close()
    #plt.show()

#print('station ,     effectiveness (%)')
#for sta in Sdict:
#    tot=Sdict[sta]['hit']+Sdict[sta]['skip']
#    if tot >0.1:
#        eff=100*Sdict[sta]['hit'] / (tot)
#    else:
#        eff=0
#    print(sta + ",    %4.1f (%i / %i)" % (eff, Sdict[sta]['hit'],tot))

if 1:
    plt.figure(8, figsize=(10,6))
    #ax3 = plt.axes(projection=ccrs.AlbersEqualArea(central_lon, central_lat))
    ax3 = plt.axes(projection=ccrs.Mercator(central_lon))
    ax3.set_extent(extent)

    ax3.add_feature(cartopy.feature.OCEAN)
    ax3.add_feature(cartopy.feature.LAND, edgecolor='black')
    ax3.add_feature(cartopy.feature.LAKES, edgecolor='black')
    ax3.add_feature(cartopy.feature.STATES, edgecolor='black')
    plt.contourf(xi, yi, zei.reshape(xi.shape), cmap=plt.cm.plasma, transform=ccrs.PlateCarree() )
    # Add color bar
    gridlines=ax3.gridlines(draw_labels=True, color='gray', alpha=.8, linestyle=':')
    gridlines.xlabels_top=False
    gridlines.ylabels_right=False
    gridlines.xlocator = mticker.FixedLocator(np.arange(lnmin,lnmax,2))
    gridlines.ylocator = mticker.FixedLocator(np.arange(ltmin,ltmax,2))
    #plt.clim(0.0,300)
    cbar=plt.colorbar()
    cbar.set_label('Horizontal location error (km)')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    #plt.title("%s - %s "% (letter.upper(), content.upper()))
    for sta in Sdict:
        plt.plot(Sdict[sta]['lon'],Sdict[sta]['lat'], 'kd', markersize=4.5, transform=ccrs.PlateCarree())
    plt.show()

    
