#!/usr/bin/env python
#Disclaimer:

#This software is preliminary or provisional and is subject to revision. It is 
#being provided to meet the need for timely best science. The software has not 
#received final approval by the U.S. Geological Survey (USGS). No warranty, 
#expressed or implied, is made by the USGS or the U.S. Government as to the 
#functionality of the software and related material nor shall the fact of release 
#constitute any such warranty. The software is provided on the condition that 
#neither the USGS nor the U.S. Government shall be held liable for any damages 
#resulting from the authorized or unauthorized use of the software.



from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
import matplotlib
from obspy.taup import TauPyModel
from scipy import signal as scisig
from matplotlib import cm

import FilterPicker as fp

# Function to do some plotting of events, the filtered version, and CF's
def PlotTn(tr,Tlong,domper,stat,mag,Edist):
    dT=tr.stats.delta
    nTlong=int(min(np.floor(Tlong/dT),np.floor(len(tr.data)-1)))
    N = int(np.ceil(np.log2(domper/dT)))+1
    fig=plt.figure(1,figsize=(12,12))
    t=np.linspace(0, (tr.stats.npts-1)/ tr.stats.sampling_rate,num=tr.stats.npts)-Tlong*2
    plt.subplot(np.floor(N/2.)+2,1,1)
    plt.plot(t,tr,'k')
    plt.ylabel('raw')
    plt.xlim([-10 , 20])
    #for n in range(N):
    n=0
    nn=1
    while n <= N:      
        nn=nn+1
        plt.subplot(np.floor(N/2.)+2,1,nn)
        plt.plot(t,fp.DiffFilt(tr,n,Tlong),'k')
        plt.ylabel('T%i=%2.2fs' % (n, (2.**n)*dT) )
        plt.xlim([-10 , 20])
        n=n+2
    plt.xlabel('Time [s]')
    plt.suptitle(stat.code + ' m' + str(mag) + ' dist=' + str(Edist)+'km')
    fig=plt.figure(2,figsize=(12,12))
    plt.subplot(np.floor(N/2.)+3,1,1)
    plt.plot(t,tr,'k')
    plt.ylabel('raw')
    plt.xlim([-10 , 20])
    n=0
    nn=1
    while n <= N:      
        nn=nn+1
        plt.subplot(np.floor(N/2.)+3,1,nn)
        plt.plot(t,fp.CreateCF(tr,n,Tlong),'k')
        plt.xlim([-10 , 20])
        ax=plt.gca()
        plt.text(-9,max(ax.get_ylim())*.75,'CF T%i=%2.2fs' % (n, (2.**n)*dT) )
        n=n+2
    plt.subplot(np.floor(N/2.)+3,1,np.floor(N/2.)+3)
    plt.plot(t,fp.CreateSummaryCF(tr,Tlong,domper),'k')
    plt.xlim([-10 , 20])
    ax=plt.gca()
    plt.text(-9,max(ax.get_ylim())*.75,'Summary CF')
    plt.xlabel('Time [s]')
    plt.suptitle(stat.code + ' m' + str(mag) + ' dist=' + str(Edist)+'km')
    fig=plt.figure(3,figsize=(12,12))
    viridis = cm.get_cmap('viridis', N)
    y = np.append([np.mean(tr.data[1:nTlong])], tr.data)
    yp=[0.]
    for i in range(1,len(tr.data)):
        yp.append(y[i]-y[i-1])
    f1, Pxx1 = scisig.periodogram(yp, 1./dT)
    n=0
    while n <= N:      
        f, Pxx = scisig.periodogram(fp.DiffFilt(tr,n,Tlong), 1./dT)
        Pxdiff=10*np.log10(Pxx)-10*np.log10(Pxx1)
        plt.semilogx(1./f, Pxdiff,Color=viridis(1.0*n/N))
        plt.text(50,Pxdiff[1],'T%i=%2.2fs' % (n, (2.**n)*dT), color=viridis(1.0*n/N))
        n=n+2
    plt.xlabel('Period [s]')
    plt.ylabel('dB relative to unfiltered')
    plt.suptitle(stat.code + ' m' + str(mag) + ' dist=' + str(Edist)+'km')
    plt.show()



model = TauPyModel(model="iasp91")
client = Client("IRIS")

# stuff to define for the FilterPiker
Tlong=30  # a time averaging scale in seconds
domper = 20 # dominant period that you want to pick up to

starttime = UTCDateTime("2018-08-01")
endtime = UTCDateTime("2019-08-01")
# coordinates and radius of study area
lat=34.9
lon=-106.5
rad=1.5 # in degrees
#max sensor to event distance to analyze
max_epi_dist = 250
#minimum magnitude to analyze
minmag=2.2

stas= "*"
nets="IU"
chans="HHZ,BHZ"

debug = True

# grab some earthquakes
cat = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=minmag,latitude=lat, longitude=lon, maxradius=rad)

# grab a station list
inventory = client.get_stations(network=nets,station=stas,channel=chans,starttime=starttime, endtime=endtime, latitude=lat, longitude=lon, maxradius=rad)
print(inventory)

for cnet in inventory:
    for stat in cnet:
        print(stat)
        for evt in cat:
            try:
                tim=evt.origins[0].time
                epi_dist, az, baz = gps2dist_azimuth(evt.origins[0].latitude,evt.origins[0].longitude, stat.latitude, stat.longitude)
                epi_dist = epi_dist / 1000
                arrivals = model.get_travel_times(source_depth_in_km=evt.origins[0].depth / 1000,
                    distance_in_degree=epi_dist/111.1949,
                    phase_list=["p","P"])
                arrp=arrivals[0]
                arrivals = model.get_travel_times(source_depth_in_km=evt.origins[0].depth / 1000,
                    distance_in_degree=epi_dist/111.1949,
                    phase_list=["s","S"])
                arrs=arrivals[0]
                print(arrp, arrs)
                if epi_dist <= max_epi_dist:     
                    st = client.get_waveforms(cnet.code, stat.code, "*", chans, tim+arrp.time-Tlong*2, tim+arrs.time+30, attach_response=True)
                    print(cnet.code, stat.code, chans, tim, len(st))
                    if debug == True:
                        PlotTn(st[0],Tlong,domper,stat,evt.magnitudes[0].mag,round(epi_dist))
            except:
                print("Could not fetch %s-%s %s" % (cnet.code, stat.code, tim))
                        

