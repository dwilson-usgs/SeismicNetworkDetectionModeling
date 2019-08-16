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
import numpy as np
from obspy.core import UTCDateTime
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel

import pickle

import FilterPicker as fp
model = TauPyModel(model="iasp91")
client = Client("IRIS")

def localmeantime(utc, longitude):
    """
    :param utc: string Ex. '2008-12-2'
    :param longitude: longitude
    :return: Local Mean Time Timestamp
    """
    lmt = utc + (4*60*longitude)
    return lmt

def process_stat(stat,cat):
    cnet.code=stat.alternate_code
    nfail=0;
    slats =[]
    slons =[]
    results=[]
    try:
        with open('FPinputparams.pickle', 'rb') as f:
            Tlong, domper, Tup, s1, s2, starttime, endtime, lat, lon, rad, max_epi_dist, stas, nets, chans, debug, Terr, fmin, fmax = pickle.load(f)
        f.close()
    except:
        print("Can't open input params")
    slats.append(stat.latitude)
    slons.append(stat.longitude)
    
    n=-1
    lats = []
    lons = []
    for evt in cat:
        mags=evt.magnitudes[0].mag
        lats.append(evt.origins[0].latitude)
        lons.append(evt.origins[0].longitude)
        tim=evt.origins[0].time
        edeps=evt.origins[0].depth / 1000

        try:
        #if 1:
            n=n+1
            epi_dist, az, baz = gps2dist_azimuth(lats[n],lons[n], stat.latitude, stat.longitude)
            epi_dist = epi_dist / 1000
            #print(epi_dist)
            arrivals = model.get_travel_times(source_depth_in_km=edeps,
                  distance_in_degree=epi_dist/111.1949,
                  phase_list=["p","P"])
            arrp=arrivals[0]
            arrivals = model.get_travel_times(source_depth_in_km=edeps,
                  distance_in_degree=epi_dist/111.1949,
                  phase_list=["s","S"])
            arrs=arrivals[0]
            if epi_dist <= max_epi_dist:
                try:
                    st = client.get_waveforms(cnet.code, stat.code, "*", chans, tim+arrp.time-Tlong*2, tim+arrs.time+30, attach_response=True)
                    print(cnet.code, stat.code, chans, tim, len(st))
                    CF=fp.CreateSummaryCF(st[0],Tlong,domper)
                    dT=st[0].stats.delta
                    maxCF=np.amax(CF[int(np.floor((Tlong*2-Terr)/dT)):int(np.floor((Tlong*2+Terr)/dT))])
                    stdCF=np.std(CF[0:int(np.floor((Tlong*2-Terr)/dT))])
                    meanCF=np.mean(CF[0:int(np.floor((Tlong*2-Terr)/dT))])
                    ICF=fp.IntegrateCF(CF,dT,Tup,s1)
                    maxICF=np.amax(ICF[int(np.floor((Tlong*2-Terr)/dT)):int(np.floor((Tlong*2+Terr)/dT))])
                    st.remove_response(output="ACC")
                    st.filter('bandpass', freqmin=fmin, freqmax=fmax)
                    stdacc = np.std(st[0].data[0:int(np.floor((Tlong*2-Terr)/dT))])
                    print(mags,epi_dist,maxCF,stdacc, localmeantime(tim, stat.longitude), meanCF, stdCF, maxICF)
                    results.append([mags,epi_dist,maxCF,stdacc, localmeantime(tim, stat.longitude), meanCF, stdCF, maxICF])
                except:
                    print("Could not fetch %s-%s %s" % (cnet.code, stat.code, tim))
                    nfail=nfail+1
        except:
            print("Problem getting arrivals %s-%s %s" % (cnet.code, stat.code, tim))
            nfail=nfail+1   
    with open('FPoutput/resultsICF%s%s.pickle'%(cnet.code, stat.code), 'wb') as f:
        pickle.dump([results, slats, slons, lats, lons, inventory, cat], f)
    f.close()
    res="%s Done with %i fails out of %i" % (stat.code, nfail, n-1)
    return res

##############################################
#  Main
##############################################

with open('FPinputparams.pickle', 'rb') as f:
    Tlong, domper, Tup, s1, s2, starttime, endtime, lat, lon, rad, max_epi_dist, stas, nets, chans, debug, Terr, fmin, fmax = pickle.load(f)
f.close()
        
cat1 = client.get_events(starttime=starttime, endtime=endtime, minmagnitude=1.9,latitude=lat, longitude=lon, maxradius=rad)
print(len(cat1))

inventory = client.get_stations(network=nets,station=stas,channel=chans,starttime=starttime, endtime=endtime, latitude=lat, longitude=lon, maxradius=rad)
print(inventory)

Para=True

from multiprocessing import Pool
from functools import partial

statlist=[]
for cnet in inventory:
    for stat in cnet:
        stat.alternate_code=cnet.code
        statlist.append(stat)

def parallel_runs(statlist):
        pool = Pool(processes=5)
        prod_x=partial(process_stat, cat=cat1)  
        result_list = pool.map(prod_x, statlist) 
        print(result_list)
        
if Para==True:
    if __name__ == '__main__':
        parallel_runs(statlist)
else:
    for stat in statlist:
        result_list=process_stat(stat,cat1)
        print(result_list)

