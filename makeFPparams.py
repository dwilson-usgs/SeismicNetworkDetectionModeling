#!/usr/bin/env python
# stuff to define for the FilterPiker
import pickle
from obspy.core import UTCDateTime

Tlong=12  # a time averaging scale in seconds
domper = 3. # dominant period that you want to pick up to
Tup=0.388
s1=9.36
s2=9.6

starttime = UTCDateTime("2017-08-01")
endtime = UTCDateTime("2019-08-01")
lat=36.3
lon=-87
rad=5.0
max_epi_dist = 500

stas= "*"
nets="US,N4,IU,NM,ET"
chans="HHZ,BHZ"

debug = False
Terr=3.0  # time error for selecting CF max around P arrival
fmin=1
fmax=16


with open('FPinputparams.pickle', 'wb') as f:
    pickle.dump([Tlong, domper, Tup, s1, s2, starttime, endtime,
                 lat, lon, rad, max_epi_dist, stas, nets, chans,
                 debug, Terr, fmin, fmax], f)
f.close()
