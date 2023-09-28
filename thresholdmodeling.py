#!/usr/bin/env python
import sys
import numpy as np
import collections
from obspy.clients.fdsn import Client
from copy import copy
import os
from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy import UTCDateTime
model = TauPyModel(model="iasp91")
client = Client("IRIS")
import csv

import requests
from io import StringIO

#from numpy import log10 as l10
#from numpy import where as npwh

def calc_model(x,y,coeffs='CEUS',phase='P'):
    """
    Returns the minimum detectable magnitude for
    noise level x (in dB) and distance y (in km).
    If x or y are vectors, they must be the same length.
    coeffs specifies which model coefficients to use (options are 'CEUS' and 'UTAH')
    """
    a=get_coeffs(coeffs,phase)
    model=[]
    if np.size(y) > 1:
        for n in range(len(y)):
            if 'UTAH' in coeffs:
                model.append(calc_model_log(a,x[n],y[n]))
            else:
                model.append(calc_model_hinged(a,x[n],y[n]))
    else:
        if 'UTAH' in coeffs:
            model.append(calc_model_log(a,x,y))
        else:
            model.append(calc_model_hinged(a,x,y))

    return np.asarray(model)

def calc_model_hinged(a,x,y):
    #order of coefficients in a
    #  0, 1,  2,  3,  4,  5,  6,  7,  8
    # R1, R2, c, a1, a2, a3, b1, b2,  d
    if y <= a[0]:
        model=(a[2]+a[3]*np.log10(y)+ a[6]*y +a[8]*(x/1))
    elif y >= a[1]:
        model=(a[2]+a[5]*np.log10(y/a[1])+a[4]*np.log10(a[1]/a[0]) +a[3]*np.log10(a[0])  +a[7]*(a[1]-a[0]) + a[6]*a[0]+a[8]*(x/1))
    else:
        model=(a[2]+a[4]*np.log10(y/a[0])+a[3]*np.log10(a[0])+a[7]*(y-a[0])  +a[6]*a[0]+a[8]*(x/1))
    return model

def calc_model_log(a,x,y):
    model=a[0]+a[1]*x + a[2]*np.log10(y)
    return model


def calc_magdist(x,m,coeffs='CEUS',phase='P',xmax=1000):
    """
    returns the max distance (in km) at which a magnitude m event
    can be automatically picked at a station with noise
    level x (in dB).  xmax is the max distance to check (in km).
    coeffs specifies which model coefficients to use (only option is 'CEUS')
    """
    #a=get_coeffs(coeffs)
    y=np.arange(1.,xmax,1.)
    xx=y*0 + x
    model=calc_model(xx,y,coeffs,phase)
    indx0=np.where(model<=m)
    d=0.0
    if len(indx0[-1]):
        d=y[indx0[-1][-1]]
    return d


def get_coeffs(coeffs='CEUS',phase='P'):
    """
    returns the model cofficients for the specified model and phase pair.
    Options are 'CEUS' or 'UTAH' for model and 'P' or 'S' for phase.
    """
    if coeffs=='CEUS' and phase == 'P':
        a=[  9.04456865e+01,   3.34962569e+02,   3.95187868e+00,
         9.26180141e-02,  -3.84654266e+00,   9.69092134e-01,
         1.03291267e-02,   1.45788071e-02,   (2.598407e-02)]
    elif coeffs=='CEUS' and phase == 'S':
        a=[  1.34670696e+02,   2.68504171e+02,   4.58460403e+00,
         3.71134520e-01,  -1.93505716e+01,   9.41230691e-01,
         2.99845648e-03,   5.25275336e-02,   (3.117564e-02)]
    elif coeffs=='UTAH' and phase == 'P':
        a=[ 2.76022081 , 0.04084172 , 1.76320561 ]
    elif coeffs=='UTAH' and phase == 'S':
        a=[ 1.79851105 , 0.03525341 , 2.02974234 ]
    else:
        a=0
        print('no such coefficient and phase pair')
    return a

def parse_MUSTANG_psd(stringio):
    """
    parse_MUSTANG_psd(stringio):

    Returns the mean of a list of PSDs requested from IRIS MUSTANG.
    Input object can be either an open file like open('infile','r')
    or text from a returned Request opened as stringio: StringIO(res.text)
    """
    lines = stringio.readlines()
    n = 0
    f, Px = [], []
    f_all = []
    for line in lines:
        if line[0] == '#' or line[0].isalpha():
            continue
        else: 
            freq, pwr = line.strip('\n').split(',')
            f_all.append(freq)
            if freq not in f:
                f.append(freq) # freq is a str here for easy lookup
                Px.append(float(pwr))
            else:
                i_f = f.index(freq)
                Px[i_f] += float(pwr)
    try: 
        n = f_all.count(f_all[0])
        f = np.array(f, dtype='float')
        Px = np.array(Px)/n
    except: # if no data in returned request, force a "dead" channel
        print('no data in returned PSD')
        f = np.array([0.2])
        Px = np.array([-200])
    return f, Px


def sdict_to_csv(Sdict, csvname):
   """
   sdict_to_csv: 

   Write station dictionary to a csv file for editing
   """
   csvfile = open(f'{csvname}', 'w') 
   for stakey in Sdict.keys():
       net = Sdict[stakey]['netsta'].split('-')[0]
       sta = Sdict[stakey]['netsta'].split('-')[1]
       lat = Sdict[stakey]['lat']
       lon = Sdict[stakey]['lon']
       chns = Sdict[stakey]['chans']
       for cha in chns.keys():
           csvfile.write(f'{net}, {sta}, {lat}, {lon}, {cha}, {chns[cha]}\n')
       
   csvfile.close()
   print(f'Wrote station noise values to CSV file {csvname}')
       

def get_noise_MUSTANG(inventory, starttime, endtime, fmin=1.25, fmax=25., fminS=0.8,fmaxS=12.5, 
                      use_profile=False, profile_stat=None):
    """
    get_noise_MUSTANG(inventory, starttime, endtime, fmin=1.25, fmax=25., fminS=0.8,fmaxS=12.5, use_profile=False, profile_stat=None):

    Returns a station dictionary that has station info and noise values computed from PSDs calculated by IRIS MUSTANG.

    """
# Remember: convert dB to acceleration RMS values before returning 
# from Dave: Ok, to get the std of acceleration, start with the PSD values (Px) over a frequency range and convert them from dB back to ground units (A), then sum them over the range*df. Then take the sqrt.  So, it should look like: Stdict[sta.code]['chans']['H'] = np.sqrt(np.sum(df * 10**(Px/10))) 
# df = spacing in frequency space between each point - DW 29 Jan
# it's an integration        
# MUSTANG freqs are log-spaced so we have to calculate df explicitly
# Can use noise profiles - this is a fast way to handle endtime-starttime >> 1 day 
# noise profiles are returned by noise-pdf webservice, which cannot do time slices smaller than 1 day
# Decision: let the user pick what they want, with some warnings?
    if use_profile:
        if type(profile_stat) != str:
            print('error: argument of profile_stat must be a string')
            sys.exit(1)
        print(f'using noise profile for statistic: {profile_stat}' )
        reqbase = 'http://service.iris.edu/mustang/noise-pdf/1/query?nodata=404'
        reqbase += f'&format=noiseprofile_text&noiseprofile.type={profile_stat}' 
        if endtime - starttime < 86400:
            print('Warning: minimum timespan for MUSTANG noise_profile metrics is 1 day')
            print('but your requested endtime - starttime is less than 1 day')
            print('Proceeding anyway') 
    else:
        reqbase = 'http://service.iris.edu/mustang/noise-psd/1/query?nodata=404&format=text'
        if endtime - starttime > 86400:
            print('Requested endtime - starttime is longer than 1 day')
            print('Warning: Requesting many hourly PSDs may slow processing')
            print('Proceeding anyway, but consider using noise_profile option for faster performance') 

    Sdict = collections.defaultdict(dict)
    stnum=-1
    fmax_orig = fmax
    for cnet in inventory:
        for sta in cnet:
            hcount=0
            hsum=0
            if not Sdict[sta.code]:
                Sdict[sta.code] = collections.defaultdict(dict)
            Sdict[sta.code]['netsta']="%s-%s" % (cnet.code, sta.code)
            Sdict[sta.code]['lat']=sta.latitude
            Sdict[sta.code]['lon']=sta.longitude
            for chan in sta:
                print("Working on %s-%s-%s" % (cnet.code, sta.code,chan.code))
                target = f'{cnet.code}.{sta.code}.{chan.location_code}.{chan.code}.M'
                fny = chan.sample_rate/2
                fmax = fmax_orig
                if fmax >= fny*0.75:
                    print(f'requested fmax {fmax} Hz is close to Nyquist frequency')
                    print('this can cause issues due to HF spikes in MUSTANG PSDs')
                    fmax = fny*0.75
                    print(f'using 0.75*fny = {fmax} for this channel instead')
                if fmaxS >= fny*0.75:
                    fmaxS = fny*0.75
                    print('requested fmax is close to Nyquist frequency')
                    print('this can cause issues due to HF spikes in MUSTANG PSDs')
                    print(f'using 0.75*fny = {fmaxS} for this channel instead')
                if not Sdict[sta.code]['chans']:
                        Sdict[sta.code]['chans'] = collections.defaultdict(dict)
                try:
                    if starttime > UTCDateTime() - 3*86400:
                        print('ERROR: start time must be at least 3 days ago to use MUSTANG metrics')
                    t1str = starttime.strftime('%Y-%m-%dT%H:%M:%S')
                    t2str = endtime.strftime('%Y-%m-%dT%H:%M:%S')
                    reqstring = reqbase + f'&target={target}&starttime={t1str}&endtime={t2str}'
                    res = requests.get(reqstring)
                    print(res.status_code)
                    if res.status_code == 200:
                        if use_profile:
                            #not sure if we want to write out profiles
                            #outfile = open(f'{target}_{profile_stat}.txt', 'w')
                            #outfile.write(res.text)
                            #outfile.close()
                            stringio = StringIO(res.text)
                            try:
                                f, Px = np.loadtxt(stringio, unpack=True, delimiter=',')
                            except:
                                print(f'unable to read returned text for {target}')
                                continue
                        else:
                            #not sure if we want to write out PSDs
                            #outfile = open(f'{target}.txt', 'w')
                            #outfile.write(res.text)
                            #outfile.close()
                            stringio = StringIO(res.text)
                            try:
                                f, Px = parse_MUSTANG_psd(stringio)
                            except Exception as e:
                                raise(e)
                                print(f'unable to parse returned text for {target}')
                                continue
                    

                        if np.abs(chan.dip) > 45:
                            i_calc, = np.where((f >= fmin) & (f<=fmax)) #fmin, fmax for vertical (P)
                            comp = 'V'
                        else:
                            i_calc, = np.where((f >= fminS) & (f<=fmaxS)) #fminS, fmaxS for horizontal (S)
                            comp = 'H'

                        # Check to make sure it is not dead: 
                        # PSD vals should be > -150 dB between 4 and 8 s
                        i_checkdead, = np.where((f <= 1./4) & (f >= 1./8))
                        if (Px[i_checkdead] > -150).all():
                            df = np.diff(f[i_calc])
                            stdacc = np.sqrt(np.sum(df*10**(Px[i_calc[1:]]/10)))
                            if comp == 'H':
                                hcount +=1
                                hsum += stdacc
                            if comp == 'V':
                                if not Sdict[sta.code]['chans'][comp]:
                                    Sdict[sta.code]['chans']['V'] = stdacc
                            if comp == 'H' and hcount > 0:
                                Sdict[sta.code]['chans']['H'] = hsum/hcount

                        else:
                            print("Possible dead channel %s-%s-%s" % (cnet.code, sta.code,chan.code))
                    else: 
                        print('data request not successful')
                except:
                    print("Could not fetch %s-%s-%s" % (cnet.code, sta.code,chan.code))
                    #raise(e)
    return Sdict


def calc_noise(inventory,starttime, endtime, fmin=1.25, fmax=25., fminS=0.8,fmaxS=12.5):
    """
    calc_noise(inventory,starttime, endtime, fmin=1.25, fmax=25., fminS=0.8,fmaxS=12.5)
    
    Returns a station dictionary that has station info and noise values computed
    from data fetched from IRIS.
    """
    Sdict = collections.defaultdict(dict)
    stnum=-1
    for cnet in inventory:
        for sta in cnet:
    #        stnum=stnum+1
            hcount=0
            hsum=0
            if not Sdict[sta.code]:
                Sdict[sta.code] = collections.defaultdict(dict)
            Sdict[sta.code]['netsta']="%s-%s" % (cnet.code, sta.code)
            Sdict[sta.code]['lat']=sta.latitude
            Sdict[sta.code]['lon']=sta.longitude
            for chan in sta:
                print("Working on %s-%s-%s" % (cnet.code, sta.code,chan.code))
                if not Sdict[sta.code]['chans']:
                        Sdict[sta.code]['chans'] = collections.defaultdict(dict)
                if np.abs(chan.dip) > 45:
                    try:
                        st = client.get_waveforms(cnet.code, sta.code, chan.location_code, chan.code, starttime, endtime, attach_response=True)
                        st.remove_response(output="ACC")
                        st2=st.copy()
                        st2.filter('bandpass', freqmin=1/8, freqmax=1/4, corners=2)
                        st.filter('bandpass', freqmin=fmin, freqmax=fmax, corners=2)
                        #print(st)
                        if 20*np.log10(np.std(st2[0].data)*2)>-150:
                            if not Sdict[sta.code]['chans']['V']:
                                Sdict[sta.code]['chans']['V'] = np.std(st[0].data)
                        else:
                            print("Possible dead channel %s-%s-%s" % (cnet.code, sta.code,chan.code))
                    except:
                        print("Could not fetch %s-%s-%s" % (cnet.code, sta.code,chan.code))
                if np.abs(chan.dip) < 1.0:
                    try:
                        st = client.get_waveforms(cnet.code, sta.code, chan.location_code, chan.code, starttime, endtime, attach_response=True)
                        st.remove_response(output="ACC")
                        st2=st.copy()
                        st2.filter('bandpass', freqmin=1/8, freqmax=1/4, corners=2)
                        st.filter('bandpass', freqmin=fminS, freqmax=fmaxS, corners=2)
                        #print(st)
                        if 20*np.log10(np.std(st2[0].data)*2)>-150:
                            hcount +=1
                            hsum += np.std(st[0].data)
                        else:
                            print("Possible dead channel %s-%s-%s" % (cnet.code, sta.code,chan.code))
                    except:
                        print("Could not fetch %s-%s-%s" % (cnet.code, sta.code,chan.code))   
            if hcount>0:
                if not Sdict[sta.code]['chans']['H']:
                            Sdict[sta.code]['chans']['H'] = hsum/hcount
    return Sdict

def calc_noise_csv(csvfile):
    """
    calc_noise_csv(csvfile)
    
    Returns a station dictionary that has station info and noise values read
    in from a csv file. csv file format shoult be
    Net, Station, Lat, Lon, V or H, noiseval

    V or H indicates if it is a vertical of horizontal channel.
    If more than one H channel is listed for a station, the mean H value will be used.
    noiseval is the std of the time domain acceleration data, units are m/s/s.
    """
    Sdict = collections.defaultdict(dict)
    stnum=-1
    with open(csvfile) as csvf:
        readCSV = csv.reader(csvf, delimiter=',')
        for row in readCSV:
            if not Sdict[row[1]]:
                Sdict[row[1]] = collections.defaultdict(dict)
                Sdict[row[1]]['netsta']="%s-%s" % (row[0].strip(),row[1].strip())
                Sdict[row[1]]['lat']=np.float(row[2])
                Sdict[row[1]]['lon']=np.float(row[3])
                Sdict[row[1]]['chans'] = collections.defaultdict(dict)
            if row[4].strip() == 'V' and not Sdict[row[1]]['chans']['V']:
                Sdict[row[1]]['chans']['V'] = np.float(row[5])
            if row[4].strip() == 'H' and not Sdict[row[1]]['chans']['H']:
                Sdict[row[1]]['chans']['H'] = np.float(row[5])
            elif row[4].strip() == 'H':
                Sdict[row[1]]['chans']['H'] = (Sdict[row[1]]['chans']['H'] +np.float(row[5]))/2
    return Sdict

def model_thresh(Sdict,x,y,npick,velerr,nsta=5,dist_cut=250,coeffs='CEUS',xl=[]):
    results=[]
    lnx=0
    for xi in x:
        lnx += 1
        
        for yi in y:
            stat_results=[]
            nsta_results=[]
            dists=[]
            for sta in Sdict:
                if sta not in xl:
                    stamag=9.0
                    epi_dist, az, baz = gps2dist_azimuth(yi,xi,Sdict[sta]['lat'],Sdict[sta]['lon'])
                    ya=epi_dist / 1000
                    for chan in Sdict[sta]['chans']:
                        xa=np.log10(Sdict[sta]['chans'][chan])*20.
                        if 'V' in chan:
                            model=calc_model(xa,ya,coeffs,phase='P')
                            stamag=np.min([stamag,np.float(model)])
                        elif 'H' in chan:
                            model=calc_model(xa,ya,coeffs,phase='S')
                            stamag=np.min([stamag,np.float(model)])
                        stat_results.append(model[0])
                        dists.append(epi_dist / 1000)
                    Sdict[sta]['Msta']=np.float(stamag)
                    Sdict[sta]['edist']=ya
                    nsta_results.append(np.float(stamag))
            if np.min(dists) < dist_cut:
                imags=np.argsort(stat_results)
                magdist=0
                errtot=0
                errnum=0
                for ii in imags[0:(npick)]:
                    magdist=np.max([magdist,dists[ii]])
                    errnum+=1
                    errtot+=(dists[ii]*velerr)**2
                errtot=errtot/errnum
                errtot=np.sqrt(errtot)
                #print("magdist: %3.1f, pick_dists: "%(magdist) + str(dists[imags[0]]))
                dists=np.sort(dists)
                stat_results=np.sort(stat_results)
                nsta_results=np.sort(nsta_results)                    
                #check to see if the nsta stamag is >= stat_results[npick-1]
                finmag=np.max([stat_results[npick-1],nsta_results[nsta-1]])
                results.append([xi, yi, finmag, np.min(dists), dists[npick-1], magdist, magdist/dists[npick-1],errtot])
                for sta in Sdict:
                    if Sdict[sta]['edist'] <= magdist:
                        if Sdict[sta]['Msta'] >  finmag:
                            Sdict[sta]['skip'] = Sdict[sta]['skip'] + 1
                        else:
                            Sdict[sta]['hit'] = Sdict[sta]['hit'] + 1
                        
                #print(xi, yi, stat_results[nsta-1])
        print('%3.1f %% done'%(100*lnx/len(x)))
    results=np.asarray(results)
    return results, Sdict
