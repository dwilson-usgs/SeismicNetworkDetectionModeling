#!/usr/bin/env python

import numpy as np
import collections
from obspy.clients.fdsn import Client

from obspy.geodetics import gps2dist_azimuth
from obspy.taup import TauPyModel
model = TauPyModel(model="iasp91")
client = Client("IRIS")
import csv

#from numpy import log10 as l10
#from numpy import where as npwh

def calc_model(x,y,coeffs='CEUS',phase='P'):
    """
    Returns the minimum detectable magnitude for
    noise level x (in dB) and distance y (in km).
    If x or y are vectors, they must be the same length.
    coeffs specifies which model coefficients to use (only option is 'CEUS')
    """
    a=get_coeffs(coeffs,phase)
    model=[]
    if np.size(y) > 1:
        for n in range(len(y)):
            model.append(calc_model_hinged(a,x[n],y[n]))
    else:
        model.append(calc_model_hinged(a,x,y))

    return np.asarray(model)

def calc_model_hinged(a,x,y):
    #order of coefficients in a
    #  0, 1,  2,  3,  4,  5,  6,  7,  8, 9
    # R1, R2, c, a1, a2, a3, b1, b2, b3, d
    if y <= a[0]:
        model=(a[2]+a[3]*np.log10(y)+ a[6]*y +a[8]*(x/20))
    elif y >= a[1]:
        model=(a[2]+a[5]*np.log10(y/a[1])+a[4]*np.log10(a[1]/a[0]) +a[3]*np.log10(a[0]) + +a[7]*(a[1]-a[0]) + a[6]*a[0]+a[8]*(x/20))
    else:
        model=(a[2]+a[4]*np.log10(y/a[0])+a[3]*np.log10(a[0])+a[7]*(y-a[0])  +a[6]*a[0]+a[8]*(x/20))
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
    Options are 'CEUS' for model and 'P' or 'S' for phase.
    """
    if coeffs=='CEUS' and phase == 'P':
        a=[  1.06388089e+02,   2.95649447e+02,  -2.83799301e-01,
         3.65990396e+00,  -6.73726922e+00,   1.52716745e+00,
        -5.29509900e-03,   2.09248651e-02,   6.85527053e-01]
    elif coeffs=='CEUS' and phase == 'S':
        #a=[  1.09921161e+02,   2.42091508e+02,   4.48352320e+00,
        #-8.42355568e-01,  -2.24932817e+01,   1.77700026e+00,
         #1.92439289e-02,   6.21760423e-02,   4.12947199e-01]
        a=[  1.05904091e+02,   2.33302291e+02,   4.54873180e+00,
        -6.22339193e-01,  -2.11649770e+01,   1.60184414e+00,
         1.96123207e-02,   6.22953721e-02,   5.06311904e-01]
    else:
        a=0
        print('no such coefficient and phase pair')
    return a

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
    If more that one H channel is listed for a station, the mean H value will be used.
    noiseval is the std of the time domain acceleration data, units are m/s/s.
    """
    Sdict = collections.defaultdict(dict)
    stnum=-1
    with open(csvfile) as csvf:
        readCSV = csv.reader(csvf, delimiter=',')
        for row in readCSV:
            if not Sdict[row[1]]:
                Sdict[row[1]] = collections.defaultdict(dict)
                Sdict[row[1]]['netsta']="%s-%s" % (row[0],row[1])
                Sdict[row[1]]['lat']=np.float(row[2])
                Sdict[row[1]]['lon']=np.float(row[3])
                Sdict[row[1]]['chans'] = collections.defaultdict(dict)
            if row[4] == 'V' and not Sdict[row[1]]['chans']['V']:
                Sdict[row[1]]['chans']['V'] = np.float(row[5])
            if row[4] == 'H' and not Sdict[row[1]]['chans']['H']:
                Sdict[row[1]]['chans']['H'] = np.float(row[5])
            elif row[4] == 'H':
                Sdict[row[1]]['chans']['H'] = (Sdict[row[1]]['chans']['H'] +np.float(row[5]))/2
    return Sdict

def model_thresh(Sdict,x,y,npick,velerr,dist_cut=250,coeffs='CEUS'):
    results=[]
    lnx=0
    for xi in x:
        lnx += 1
        
        for yi in y:
            stat_results=[]
            dists=[]
            for sta in Sdict:
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
                results.append([xi, yi, stat_results[npick-1], np.min(dists), dists[npick-1], magdist, magdist/dists[npick-1],errtot])
                for sta in Sdict:
                    if Sdict[sta]['edist'] <= magdist:
                        if Sdict[sta]['Msta'] >  stat_results[npick-1]:
                            Sdict[sta]['skip'] = Sdict[sta]['skip'] + 1
                        else:
                            Sdict[sta]['hit'] = Sdict[sta]['hit'] + 1
                        
                #print(xi, yi, stat_results[nsta-1])
        print('%3.1f %% done'%(100*lnx/len(x)))
    results=np.asarray(results)
    return results, Sdict
