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

import numpy as np


def DiffFilt(tr,n,Tlong):
    """
    Function to produce filtered differential signals as a function of n
    according to Lomax et al, 2012, eqns. 2-5.  (note that eqn. 5 in the
    oridinal paper and was corrected later
    see:http://alomax.free.fr/FilterPicker/erratum.html)
    :param tr - an obspy trace object
    :param n - the band number
    :param Tlong - the time-averaging scale
    :return: a filtered differential signal in the form of a obspy trace object
    """
    dT=tr.stats.delta
    nTlong=int(min(np.floor(Tlong/dT),np.floor(len(tr.data)-1)))
    Tn = (2**n)*dT
    wn = Tn/(2.*np.pi)
    y=np.diff(tr.data)
    CHP=wn/(wn+ dT)
    CLP = dT/(wn + dT) 
    y = np.append([np.mean(tr.data[1:nTlong])], tr.data)
    yp=[0.]
    for i in range(1,len(tr.data)):
       yp.append(y[i]-y[i-1])
    YHP1, YHP2, YLP = [0.], [0.], [0.]
    for i in range(1,len(yp)):            
        YHP1.append(CHP*(YHP1[i-1] + yp[i] - yp[i-1]))
        YHP2.append(CHP*(YHP2[i-1] + YHP1[i] - YHP1[i-1]))
        #YLP.append(YLP[i-1] + CLP*(YHP2[i] - YHP2[i-1]))
        #the line above was the original incorect eqn. 5, below is correct.
        YLP.append(YLP[i-1] + CLP*(YHP2[i] - YLP[i-1]))
    TRYLP = tr.copy()
    TRYLP.data=np.array(YLP)
    return TRYLP

def CreateCF(tr,n,Tlong):
    """
    Function to produce a characteristic function (CF)
    according to Lomax et al, 2012, eqns. 6-7
    :param tr - an obspy trace object
    :param n - the band number
    :param Tlong - the time-averaging scale
    :return: The CF which is the same length as tr.data
    """
    YLP=DiffFilt(tr,n,Tlong)
    E=np.power(YLP.data,2)
    dT=YLP.stats.delta
    nTlong=int(min(np.floor(Tlong/dT),len(YLP.data)-1))
    CF=[0., 0.]
    for i in range(2,len(E)):
        Eave=np.mean(E[int(max(0,i-nTlong)):i])
        sigma=np.std(E[int(max(0,i-nTlong)):i])
        CF.append( (E[i] - Eave)/sigma)
    return CF

def CreateSummaryCF(tr,Tlong,domper):
    """
    Function to produce a characteristic function (CF)
    according to Lomax et al, 2012, eqns. 6-7 and in
    the text after eqn. 7.
    :param tr - an obspy trace object
    :param n - the band number
    :param Tlong - the time-averaging scale
    :return: The CF which is the same length as tr.data
    """
    dT=tr.stats.delta
    N = int(np.ceil(np.log2(domper/dT)))+1
    CFmatrix=[]
    for n in range(N):    
        CFmatrix.append(CreateCF(tr,n,Tlong))
    CFsum=np.amax(CFmatrix, axis=0)
    CFind=np.argmax(CFmatrix, axis=0)
    return CFsum, CFind

def IntegrateCF(CFsum,dT,Tup,s1):
    """
    Function to Integrate a characteristic function (CF)
    according to Lomax et al, 2012, text after eqn. 7
    on pick declaration.
    :param CFsum - the summary CF
    :param dT - the sampling interval
    :param Tup - the timewidth for pick declaration (in s)
    :param s1 - Threshold 1 from Table 1
    :return: The integrated CF which is the same length as CF
    """
    b=np.ones(CFsum.shape)*5.*s1
    c=np.minimum(CFsum,b)
    ICF=np.zeros(CFsum.shape)
    nn=int(np.floor(Tup/dT))
    print(nn)
    for n in range(int(np.floor(len(c)-Tup/dT))):
        ICF[n]=np.sum(c[n:n+nn])*dT
    return ICF
