# SeismicNetworkDetectionModeling
 Python codes to model the detection threshold of seismic networks using FilterPicker

EventFPPickerExample.py is for running and plotting event examples.
FilterPicker.py is the module that contains the various functions corresponding to the equations in Lomax et al., 2012.

For larger batch picker runs:
* Step 1. Run makeFPparams to create a pickle file with input paramaters
* Step 2. Run EventPicker_para.py (edit number of cores to run on in this file).

For network modeling,
- NetThresh\_main\_work\_flow.py
script runs through several approaches to get station info and noise values (from IRIS, from a CSV file, etc.) and model the network detection threshold.

- thresholdmodeling.py
is the module containing all the network modeling functions

- environment.yml
contains the conda environment info that this code was developed on.

---------------------------------------------------------

**Disclaimer:**

>This software is preliminary or provisional and is subject to revision. It is 
being provided to meet the need for timely best science. The software has not 
received final approval by the U.S. Geological Survey (USGS). No warranty, 
expressed or implied, is made by the USGS or the U.S. Government as to the 
functionality of the software and related material nor shall the fact of release 
constitute any such warranty. The software is provided on the condition that 
neither the USGS nor the U.S. Government shall be held liable for any damages 
resulting from the authorized or unauthorized use of the software.

---------------------------------------------------------
