# EXAMPLE CODE TO APPLY TWO-PART EEG BAND DISCOVERY METHOD TO AN INPUT POWER
# SPECTRUM.

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: decisionTreeBinning
# GitHub: https://github.com/mi3nts/decisiontreeBinning

# IMPORT FUNCTIONS
from functions import artificialSpectrum
from getOptimalBands import getOptimalBands

# GENERATE DATA
ps, f = artificialSpectrum(1, 30, 150)

# APPLY METHOD
optimal_num_bands, optimal_band_dict, clf, min_QS = getOptimalBands(f, ps)

# PRINT BAND BOUNDARIES
print(optimal_band_dict)
