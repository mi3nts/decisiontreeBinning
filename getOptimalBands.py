# getOptimalBands

# FUNCTION TO DERIVE OPTIMAL BAND FOR AN INPUT POWER SPECTRUM USING A TWO-PART
# METHOD. FIRST, DECISION TREE IS USED TO DERIVE OPTIMAL BANDS FOR EVERY CHOICE
# OF BAND COUNT. SECOND, BEST CHOICE OF BAND COUNT IS PICK BASED ON THE QUALITY
# SCORE
# NOTE: THIS IS A DUPLIATE OF THE FUNCTION WITH THE SAME NAME IN THE functions.py
# FILE. FUNCTION IS DUPLICATED FOR EASE OF ACCESS.

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: decisionTreeBinning
# GitHub: https://github.com/mi3nts/decisiontreeBinning

# IMPORT LIBRARIES
import numpy as np
from sklearn.metrics import r2_score
from functions import decisionTreeBinning, computeQS

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values

# OUTPUTS
#   - optimal_num_bands = optimal number of bands
#   - optimal_band_dict = dictionary where keys are bin numbers and values are band boundaries
#   - clf = sklearn.tree.DecisionTreeRegressor() class. Decision tree model for optimal bands.
#   - min_QS = minimum quality score value

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score
#   - decisionTreeBinning()
#   - computeQS()

# DEPENDERS
#   - plotOptimalBands()

def getOptimalBands(f, ps):

    # define number of observed power/frequnecy values
    numSamples = len(ps)

    # intialize array to store r^2 and adjusted r^2 values
    r2_array = np.zeros((numSamples-1,1))
    fitness_term = np.zeros((numSamples-1,1))
    penalty_term = np.zeros((numSamples-1,1))
    QS_array = np.zeros((numSamples-1,1))

    # define array with bin counts
    band_counts = np.linspace(2,numSamples, numSamples-1).astype(int)

    for i in range(numSamples-1):

        # PART 1: compute bins with decision tree
        bin_dict, clf = decisionTreeBinning(band_counts[i], f, ps)
        ps_pred = np.exp(clf.predict(f.reshape(-1, 1)))

        # PART 2: compute quality score
        r2_array[i] = np.round(r2_score(ps,ps_pred),4)
        QS_array[i], fitness_term[i], penalty_term[i] =  computeQS(r2_array[i], band_counts[i], numSamples)

    # get function outputs
    optimal_num_bands = band_counts[np.argmin(QS_array)]
    optimal_band_dict, clf = decisionTreeBinning(optimal_num_bands, f, ps)
    min_QS = QS_array[np.argmin(QS_array)][0]

    return optimal_num_bands, optimal_band_dict, clf, min_QS

"""
EXAMPLE CODE:
from functions import artificialSpectrum
from getOptimalBands import getOptimalBands

ps, f = artificialSpectrum(1, 30, 150)
optimal_num_bands, optimal_band_dict, clf, min_QS = getOptimalBands(f, ps)
print(optimal_band_dict)
"""
