# FUNCTION FILE FOR DECISION TREE BINNING TECHNIQUE

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: decisionTreeBinning
# GitHub: https://github.com/mi3nts/decisiontreeBinning

# IMPORT LIBRARIES
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

from pyedflib import highlevel
from scipy import signal

# ==============================================================================
# artificialSpectrum()

# FUNCTION TO GENERATE A ARTIFICIAL EEG POWER SPECTRUM WITH CHARACTERISTIC 1/f
# SHAPE WITH ADDED WHITE NOISE

# INPUTS
#   - f_start = first frequnecy value
#   - f_end = last frequnecy value
#   - n_freqs = total number of frequnecy values

# OUTPUTS
#   - ps = numpy array of artificial power values
#   - f = numpy array of frequnecy values

# DEPENDENCIES
#   - numpy

# DEPENDERS
#   - none

def artificialSpectrum(f_start, f_end, n_freqs):
    # define range of frequencies
    f = np.linspace(f_start, f_end, n_freqs)

    # 1/f term
    invf = 1/f
    # uniformly distrbuted random noise term
    np.random.seed(10)
    n = 0.04*np.random.rand(len(f))

    # sum terms
    ps = invf + n

    return ps, f
# ==============================================================================
# plotArtificialSpectrum()

# FUNCTION TO PLOT AND SAVE INPUT POWER SPECTRUM

# INPUTS
#   - ps = numpy array of artificial power values
#   - f = numpy array of frequnecy values

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - matplotlib.pyplot

# DEPENDERS
#   - none

def plotArtificialSpectrum(ps, f):
    # set plot parameters
    font_size = 24
    lw = 3
    plt.rcParams.update({'font.size': font_size})

    # plot power spectrum
    plt.plot(f,ps, linewidth=lw)
    plt.title("Artificial Power Spectrum", fontweight='bold')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")

    # adjust figure
    fig = plt.gcf()
    fig.set_figheight(6)
    fig.set_figwidth(12)
    fig.tight_layout(pad=3.0)
    plt.grid()

    # save figure
    plt.savefig("visuals/artificialSpectrum.png", facecolor='white', transparent=True)
# ==============================================================================
# decisionTreeBinning()

# IMPLEMENTATION OF DECISION TREE BASED FREQUENCY BINNING TECHNIQUE

# INPUTS
#   - num_bins = integer value (>2) of desired number of bins
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values

# OUTPUTS
#   - bin_dict = dictionary where keys are bin numbers and values are bin boundaries
#   - clf = sklearn.tree.DecisionTreeRegressor() class. Decision tree model.

# DEPENDENCIES
#   - numpy
#   - pandas
#   - sklearn.tree.DecisionTreeRegressor()

# DEPENDERS
#   - plotDTB_2to6_bands()
#   - plotR2vsNumBands()
#   - getOptimalBands

def decisionTreeBinning(num_bins, f, ps):

    # fitting the regression tree X as features/predictor and y as label/target
    clf = DecisionTreeRegressor(max_leaf_nodes = num_bins).fit(f.reshape(-1, 1), np.log(ps))

    # variables creation
    num_nodes = clf.tree_.node_count
    left_child = clf.tree_.children_left
    right_child = clf.tree_.children_right
    threshold = clf.tree_.threshold

    # list to store the bin edges
    bin_edges = [f[0],f[len(f)-1]]

    # loop through all the nodes
    for i in range(num_nodes):
        # If the left and right child of a node is not the same(-1) we have an internal node
        # which we will append to bin_node list
        if left_child[i]!=right_child[i]:
            bin_edges.append(np.round(threshold[i],1))

    # sort the nodes in increasing order
    bin_edges.sort()

    # create dictionary to store epoch bin edges
    bin_dict = {}

    # put in each dictionary index 2 consecutive bin edges
    for i in range(num_bins):
        bin_dict[str(i+1)] = [bin_edges[i], bin_edges[i+1]]

    return bin_dict, clf
# ==============================================================================
# computeBandR2()

# FUNCTION TO COMPUTE r^2 REGRESSION SCORE FOR A SET OF BIN BOUNDARIES

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values
#   - band_dict = dictionary where keys are bin numbers and values are bin boundaries

# OUTPUTS
#   - r2 = r^2 regression score of input bin boundaries in reproducing true power spectrum

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score

# DEPENDERS
#   - plotDTB_2to6_bands()

def computeBandR2(f, ps, band_dict):

    numBands = len(band_dict)
    ps_est = []

    for i in range(numBands):
        f_band =  list(band_dict.values())[i]

        band_powers = [ps[j] for j in range(len(f)) if f[j] >= f_band[0] and f[j] <= f_band[1]]
        mean_power = np.mean(band_powers)
        ps_est = np.append(ps_est, np.ones((len(band_powers), 1))* mean_power)

    if len(ps)!=len(ps_est):
        ps = ps[(len(ps)-len(ps_est)):len(ps_est)+1]

    r2 = r2_score(ps,ps_est)

    return r2

# ==============================================================================
# plotDTB_2to6_bands()

# FUNCTION TO GENERATE DECISION TREE BASED BANDS ON INPUT POWER SPECTRUM. BANDS
# FOR EACH CASE ARE PLOTTED AND SAVED TO FILE.

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values
#   - name = string. unique name for visalization filename
#       ~ example: name = "artificialSpectrum"

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score
#   - matplotlib.pyplot
#   - decisionTreeBinning()
#   - computeBandR2()

# DEPENDERS
#   - none

def plotDTB_2to6_bands(f, ps, name):

    # define bin sizes
    bin_sizes = [2, 3, 4, 5, 6]
    # intialize array to store r^2 values
    r2_array = np.zeros((len(bin_sizes),1))

    plt.figure(figsize=(18, 20))
    for num_bins in bin_sizes:

        title_fonts=20
        bin_edge_fonts=16
        other_fonts=24
        line_width = 3

        # compute bins with decision tree
        bin_dict, clf = decisionTreeBinning(num_bins, f, ps)
        ps_pred = np.exp(clf.predict(f.reshape(-1, 1)))

        plt.subplot(len(bin_sizes)+1, 1, num_bins-1)

        r2_array[num_bins-2] = np.round(r2_score(ps,ps_pred),2)
        plt.title('%s bands (r$^2$ = %s)\n ' % (num_bins, str(r2_array[num_bins-2][0])), fontsize=title_fonts)
        plt.plot(f,ps, linewidth=line_width)
        plt.plot(f,ps_pred, '-.', linewidth=line_width)
        plt.plot([f[0], f[0]], [0, np.max(ps)], 'r--', linewidth=line_width)
        plt.grid()
        plt.text(f[0], 1.1*np.max(ps), str(f[0]), color='r', fontsize=bin_edge_fonts, horizontalalignment='center')

        plt.xticks(fontsize= other_fonts)
        plt.yticks(fontsize= other_fonts)

        for i in bin_dict:
            edge = bin_dict[i][1]
            plt.plot([edge, edge], [0, np.max(ps)], 'r--', linewidth=line_width)
            plt.text(edge, 1.1*np.max(ps), str(edge), color='r', fontsize=bin_edge_fonts, horizontalalignment='center')
            plt.ylabel("Power", fontsize=other_fonts)

            if num_bins == bin_sizes[0]:
                plt.legend(["True Power", "Tree Estimation", "Band Edges"], fontsize=16, loc='upper center', bbox_to_anchor=(0.65, 1.05), ncol=2)

    # plot "standard" bins
    sbins_dict = {'1': [1, 3.5], '2': [3.5, 7.5], '3': [7.5, 13], '4': [13, 30]}

    plt.subplot(len(bin_sizes)+1, 1, len(bin_sizes)+1)
    plt.title('Standard 4 bands (r$^2$ = %s)\n ' % str(np.round(computeBandR2(f, ps, sbins_dict), 2)), fontsize=title_fonts)
    plt.plot(f,ps, linewidth=line_width)
    plt.plot([f[0], f[0]], [0, np.max(ps)], 'b--', linewidth=line_width)
    plt.grid()
    plt.text(f[0], 1.1*np.max(ps), str(f[0]), color='b', fontsize=bin_edge_fonts, horizontalalignment='center')
    for i in sbins_dict:
        edge = sbins_dict[i][1]
        plt.plot([edge, edge], [0, np.max(ps)], 'b--', linewidth=line_width)
        plt.text(edge, 1.1*np.max(ps), str(edge), color='b', fontsize=bin_edge_fonts, horizontalalignment='center')
        plt.ylabel("Power", fontsize=other_fonts)
        plt.xticks(fontsize= other_fonts)
        plt.yticks(fontsize= other_fonts)

    # xlabel
    plt.xlabel("Frequency", fontsize=20)

    fig = plt.gcf()
    fig.tight_layout(pad=3.0)
    #     fig.set_size_inches(21, 10)
    plt.savefig("visuals/" + name + "_2to6_bands.png", facecolor='white', transparent=True)

# ==============================================================================
# plotR2vsNumBands()

# FUNCTION TO PLOT R^2 REGRESSION SCORE VERSUS NUMBER OF BANDS. FUNCTION SAVES
# FIGURE TO FILE

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values
#   - name = string. unique name for visalization filename
#       ~ example: name = "artificialSpectrum"

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score
#   - matplotlib.pyplot
#   - matplotlib
#   - decisionTreeBinning()

# DEPENDERS
#   - none

def plotR2vsNumBands(f, ps, name):

    # define max number of bands
    max_num_bands = len(ps)

    # intialize array to store r^2 and adjusted r^2 values
    r2_array = np.zeros((max_num_bands-1,1))

    # define array with bin counts
    band_counts = np.linspace(2,max_num_bands, max_num_bands-1).astype(int)

    plt.figure(figsize=(16, 8))
    for i in range(max_num_bands-1):

        # compute bins with decision tree
        bin_dict, clf = decisionTreeBinning(band_counts[i], f, ps)
        ps_pred = np.exp(clf.predict(f.reshape(-1, 1)))

        r2_array[i] = np.round(r2_score(ps,ps_pred),4)

    plt.semilogx([0, max_num_bands], [1, 1], 'k--', linewidth=1.5)
    plt.semilogx(band_counts, r2_array, linewidth=3)

    steps = [4,6,8,11,36];
    viridis = mpl.cm.get_cmap('inferno', 128)

    for i in range(len(steps)):
        plt.semilogx([steps[i], steps[i]], [0.5, 1], linestyle='-.', color=viridis(i/(len(steps)+1)), linewidth=1.25)
        plt.text(steps[i], 1.02, "n = " + str(steps[i]), color=viridis(i/(len(steps)+1)), fontsize=16, horizontalalignment='center')

    # plt.plot(ar2_array, linewidth=2)
    plt.xlabel('Number of bands (n)')
    plt.ylabel('r$^2$')
    plt.ylim([0.5, 1.05])
    plt.grid(alpha=0.5)
    plt.legend(["r$^2$ = 1", "r$^2$ score"])
    plt.savefig("visuals/"+ name +"_r2-numBands.png", facecolor='white', transparent=True)


# ==============================================================================
# computeQS()

# FUNCTION TO COMPUTE THE QUALITY SCORE FOR A GIVEN CHOICE OF BAND BOUNDARIES

# INPUTS
#   - r2 = r^2 regression score of band boundaries in reproducing true power spectrum
#   - numBands = number of bands used
#   - numSamples = total number of observed power/frequnecy values

# OUTPUTS
#   - quality_score = QS corresponding choice of bands
#   - fitness_term = value of r^2-based fitness term
#   - penalty_term = value of band count penalty term

# DEPENDENCIES
#   - numpy

# DEPENDERS
#   - plotQSvsNumBands()
#   - getOptimalBands()

def computeQS(r2, numBins, numSamples):

    fitness_term = -np.log(r2)
    penalty_term = 2*numBins/numSamples

    quality_score = fitness_term + penalty_term

    return quality_score, fitness_term, penalty_term

# ==============================================================================
# plotQSvsNumBands()

# FUNCTION TO PLOT QUALITY SCORE VERSUS NUMBER OF BANDS. FUNCTION SAVES FIGURE
# TO FILE

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values
#   - name = string. unique name for visalization filename
#       ~ example: name = "artificialSpectrum"

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score
#   - matplotlib.pyplot
#   - matplotlib
#   - decisionTreeBinning()
#   - computeQS()

# DEPENDERS
#   - none

def plotQSvsNumBands(f, ps, name):

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

        # compute bins with decision tree
        bin_dict, clf = decisionTreeBinning(band_counts[i], f, ps)
        ps_pred = np.exp(clf.predict(f.reshape(-1, 1)))

        r2_array[i] = np.round(r2_score(ps,ps_pred),4)
        QS_array[i], fitness_term[i], penalty_term[i] =  computeQS(r2_array[i], band_counts[i], numSamples)

    plt.figure(figsize=(16, 8))
    plt.semilogx(band_counts,fitness_term, '--', linewidth=2)
    plt.semilogx(band_counts,penalty_term, '--', linewidth=2)
    plt.semilogx(band_counts, QS_array, linewidth=3)
    plt.scatter(band_counts[np.argmin(QS_array)], QS_array[np.argmin(QS_array)], s=300, color='y', marker='*', linewidths=1.5, edgecolors='k')
    plt.xlabel('Number of bands')
    plt.ylabel('Quality Score')
    plt.grid(alpha=0.5)
    plt.legend(['Fitness Term', 'Penalty Term', 'Quality Score', "Minimum (" + str(band_counts[np.argmin(QS_array)]) + " bands)"])
    plt.savefig("visuals/"+name+"_QS-numBands.png", facecolor='white', transparent=True)

# ==============================================================================
# getOptimalBands

# FUNCTION TO DERIVE OPTIMAL BAND FOR AN INPUT POWER SPECTRUM USING A TWO-PART
# METHOD. FIRST, DECISION TREE IS USED TO DERIVE OPTIMAL BANDS FOR EVERY CHOICE
# OF BAND COUNT. SECOND, BEST CHOICE OF BAND COUNT IS PICK BASED ON THE QUALITY
# SCORE

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
# ==============================================================================
# plotOptimalBands()

# FUNCTION TO PLOT PLOT OPTIMAL BANDS AND COMPARE THEM TO STANDARD DELTA, THETA,
# ALPHA, AND BETA BANDS

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values
#   - name = string. unique name for visalization filename
#       ~ example: name = "artificialSpectrum"

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score
#   - matplotlib.pyplot
#   - matplotlib
#   - decisionTreeBinning()
#   - computeQS()
#   - getOptimalBands()

# DEPENDERS
#   - none

def plotOptimalBands(f, ps, name):

    # define plotting parameters
    title_fonts=24
    bin_edge_fonts=18
    other_fonts=24
    line_width = 3

    # compute optimal bands
    optimal_num_bands, optimal_band_dict, clf, min_QS = getOptimalBands(f, ps)

    # compute r2 score
    ps_pred = np.exp(clf.predict(f.reshape(-1, 1)))
    r2 = r2_score(ps,ps_pred)

    plt.figure(figsize=(24, 10))
    plt.subplot(2, 1, 1)
    plt.title('Discovered %s bands (r$^2$ = %s, QS = %s)\n ' % (optimal_num_bands, str(np.round(r2,2)), str(np.round(min_QS,2))), fontsize=title_fonts)
    plt.plot(f,ps, linewidth=line_width)
    plt.plot([f[0], f[0]], [0, np.max(ps)], 'r--', linewidth=line_width)
    plt.text(f[0], 1.1*np.max(ps), str(f[0]), color='r', fontsize=bin_edge_fonts, horizontalalignment='center')
    plt.grid(alpha=0.5)
    plt.xticks(fontsize= other_fonts)
    plt.yticks(fontsize= other_fonts)

    for i in optimal_band_dict:
        edge = optimal_band_dict[i][1]
        plt.plot([edge, edge], [0, np.max(ps)], 'r--', linewidth=line_width)
        plt.text(edge, 1.1*np.max(ps), str(edge), color='r', fontsize=bin_edge_fonts, horizontalalignment='center')
        plt.ylabel("Power", fontsize=other_fonts)

    plt.legend(["Power", "Discovered Boundaries"], fontsize=other_fonts, loc='upper center', bbox_to_anchor=(0.75, 1))

    # plot "standard" bins
    sbins_dict = {'1': [1, 3.5], '2': [3.5, 7.5], '3': [7.5, 13], '4': [13, 30]}

    r2 = computeBandR2(f, ps, sbins_dict)
    QS, fitness_term, penalty_term =  computeQS(r2, len(sbins_dict), len(ps))

    plt.subplot(2, 1, 2)
    plt.title('Standard 4 bands (r$^2$ = %s, QS = %s)\n ' % (str(np.round(r2,2)), str(np.round(QS,2))), fontsize=title_fonts)
    plt.plot(f,ps, linewidth=line_width)
    plt.plot([f[0], f[0]], [0, np.max(ps)], 'b--', linewidth=line_width)
    plt.text(f[0], 1.1*np.max(ps), str(f[0]), color='b', fontsize=bin_edge_fonts, horizontalalignment='center')
    for i in sbins_dict:
        edge = sbins_dict[i][1]
        plt.plot([edge, edge], [0, np.max(ps)], 'b--', linewidth=line_width)
        plt.text(edge, 1.1*np.max(ps), str(edge), color='b', fontsize=bin_edge_fonts, horizontalalignment='center')
        plt.ylabel("Power", fontsize=other_fonts)
        plt.xticks(fontsize= other_fonts)
        plt.yticks(fontsize= other_fonts)

    # xlabel
    plt.xlabel("Frequency", fontsize=title_fonts)
    plt.legend(["Power", "Standard Boundaries"], fontsize=other_fonts, loc='upper center', bbox_to_anchor=(0.75, 1))
    plt.grid(alpha=0.5)

    fig = plt.gcf()
    fig.tight_layout(pad=3.0)

    plt.savefig("visuals/" + name + "_optimal-standard-compare.png", facecolor='white', transparent=True)

# ==============================================================================
# getExperimentalData()

# FUNCTION TO LOAD AND PREPARE EXPERIMENTAL DATA
# DATA FROM PhysioNet: EEG During Mental Arithmetic Tasks

# INPUTS
#   - n/a (data is grabbed from data/ subdirectory. ensure file exists)

# OUTPUTS
#   - O = aggregated occiptal signal from subject 00's baseline recording.
#   - f = numpy array of frequnecy values
#   - p = numpy array of power spectral density values corresponding to the
#       frequnecy values in f.

# DEPENDENCIES
#   - pyedflib.highlevel
#   - scipy.signal
#   - numpy

# DEPENDERS
#   - none

def getExperimentalData():
    # read an edf file
    signals, signal_headers, header = highlevel.read_edf('data/Subject00_1.edf')
    fs = signal_headers[0]['sample_frequency']

    # create aggregated occiptal signal
    O = signals[14:16,:]
    O = np.mean(O, axis=0)

    # compute power spectrum using welch method
    f, Pxx_den = signal.welch(O, fs, nperseg=1028)

    # redefine signal to range from about 1 -- 30 Hz
    istart= 2
    iend = 62
    f = f[istart:iend]
    p = Pxx_den[istart:iend]

    return O, f, p
# ==============================================================================
# plotExperimentalData()

# FUNCTION TO PLOT EXPERIMENTAL DATA. BOTH TIME SERIES AND POWER SPECTRUM ARE
# PLOTTED

# INPUTS
#   - O = aggregated occiptal signal from subject 00's baseline recording.
#   - f = numpy array of frequnecy values
#   - p = numpy array of power spectral density values corresponding to the
#       frequnecy values in f.

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - matplotlib.pyplot

# DEPENDERS
#   - none

def plotExperimentalData(O, f, p):

    plt.figure(figsize=(24, 10))
    font_size = 24
    lw = 3
    plt.rcParams.update({'font.size': font_size})

    plt.subplot(2,1,1)
    plt.plot(O, 'k')
    plt.xlabel('Time Index')
    plt.ylabel('Amplitude')
    plt.grid(alpha=0.5)
    plt.subplot(2,1,2)
    plt.plot(f,p,linewidth=lw)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(alpha=0.5)

    fig = plt.gcf()
    fig.tight_layout(pad=3.0)

    plt.savefig("visuals/experimentalData.png", facecolor='white', transparent=True)

# ==============================================================================
# plotOptimalBands_withCS1()

# FUNCTION TO PLOT PLOT OPTIMAL BANDS AND COMPARE THEM TO STANDARD DELTA, THETA,
# ALPHA, AND BETA BANDS

# INPUTS
#   - f = numpy array of frequnecy values
#   - ps = numpy array of power values
#   - name = string. unique name for visalization filename
#       ~ example: name = "artificialSpectrum"

# OUTPUTS
#   - n/a (plot is saved to visuals/ subdirectory)

# DEPENDENCIES
#   - numpy
#   - sklearn.metrics.r2_score
#   - matplotlib.pyplot
#   - matplotlib
#   - decisionTreeBinning()
#   - computeQS()
#   - getOptimalBands()

# DEPENDERS
#   - none

def plotOptimalBands_withCS1(f, ps, name):

    # define plotting parameters
    title_fonts=24
    bin_edge_fonts=18
    other_fonts=24
    line_width = 3

    # compute optimal bands
    optimal_num_bands, optimal_band_dict, clf, min_QS = getOptimalBands(f, ps)

    # compute r2 score
    ps_pred = np.exp(clf.predict(f.reshape(-1, 1)))
    r2 = r2_score(ps,ps_pred)

    plt.figure(figsize=(24, 15))
    plt.subplot(3, 1, 1)
    plt.title('Discovered %s bands (r$^2$ = %s, QS = %s)\n ' % (optimal_num_bands, str(np.round(r2,2)), str(np.round(min_QS,2))), fontsize=title_fonts)
    plt.plot(f,ps, linewidth=line_width)
    plt.plot([f[0], f[0]], [0, np.max(ps)], 'r--', linewidth=line_width)
    plt.text(f[0], 1.1*np.max(ps), str(np.round(f[0],1)), color='r', fontsize=bin_edge_fonts, horizontalalignment='center')
    plt.grid(alpha=0.5)
    plt.xticks(fontsize= other_fonts)
    plt.yticks(fontsize= other_fonts)

    for i in optimal_band_dict:
        edge = optimal_band_dict[i][1]
        plt.plot([edge, edge], [0, np.max(ps)], 'r--', linewidth=line_width)
        plt.text(edge, 1.1*np.max(ps), str(np.round(edge,1)), color='r', fontsize=bin_edge_fonts, horizontalalignment='center')
        plt.ylabel("Power", fontsize=other_fonts)

    plt.legend(["Power", "Discovered Boundaries"], fontsize=other_fonts, loc='upper center', bbox_to_anchor=(0.75, 1))

    # ---------------------------------
    # plot optimal bands from artifical dataset
    bin_dict = {'1': [1.0, 1.9],
     '2': [1.9, 3.2],
     '3': [3.2, 7.3],
     '4': [7.3, 13.0],
     '5': [13.0, 17.1],
     '6': [17.1, 30.0]}

    r2 = computeBandR2(f, ps, bin_dict)
    QS, fitness_term, penalty_term =  computeQS(r2, len(bin_dict), len(f))

    plt.subplot(3, 1, 2)
    plt.title('Case Study 1 Discovered Bands (r$^2$ = %s, QS = %s)\n ' % (str(np.round(r2,2)), str(np.round(QS,2))), fontsize=title_fonts)
    plt.plot(f,ps, linewidth=line_width)
    plt.plot([f[0], f[0]], [0, np.max(ps)], 'g--', linewidth=line_width)
    plt.text(f[0], 1.1*np.max(ps), str(np.round(f[0],1)), color='g', fontsize=bin_edge_fonts, horizontalalignment='center')
    for i in bin_dict:
        edge = bin_dict[i][1]
        plt.plot([edge, edge], [0, np.max(ps)], 'g--', linewidth=line_width)
        plt.text(edge, 1.1*np.max(ps), str(edge), color='g', fontsize=bin_edge_fonts, horizontalalignment='center')
        plt.ylabel("Power", fontsize=other_fonts)
        plt.xticks(fontsize= other_fonts)
        plt.yticks(fontsize= other_fonts)

    # xlabel
    plt.xlabel("Frequency", fontsize=20)
    plt.legend(["Power", "CS1 Discovered Boundaries"], fontsize=other_fonts, loc='upper center', bbox_to_anchor=(0.6, 1))
    plt.grid(alpha=0.5)

    # ---------------------------------
    # plot "standard" bins
    sbins_dict = {'1': [1, 3.5], '2': [3.5, 7.5], '3': [7.5, 13], '4': [13, 30]}

    r2 = computeBandR2(f, ps, sbins_dict)
    QS, fitness_term, penalty_term =  computeQS(r2, len(sbins_dict), len(ps))

    plt.subplot(3, 1, 3)
    plt.title('Standard 4 bands (r$^2$ = %s, QS = %s)\n ' % (str(np.round(r2,2)), str(np.round(QS,2))), fontsize=title_fonts)
    plt.plot(f,ps, linewidth=line_width)
    plt.plot([f[0], f[0]], [0, np.max(ps)], 'b--', linewidth=line_width)
    plt.text(f[0], 1.1*np.max(ps), str(np.round(f[0],1)), color='b', fontsize=bin_edge_fonts, horizontalalignment='center')
    for i in sbins_dict:
        edge = sbins_dict[i][1]
        plt.plot([edge, edge], [0, np.max(ps)], 'b--', linewidth=line_width)
        plt.text(edge, 1.1*np.max(ps), str(edge), color='b', fontsize=bin_edge_fonts, horizontalalignment='center')
        plt.ylabel("Power", fontsize=other_fonts)
        plt.xticks(fontsize= other_fonts)
        plt.yticks(fontsize= other_fonts)

    # xlabel
    plt.xlabel("Frequency", fontsize=title_fonts)
    plt.legend(["Power", "Standard Boundaries"], fontsize=other_fonts, loc='upper center', bbox_to_anchor=(0.75, 1))
    plt.grid(alpha=0.5)

    fig = plt.gcf()
    fig.tight_layout(pad=3.0)

    plt.savefig("visuals/" + name + "_optimal-cs1-standard-compare.png", facecolor='white', transparent=True)
