# MAIN FILE FOR DECISION TREE BINNING TECHNIQUE. SCRIPT GENERATES CASE STUDY
# PLOTS IN THE ASSOCIATED PAPER: DATA-DRIVEN EEG BAND DISCOVERY WITH DECISION TREES

# CODE AUTHORED BY: SHAWHIN TALEBI
# PROJECT: decisionTreeBinning
# GitHub: https://github.com/mi3nts/decisiontreeBinning

# IMPORT FUNCTIONS
from functions import *

# CASE STUDY 1
name = 'caseStudy1'
ps, f = artificialSpectrum(1, 30, 150)
plotArtificialSpectrum(ps, f)
plotDTB_2to6_bands(f, ps, name)
plotR2vsNumBands(f, ps, name)
plotQSvsNumBands(f, ps, name)
plotOptimalBands(f, ps, name)

# CASE STUDY 2
name = 'caseStudy2'
O, f, ps = getExperimentalData()
plotExperimentalData(O, f, ps)
plotQSvsNumBands(f, ps, name)
plotOptimalBands_withCS1(f, ps, name)
