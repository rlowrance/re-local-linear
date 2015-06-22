# disable built-in rules
.SUFFIXES:



PYTHON = ~/anaconda/bin/python

INPUT = ../data/input
WORKING = ../data/working
CVCELL = ../data/working/cv-cell

INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F1.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F2.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F3.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_07/CAC06037F4.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F5.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F6.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F7.zip
INPUT_DEEDS += $(INPUT)/corelogic-deeds-090402_09/CAC06037F8.zip

INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F1.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F2.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F3.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F4.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F5.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F6.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F7.zip
INPUT_TAXROLLS += $(INPUT)/corelogic-taxrolls-090402_05/CAC06037F8.zip

INPUT_CENSUS += $(INPUT)/neighborhood-data/census.csv

#ALL += $(WORKING)/census.RData
#ALL += $(WORKING)/deeds-al-g.RData
#ALL += $(WORKING)/parcels-coded.RData
#ALL += $(WORKING)/parcels-derived-features.RData
#ALL += $(WORKING)/parcels-sfr.RData
#ALL += $(WORKING)/transactions.RData
ALL += $(WORKING)/transactions-subset2.csv
ALL += $(WORKING)/transactions-subset2.pickle
ALL += $(WORKING)/transactions-subset2-test.pickle
ALL += $(WORKING)/transactions-subset2-train.pickle
ALL += $(WORKING)/chart-01.pdf
ALL += $(WORKING)/chart-02-ols-2003on-ct-t-mean-mean.txt
ALL += $(WORKING)/chart-02-ols-2003on-ct-t-mean-wi10.txt
ALL += $(WORKING)/chart-02-ols-2003on-ct-t-median-median.txt
ALL += $(WORKING)/chart-02-ransac-2003on-ct-t-mean-mean.txt
ALL += $(WORKING)/chart-02-ransac-2003on-ct-t-mean-wi10.txt
ALL += $(WORKING)/chart-02-ransac-2003on-ct-t-median-median.txt
ALL += $(WORKING)/chart-02-ols-2008-act-ct-mean-mean.txt
ALL += $(WORKING)/chart-02-ols-2008-act-ct-median-median.txt
ALL += $(WORKING)/chart-02-ransac-2008-act-ct-median-median.txt
ALL += $(WORKING)/chart-03.txt
ALL += $(WORKING)/chart-04.nz-count-all-periods.txt
ALL += $(WORKING)/record-counts.tex
#ALL += $(WORKING)/transactions-subset2-rescaled.csv
#ALL += $(WORKING)/python-dependencies.makefile

all: $(ALL)

$(WORKING)/transactions-subset2.csv: \
	unpickle-transactions-subset2.py $(WORKING)/transactions-subset2.pickle
	python unpickle-transactions-subset2.py \
		< $(WORKING)/transactions-subset2.pickle \
		> $(WORKING)/transactions-subset2.csv

$(WORKING)/transactions-subset2-rescaled.csv: \
	rescale.py $(WORKING)/transactions-subset2.csv
	python rescale.py \
		< $(WORKING)/transactions-subset2.csv \
		> $(WORKING)/transactions-subset2-rescaled.csv
	
# dependencies found in python source files
include $(WORKING)/python-dependencies.makefile

$(WORKING)/python-dependencies.makefile: python-dependencies.py


# Creation of cvcell
$(CVCELL)/%.cvcell:
	$(PYTHON) cv-cell.py $*

# rules for CHARTS
include chart-01.makefile
include chart-02-ols-2003on-ct-t-mean-mean.makefile
include chart-02-ols-2003on-ct-t-mean-wi10.makefile
include chart-02-ols-2003on-ct-t-median-median.makefile
include chart-02-ransac-2003on-ct-t-mean-mean.makefile
include chart-02-ransac-2003on-ct-t-mean-wi10.makefile
include chart-02-ransac-2003on-ct-t-median-median.makefile
include chart-02-ols-2008-act-ct-mean-mean.makefile
include chart-02-ols-2008-act-ct-median-median.makefile
include chart-02-ransac-2008-act-ct-median-median.makefile
include chart-03.makefile
#include chart-04.makefile
#include chart-04.makefile
#include chart-05.makefile

# rules for other *.makefile files
chart-02-ols-2003on-ct-t-mean-mean.makefile: \
	chart-02-ols-2003on-ct-t-mean-mean.py
	python chart-02-ols-2003on-ct-t-mean-mean.py makefile

chart-02-ols-2003on-ct-t-mean-wi10.makefile: \
	chart-02-ols-2003on-ct-t-mean-wi10.py
	python chart-02-ols-2003on-ct-t-mean-wi10.py makefile

chart-02-ols-2003on-ct-t-median-median.makefile: \
	chart-02-ols-2003on-ct-t-median-median.py
	python chart-02-ols-2003on-ct-t-median-median.py makefile

chart-02-ransac-2003on-ct-t-mean-mean.makefile: \
	chart-02-ransac-2003on-ct-t-mean-mean.py
	python chart-02-ransac-2003on-ct-t-mean-mean.py makefile

chart-02-ransac-2003on-ct-t-mean-wi10.makefile: \
	chart-02-ransac-2003on-ct-t-mean-wi10.py
	python chart-02-ransac-2003on-ct-t-mean-wi10.py makefile

chart-02-ransac-2003on-ct-t-median-median.makefile: \
	chart-02-ransac-2003on-ct-t-median-median.py
	python chart-02-ransac-2003on-ct-t-median-median.py makefile

chart-02-ols-2008-act-ct-mean-mean.makefile: \
  chart-02-ols-2008-act-ct-mean-mean.py 
	python chart-02-ols-2008-act-ct-mean-mean.py makefile

chart-02-ols-2008-act-ct-median-median.makefile: \
  chart-02-ols-2008-act-ct-median-median.py 
	python chart-02-ols-2008-act-ct-median-median.py makefile

chart-02-ransac-2008-act-ct-median-median.makefile: \
  chart-02-ransac-2008-act-ct-median-median.py 
	python chart-02-ransac-2008-act-ct-median-median.py makefile

#chart-02-huber100-median-of-root-median-squared-errors.makefile: \
#  chart-02-huber100-median-of-root-median-squared-errors.py 
#	python chart-02-huber100-median-of-root-median-squared-errors.py makefile
#
#chart-02-theilsen-median-of-root-median-squared-errors.makefile: \
#  chart-02-theilsen-median-of-root-median-squared-errors.py 
#	python chart-02-theilsen-median-of-root-median-squared-errors.py makefile

#chart-04.makefile: \
#	chart-04.py
#	python chart-04.py makefile

# GENERATED TEX FILES

$(WORKING)/record-counts.tex: \
$(WORKING)/parcels-sfr-counts.csv \
$(WORKING)/deeds-al-g-counts.csv \
$(WORKING)/transactions-counts.csv \
$(WORKING)/transactions-subset2-counts.csv \
record-counts.py
	$(PYTHON) record-counts.py


# DATA
$(WORKING)/census.RData: $(INPUT_CENSUS) census.R
	Rscript census.R

$(WORKING)/deeds-al-g%RData \
$(WORKING)/deeds-al-g-counts%csv \
: $(INPUT_DEEDS) deeds-al-g.R
	Rscript deeds-al-g.R

$(WORKING)/parcels-coded.RData: $(INPUT_TAXROLLS) parcels-coded.R
	RScript parcels-coded.R

$(WORKING)/parcels-derived-features.RData: $(INPUT_TAXROLLS) parcels-derived-features.R
	RScript parcels-derived-features.R

$(WORKING)/parcels-sfr%RData \
$(WORKING)/parcels-sfr-counts%csv \
: $(INPUT_TAXROLLS) parcels-sfr.R
	Rscript parcels-sfr.R

$(WORKING)/transactions%RData $(WORKING)/transactions%csv:\
	$(WORKING)/census.RData \
	$(WORKING)/deed-al-g.RData \
	$(INPUT)/geocoding.tsv \
	$(WORKING)/parcels-derived-features.RData \
	$(WORKING)/parcels-sfr.RData \
	transactions.R
	Rscript transactions.R

$(WORKING)/transactions-subset2.pickle \
: $(WORKING)/transactions.csv transactions-subset2.py
	$(PYTHON) transactions-subset2.py

$(WORKING)/transactions-subset2-test%pickle \
$(WORKING)/transactions-subset2-train%pickle \
: $(WORKING)/transactions-subset2.pickle transactions-subset2-test.py
	$(PYTHON) transactions-subset2-test.py


# source file dependencies R language
census.R: \
	Directory.R InitializeR.R
deeds-al-g.R: \
	Directory.R InitializeR.R DEEDC.R Printf.R PRICATCODE.R
parcels-coded.R: \
	Directory.R InitializeR.R LUSEI.R PROPN.R ReadRawParcels.R
parcels-derived-features.R: \
	Directory.R InitializeR.R Methods.R Clock.R LUSEI.R Printf.R PROPN.R ReadParcelsCoded.R ZipN.R
parcels-sfr.R: \
	Directory.R InitializeR.R LUSEI.R Printf.R ReadRawParcels.R
transactions.R: \
	Directory.R InitializeR.R BestApns.R ReadCensus.R ReadDeedsAlG.R ReadParcelsSfr.R ZipN.R
