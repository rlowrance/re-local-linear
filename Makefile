# disable built-in rules
.SUFFIXES:

PYTHON = ~/anaconda/bin/python

INPUT = ../data/input
WORKING = ../data/working

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
ALL += $(WORKING)/transactions-subset2.pickle
ALL += $(WORKING)/transactions-subset2-test.pickle
ALL += $(WORKING)/transactions-subset2-train.pickle
ALL += $(WORKING)/chart-01.pdf
ALL += $(WORKING)/record-counts.tex



all: $(ALL)

# CHARTS
$(WORKING)/chart-01.pdf: $(WORKING)/chart-01-data.pickle chart-01.py
	$(PYTHON) chart-01.py

$(WORKING)/chart-01-data.pickle: $(WORKING)/transactions-subset2.pickle chart-01-data.py
	$(PYTHON) chart-01-data.py

# GENERATED TEX FILES

$(WORKING)/record-counts.tex: \
	$(WORKING)/parcels-sfr-counts.csv \
	record-counts.py
	$(PYTHON) record-counts.py


# DATA
$(WORKING)/census.RData: $(INPUT_CENSUS) census.R
	Rscript census.R

$(WORKING)/deeds-al-g.RData: $(INPUT_DEEDS) deeds-al-g.R
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
$(WORKING)/transactions-subset2.csv \
: $(WORKING)/transactions.csv transactions-subset2.py
	$(PYTHON) transations-subset2.py

$(WORKING)/transactions-subset2-test%pickle \
$(WORKING)/transactions-subset2-train%pickle \
: $(WORKING)/transactions-subset2.pickle transactions-subset2-test.py
	$(PYTHON) transactions-subset2-test.py

# source file dependencies charts
chart-01.py:      directory.py Logger.py
chart-01-data.py: directory.py Logger.py

# source file dependencies basic data
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
transactions-subset2.py: \
	directory.py features.py year_month_day.py Logger.py SCODE.py TRNTP.py
transactions-subset2-test.py: \
	directory.py Logger.py
