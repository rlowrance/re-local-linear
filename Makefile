# disable built-in rules
.SUFFIXES:

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

all: $(WORKING)/census.RData $(WORKING)/deeds-al-g.RDATA $(WORKING)/parcels-sfr.RData

$(WORKING)/census.RData: $(INPUT_CENSUS) census.R
	Rscript census.R

$(WORKING)/deeds-al-g.RData: $(INPUT_DEEDS) deeds-al-g.R
	Rscript deeds-al-g.R

$(WORKING)/parcels-sfr.RData: $(INPUT_TAXROLLS) parcels-sfr.R
	Rscript parcels-sfr.R

# source file dependencies
deeds-al-g.R: DEEDC.R Directory.R InitializeR.R PRICATCODE.R
parcels-sfr.R:        Directory.R InitializeR.R LUSEI.R ReadRawParcels.R
