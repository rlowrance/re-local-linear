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

ALL += $(WORKING)/census.RData
ALL += $(WORKING)/deeds-al-g.RData
ALL += $(WORKING)/parcels-coded.RData
ALL += $(WORKING)/parcels-derived-features.RData
ALL += $(WORKING)/parcels-sfr.RData
ALL += $(WORKING)/transactions.RData
ALL += $(WORKING)/transactions-subset1.RData

all: $(ALL)

$(WORKING)/census.RData: $(INPUT_CENSUS) census.R
	Rscript census.R

$(WORKING)/deeds-al-g.RData: $(INPUT_DEEDS) deeds-al-g.R
	Rscript deeds-al-g.R

$(WORKING)/parcels-coded.RData: $(INPUT_TAXROLLS) parcels-coded.R
	RScript parcels-coded.R

$(WORKING)/parcels-derived-features.RData: $(INPUT_TAXROLLS) parcels-derived-features.R
	RScript parcels-derived-features.R

$(WORKING)/parcels-sfr.RData: $(INPUT_TAXROLLS) parcels-sfr.R
	Rscript parcels-sfr.R

$(WORKING)/transactions.RData: $(INPUT_TAXROLLS) transactions.R
	Rscript transactions.R

$(WORKING)/transactions-subset1.RData: $(INPUT_TAXROLLS) transactions-subset1.R
	Rscript transactions-subset1.R

$(WORKING)/transactions-subset1-train.RData: $(INPUT_TAXROLLS) transactions-subset1-train.R
	Rscript transactions-subset1-train.R

# source file dependencies
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
transactions-subset1.R: \
	Directory.R InitializeR.R Printf.R ReadTransactions.R DEEDC.R SCODE.R TRNTP.R
transactions-subset1-train.R: \
	Directory.R InitializeR.R Printf.R ReadTransactionsSubset1.R SplitDate.R
