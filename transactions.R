# transactions.R
# Create files
#    OUTPUT/transactions.Rdata
#    OUTPUT/transactions.csv
# containing all transactions for
# arms-length grant deeds for single-family-residential parcels.

# Join these dataframes and files
# census                in WORKING/census.RData
# deeds-al              in WORKING/deeds-al.RData 
# input/geocoding.tsv
# parcels-sfr           in WORKING/parcels-sfr.RData
# features.census.tract in WORKING/parcels-derived-features.RData
# features.zip5         in WORKING/parcels-derived-features.RData
# features.zip9         in WORKING/parcels-derived-features.RData

# Join these files:
# census.csv   : census records
# deeds-al.csv : selected deeds data for arms-length deeds
# geocoding.tsv : geocodes for residences
# parcels-sfr.csv : selected parcels data for single-family-residences
# parcels-derived-features-census-tract.csv : features of census tracts
# parcels-derived-features-zip5.csv : features of 5-digit zip codes

# Join the deeds and parcels fileson the best APN (defined below), not the
# formatted or unformatted APN.

source('Directory.R')
source('InitializeR.R')

source('BestApns.R')
source('ReadCensus.R')
source('ReadDeedsAlG.R')
source('ReadParcelsSfr.R')
source('ZipN.R')

Control <- function() {
# set control variables
    me <- 'transactions'

    log <- Directory('log')
    input <- Directory('input')
    working <- Directory('working')

    control <- list( path.out.log = paste0(log, me, '.log')
                    ,path.out.transactions.rdata = paste0(working, 'transactions.RData')
                    ,path.out.transactions.csv = paste0(working, 'transactions.csv')
                    ,path.in.census = paste0(working, 'census.RData')
                    ,path.in.deeds.al.g = paste0(working, 'deeds-al-g.RData')
                    ,path.in.geocoding = paste0(input, 'geocoding.tsv')
                    ,path.in.parcels.sfr = paste0(working, 'parcels-sfr.RData')
                    ,path.in.parcels.derived.features = paste0(working, 'parcels-derived-features.RData')
                    ,testing = FALSE
                    )
    control
}

ReadGeocoding <- function(control) {
    # Read geocoding file
    # ARGS:
    # control : list of control variables
    # RETURNS data.frame
    cat('starting ReadGeocoding\n')
    geocoding <- read.table( control$path.in.geocoding
                            ,header=TRUE
                            ,sep="\t"
                            ,quote=""
                            ,comment=""
                            ,stringsAsFactor=FALSE
                            ,na.strings=""
                            )
    cat("number of geocoding records read", nrow(geocoding), "\n")
    str(geocoding)
    geocoding
}


ReadParcelsCensusTract <- function(control) {
    # Read census tract data derived from parcels
    # ARGS:
    # control : list of control vars
    # RETURNS: data.frame
    cat('starting ReadParcelsCensusTract\n')
    df <- read.csv(control$path.parcels.census.tract,
                   check.names=FALSE,
                   header=TRUE,
                   sep=",",
                   quote="",
                   comment.char="",
                   stringsAsFactors=FALSE,
                   nrows=ifelse(control$testing, 1000, -1))                  
    cat("number of parcel df records read", nrow(df), "\n")
    # fix up the column names
    df <- data.frame(census.tract              = df[['"census.tract"']],
                     census.tract.has.industry = df[['"has.industry"']],
                     census.tract.has.park     = df[['"has.park"']],
                     census.tract.has.retail   = df[['"has.retail"']],
                     census.tract.has.school   = df[['"has.school"']])
    str(df)
    df
}

ReadParcelsZip5 <- function(control) {
    # Read census tract data derived from parcels
    # ARGS:
    # control : list of control vars
    # RETURNS: data.frame
    cat('starting ReadParcelsZip5\n')
    df <- read.csv(control$path.parcels.zip5,
                   check.names=FALSE,
                   header=TRUE,
                   sep=",",
                   quote="",
                   comment.char="",
                   stringsAsFactors=FALSE,
                   nrows=ifelse(control$testing, 1000, -1))                  
    cat("number of parcel df records read", nrow(df), "\n")
    # fix up the column names
    df <- data.frame(zip5              = df[['"zip5"']],
                     zip5.has.industry = df[['"has.industry"']],
                     zip5.has.park     = df[['"has.park"']],
                     zip5.has.retail   = df[['"has.retail"']],
                     zip5.has.school   = df[['"has.school"']])
    str(df)
    df
}

BestCensusTract <- function(merged) {
    #cat('start BestCensusTract\n'); browser()
    deeds.census.tract <- merged$CENSUS.TRACT.deeds
    parcels.census.tract <- merged$CENSUS.TRACT.parcels
    is.missing.deeds <- is.na(deeds.census.tract)
    is.missing.parcels <- is.na(parcels.census.tract)
    both.present <- !is.missing.deeds & !is.missing.parcels
    result <- ifelse( both.present
                     ,ifelse( deeds.census.tract == parcels.census.tract
                             ,deeds.census.tract
                             ,NA
                             )
                     ,ifelse( is.missing.deeds
                             ,ifelse( is.missing.parcels
                                     ,NA
                                     ,parcels.census.tract
                                     )
                             ,deeds.census.tract
                             )
                     )
    result
}

BestPropertyZipcode <- function(zipcode.parcels, zipcode.deeds) {
    #cat('start BestPropertyZipcode\n'); browser()
    both.agree <- zipcode.parcels == zipcode.deeds
    is.ok.deeds <- !is.na(zipcode.deeds) & is.numeric(zipcode.deeds)
    is.ok.parcels <- !is.na(zipcode.parcels) & is.numeric(zipcode.parcels)
    result <- ifelse( both.agree
                     ,zipcode.parcels
                     ,ifelse( is.ok.parcels
                             ,zipcode.parcels
                             ,zipcode.deeds
                             )
                     )
    result
}

MergeDeedsParcels <- function(control, deeds.al, parcels.sfr) {
    # merge the deeds and parcels into one data frame
    # ARGS
    # control : list of control variables
    # RETURNS data.frame with deeds, parcels, and best APNs
    #cat('start MergeDeedsParcels\n'); browser()

    # we will merge on the APN fields, so we need to find the best APNs
    deeds.al$apn.recoded <-    BestApns( apns.unformatted = deeds.al$APN.UNFORMATTED
                                        ,apns.formatted = deeds.al$APN.FORMATTED
                                        )
    deeds.al$APN.FORMATTED.deeds <- NULL
    deeds.al$APN.UNFORMATTED.deeds <- NULL
    deeds.al$APN.SEQUENCE.NUMBER.deeds <- NULL

    parcels.sfr$apn.recoded <- BestApns( apns.unformatted = parcels.sfr$APN.UNFORMATTED
                                        ,apns.formatted = parcels.sfr$APN.FORMATTED
                                        )
    parcels.sfr$APN.UNFORMATTED.parcels<- NULL
    parcels.sfr$APN.FORMATTED.parcels <- NULL
    parcels.sfr$APN.SEQUENCE.NUMBER.parcels <- NULL

    # add zip code fields
    zips <- ZipN(parcels.sfr$PROPERTY.ZIPCODE)
    parcels.sfr$zip5 <- zips$zip5
    parcels.sfr$zip9 <- zips$zip9
    parcels.sfr$PROPERTY.ZIPCODE

    # merge deeds and parcels
    #cat('in MergeDeedsParcels; about to merge\n'); browser()
    #cat('about to merge\n'); browser()
    merged <- merge( x=deeds.al
                    ,y=parcels.sfr
                    ,by="apn.recoded"
                    ,suffixes=c(".deeds", ".parcels")
                    )
    cat("number of deeds and parcels with common recoded apn",
        nrow(merged),
        "\n")

    merged
}

MergeCensus <- function(control, census, deeds.parcels) {
    # merge the census data into a data.frame
    # ARGS:
    # df : data frame containing merged deeds and parcels features
    # control : control variables list
    # RETURNS: data.frame augmented with census data
    cat('starting MergeCensus', nrow(df), '\n') 
    #cat('about to merge in MergeCensus\n'); browser()
    merged <- merge( x = deeds.parcels
                    ,by.x = "CENSUS.TRACT"
                    ,y = census
                    ,by.y = "census.tract"
                    )
    cat("number of common deeds and parcels with known CENSUS.TRACT",
        nrow(merged),
        "\n")
    merged
}

MergeGeocoding <- function(control, census.deeds.parcels, geocoding) {
    # merge geocoding data into a data.frame
    # RETURNS: data.frame augmented with latitudes and longitudes
    cat('starting MergeGeocoding', nrow(df), '\n')
    #browser()
    merged <- merge( x = census.deeds.parcels
                    ,by.x = 'apn.recoded'
                    ,y = geocoding
                    ,by.y='G.APN'
                    )
    cat("number of transactions, after considering geocoding",
        nrow(merged),
        "\n")
    merged
}

MergeDerivedFeatures <- function(control, derived.features, transactions) {
    # add in census.tract and zip5 features
    cat('starting MergeDerivedFeatures\n')
    c <- derived.features$features.census.tract
    census.tract <- data.frame( census.tract.has.industry = c$has.industry
                               ,census.tract.has.park     = c$has.park
                               ,census.tract.has.retail   = c$has.retail
                               ,census.tract.has.school   = c$has.school
                               ,census.tract              = c$census.tract
                               )
    merged1 <- merge( x    = transactions
                     ,by.x = 'CENSUS.TRACT'
                     ,y    = census.tract
                     ,by.y = 'census.tract'
                     )

    z <- derived.features$features.zip5
    zip <- data.frame( zip5.has.industry = z$has.industry
                      ,zip5.has.park     = z$has.park
                      ,zip5.has.retail   = z$has.retail
                      ,zip5.has.school   = z$has.school
                      ,zip5              = z$zip5
                      )

    merged2 <- merge( x    = merged1
                     ,by.x = 'zip5'
                     ,y    = zip
                     ,by.y = 'zip5'
                     )
    cat(' number of transactions, after considering features derived from parcels'
        ,nrow(merged2)
        ,'\n'
        )
    merged2
}

ReadDerivedFeatures <- function(control) {
    cat('reading derived features\n')
    features.census.tract <- NULL
    features.zip5 <- NULL
    features.zip9 <- NULL
    loaded <- load(control$path.in.parcels.derived.features)
    stopifnot(!is.null(features.census.tract))
    stopifnot(!is.null(features.zip5))
    stopifnot(is.null(features.zip9))
    result <- list( features.census.tract = features.census.tract
                   ,features.zip5 = features.zip5
                   ,features.zip9 = features.zip9
                   )
    result
}

WriteTransactions <- function(control, transactions, info) {
    cat('starting WriteTransactions', nrow(transactions), '\n')

    # Drop extraneous features
    print(names(transactions))
    transactions$APN.UNFORMATTED.deeds <- NULL
    transactions$APN.FORMATTED.deeds <- NULL
    transactions$APN.UNFORMATTED.parcels <- NULL
    transactions$APN.FORMATTED.parcels <- NULL

    write.csv(transactions, control$path.out.transactions.csv)

    save(control, transactions, info, file = control$path.out.transactions.rdata)


}


# The cascading style is used to make debugging easier in the face of very 
# large input files
# The idea is to incremental build up and debug the final output, caching values
# as we go.
control <- Control()
InitializeR(duplex.output.to = control$path.out.log)
str(control)
deeds.al.g <-
    if (exists('deeds.al.g')) {
        deeds.al.g
    } else {
        cat('reading deeds\n')
        #debug(ReadDeedsAl)
        ReadDeedsAlG(path = control$path.in.deeds.al.g)
    }
parcels.sfr <- 
    if (exists('parcels.sfr')) {
        parcels.sfr
    } else {
        cat('reading parcels\n')
        ReadParcelsSfr(path = control$path.in.parcels.sfr)
    }
deeds.parcels <- 
    if (exists('deeds.parcels')) {
        deeds.parcels
    } else {
        cat('merging deeds and parcels\n')
        #debug(MergeDeedsParcels)
        MergeDeedsParcels(control, deeds.al.g, parcels.sfr)
    }
census <- 
    if (exists('census'))  {
        census
    } else {
        cat('reading census\n')
        #debug(ReadCensus)
        ReadCensus(control$path.in.census)
    }
census.deeds.parcels <-
    if (exists('census.deeds.parcels')) {
        census.deeds.parcels
    } else {
        cat('merging census\n')
        #debug(MergeCensus)
        MergeCensus(control, census, deeds.parcels)
    }
geocoding <-
    if (exists('geocoding')) {
        geocoding
    } else {
        cat('reading geocoding\n')
        #debug(ReadGeocoding)
        ReadGeocoding(control)
    }
census.deeds.parcels.geocoding <-
    if (exists('census.deeds.parcels.geocoding')) {
        census.deeds.parcels.geocoding
    } else {
        cat('merging geocoding\n')
        #debug(MergeGeocoding)
        MergeGeocoding(control, census.deeds.parcels, geocoding)
    }
derived.features <-
    if (exists('derived.features')) {
        derived.features
    } else {
        cat('reading derived features\n')
        #debug(ReadDerivedFeatures)
        ReadDerivedFeatures(control)
    }
transactions <-
    if (exists('transactions')) {
        transactions
    } else {
        cat('merging derived features\n')
        #debug(MergeDerivedFeatures)
        MergeDerivedFeatures(control, derived.features, census.deeds.parcels.geocoding)
    }
#debug(WriteTransactionsAlSfr)
info <- list( nrow.deeds.al.g = nrow(deeds.al.g)
             ,nrow.parcels.sfr = nrow(parcels.sfr)
             ,nrow.deeds.parcels = nrow(deeds.parcels)
             ,nrow.census = nrow(census)
             ,nrow.census.deeds.parcels = nrow(census.deeds.parcels)
             ,nrow.geocoding = nrow(geocoding)
             ,nrow.census.deeds.parcels.geocoding = nrow(census.deeds.parcels.geocoding)
             ,nrow.derived.census.tract = nrow(derived.features$features.census.tract)
             ,nrow.derived.zip5 = nrow(derived.features$features.zip5)
             ,nrow.derived.features = nrow(derived.features)
             ,nrow.transactions = nrow(transactions)
)
cat('writing transactions')
WriteTransactions(control, transactions, info)
str(control)
if (control$testing)
    cat('DISCARD OUTPUT: TESTING\n')
cat('done\n')
