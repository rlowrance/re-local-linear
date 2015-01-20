# parcels-coded.R
# Create file OUTPUT/parcels-coded.RData

source('Directory.R')
source('InitializeR.R')
source('LUSEI.R')
source('PROPN.R')
source('ReadRawParcels.R')

Control <- function() {
    me <-'parcels-coded'
    
    log <- Directory('log')
    working <- Directory('working')
    
    control <- list(
         path.out.parcels.coded = paste0(working, me, '.RData')
        ,path.out.log = paste0(log, me, '.log')
        ,testing = FALSE
        )
    control
}
Main <- function(control) {
    #cat('starting Main\n'); browser()

    # write control variables
    InitializeR(duplex.output.to = control$path.out.log)
    str(control)

    cat('Starting to read raw parcels (slow)\n')
    parcels.raw <- ReadRawParcels( nrows = if (control$testing) 1000 else 0
                                  ,path.to.input.directory = Directory('input')
                                  ,verbose = TRUE
                                  )
    

    # code every parcel with possible indicaters and its zip5, zip9, and census.tract
    parcels.coded <- data.frame(
          stringsAsFactors = FALSE
         ,is.industry = PROPN(parcels.raw$PROPERTY.INDICATOR.CODE, 'any.industrial')
         ,is.park = LUSEI(parcels.raw$UNIVERSAL.LAND.USE.CODE, 'park')
         ,is.retail = PROPN(parcels.raw$PROPERTY.INDICATOR.CODE, 'retail')
         ,is.school = LUSEI(parcels.raw$UNIVERSAL.LAND.USE.CODE, 'any.school')
         ,APN.FORMATTED = parcels.raw$APN.FORMATTED
         ,APN.UNFORMATTED = parcels.raw$APN.UNFORMATTED
         ,CENSUS.TRACT = parcels.raw$CENSUS.TRACT
         ,ZIPCODE = parcels.raw$PROPERTY.ZIPCODE
         )
    
    str(parcels.coded)
    print(summary(parcels.coded))

    save(control, parcels.coded, file = control$path.out.parcels.coded)
    
    str(control)
    if (control$testing) cat('TESTING: DISCARD OUTPUT\n')
}

# process the data
Main(Control())
cat('done\n')

