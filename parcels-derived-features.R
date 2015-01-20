# parcels-derived-features.R
# Create files OUTPUT/parcels-derived-features.RData

source('Directory.R')
source('Libraries.R')

source('LUSEI.R')
source('PROPN.R')
source('ReadParcelsCoded.R')
source('ZipN.R')

Control <- function() {
    me <-'parcels-derived-features'
    
    log <- Directory('log')
    working <- Directory('working')
    
    control <- list(
         path.in.parcels.coded = paste0(working, 'parcels-coded.RData')
        ,path.out.features = paste0(working, me, '.RData')
        ,path.out.log = paste0(log, me, '.log')
        ,testing = FALSE
        ,do.analysis = TRUE
        )
    control
}
FeaturesCensusTract <- function(control, parcels.coded) {
    # return data frame, one row for each census tract
    cat('start FeaturesCensusTract\n')
    #browser()
    clock <- Clock()
    HasIndustry <- function(census.tract) {
        is.selected <- parcels.coded$census.tract == census.tract
        result <- any(parcels.coded$is.industry[is.selected])
        result
    }
    HasPark <- function(census.tract) {
        is.selected <- parcels.coded$census.tract == census.tract
        result <- any(parcels.coded$is.park[is.selected])
        result
    }
    HasRetail <- function(census.tract) {
        is.selected <- parcels.coded$census.tract == census.tract
        result <- any(parcels.coded$is.retail[is.selected])
        result
    }
    HasSchool <- function(census.tract) {
        is.selected <- parcels.coded$census.tract == census.tract
        result <- any(parcels.coded$is.school[is.selected])
        result
    }
    uct <- unique(parcels.coded$census.tract)
    result <- data.frame( stringsAsFactors = false
                         ,has.industry = sapply(uct, HasIndustry)
                         ,has.park = sapply(uct, HasPark)
                         ,has.retail = sapply(uct, HasRetail)
                         ,has.school = sapply(uct, HasSchool)
                         ,census.tract = uct
                         )
    elapsed.cpu <- clock$Cpu()
    Printf('census tract features elapsed cpu %f seconds\n', elapsed.cpu)
    print(summary(result))
    result
}
FeaturesZip5 <- function(control, parcels.coded) {
    cat('start FeaturesZip5\n')
    #browser()
    clock <- Clock()
    HasIndustry <- function(zip5) {
        is.selected <- parcels.coded$zip5 == zip5
        result <- any(parcels.coded$is.industry[is.selected])
        result
    }
    HasPark <- function(zip5) {
        is.selected <- parcels.coded$zip5 == zip5
        result <- any(parcels.coded$is.park[is.selected])
        result
    }
    HasRetail <- function(zip5) {
        is.selected <- parcels.coded$zip5 == zip5
        result <- any(parcels.coded$is.retail[is.selected])
        result
    }
    HasSchool <- function(zip5) {
        is.selected <- parcels.coded$zip5 == zip5
        result <- any(parcels.coded$is.school[is.selected])
        result
    }
    uz <- unique(parcels.coded$zip5)
    result <- data.frame( stringsAsFactors = false
                         ,has.industry = sapply(uz, HasIndustry)
                         ,has.park = sapply(uz, HasPark)
                         ,has.retail = sapply(uz, HasRetail)
                         ,has.school = sapply(uz, HasSchool)
                         ,zip5 = uz
                         )
    elapsed.cpu <- clock$Cpu()
    Printf('zip5 features elapsed cpu %f seconds\n', elapsed.cpu)
    print(summary(result))
    result
}
FeaturesZip9 <- function(control, parcels.coded) {
    cat('start FeaturesZip9\n');
    clock <- Clock()
    HasIndustry <- function(zip9) {
        is.selected <- parcels.coded$zip9 == zip9
        result <- any(parcels.coded$is.industry[is.selected])
        result
    }
    HasPark <- function(zip9) {
        is.selected <- parcels.coded$zip9 == zip9
        result <- any(parcels.coded$is.park[is.selected])
        result
    }
    HasRetail <- function(zip9) {
        is.selected <- parcels.coded$zip9 == zip9
        result <- any(parcels.coded$is.retail[is.selected])
        result
    }
    HasSchool <- function(zip9) {
        is.selected <- parcels.coded$zip9 == zip9
        result <- any(parcels.coded$is.school[is.selected])
        result
    }
    uz <- unique(parcels.coded$zip9)
    result <- data.frame( stringsAsFactors = false
                         ,has.industry = sapply(uz, HasIndustry)
                         ,has.park = sapply(uz, HasPark)
                         ,has.retail = sapply(uz, HasRetail)
                         ,has.school = sapply(uz, HasSchool)
                         ,zip9 = uz
                         )
    elapsed.cpu <- clock$Cpu()
    Printf('zip9 features elapsed cpu %f seconds\n', elapsed.cpu)
    print(summary(result))
    result
}
AnalyzeParcelCoded <- function(control, parcels.coded) {
    # print report to stdout showing frequncy of possible indicators
    #cat('starting AnalyzeParcelIndicators\n'); str(parcels.coded); browser()
    Printf('There are %d parcels, of which:\n', nrow(parcels.coded))
    lapply( c('is.industry', 'is.park', 'is.retail', 'is.school')
           ,function(is) Printf(' %7d %s\n', sum(parcels.coded[[is]]), is)
    )
    lapply( c('census.tract', 'zip5', 'zip9')
           ,function(location) Printf(' %7d have a %s\n', sum(!is.na(parcels.coded[[location]])), location)
    )
    
    NumPer <- function(field.name) {
        num.unique <- length(unique(parcels.coded[[field.name]]))
        avg.per.unique <- nrow(parcels.coded) / num.unique
        Printf('\n')
        Printf('There are %d unique %s values\n', num.unique, field.name)
        Printf('There are %f parcels on average per unique %s\n', avg.per.unique, field.name)
    }
    
    lapply(c('census.tract', 'zip5', 'zip9'), NumPer)
    
    NULL
}
ParcelsCodedSubset <- function (parcels.coded) {
    #cat('start ParcelsCodedSubset\n'); browser()
    
    # ZIPCODE is a character field that is sometimes missing (NA) and sometimes not a number
    zips <- ZipN(parcels.coded$ZIPCODE)
    parcels.coded.zip5 <- zips$zip5
    parcels.coded.zip9 <- zips$zip9
    has.zipcode <- ifelse( is.na(parcels.coded.zip9) | is.na(parcels.coded.zip5)
                          ,FALSE
                          ,parcels.coded.zip5 >= 90000 & parcels.coded.zip5 <= 99999
                          )
    has.census.tract <- !is.na(parcels.coded$CENSUS.TRACT)
    

    
    # construct subset where we know zip5, zip9, and census.tract
    parcels.coded.subset <- data.frame( stringsAsFactors = FALSE
                                        ,is.industry = parcels.coded$is.industry
                                        ,is.park = parcels.coded$is.park
                                        ,is.retail = parcels.coded$is.retail
                                        ,is.school = parcels.coded$is.school
                                        ,census.tract = parcels.coded$CENSUS.TRACT
                                        ,zip5 = parcels.coded.zip5
                                        ,zip9 = parcels.coded.zip9
    )[has.zipcode & has.census.tract,]
    
    str(parcels.coded.subset)
    print(summary(parcels.coded.subset))
    
    parcels.coded.subset
}
Main <- function(control, parcels.coded.subset) {
    #cat('starting Main\n'); browser()

    # write control variables
    InitializeR(duplex.output.to = control$path.out.log)
    str(control)

    if (control$do.analysis) {
        AnalyzeParcelCoded(control, parcels.coded.subset)
    }

    # NOTE: if(TRUE/FALSE) used while debugging, as FeatureX is slow
    # calculating the zip 9 features would take many days
    features.census.tract <- if (TRUE) FeaturesCensusTract(control, parcels.coded.subset)
    features.zip5 <- if (TRUE) FeaturesZip5(control, parcels.coded.subset)
    features.zip9 <- if (FALSE) FeaturesZip9(control, parcels.coded.subset) else NULL
    
    print(summary(features.zip5))
    print(summary(features.census.tract))

    save(control, features.census.tract, features.zip5, features.zip9, file = control$path.out.features)
    
    str(control)
    if (control$testing) cat('TESTING: DISCARD OUTPUT\n')
}

# process the data
control <- Control()
parcels.coded <- 
    if (exists('parcels.coded')) parcels.coded else ReadParcelsCoded(control$path.in.parcels.coded)
parcels.coded.subset <-
    if (exists('parcels.coded.subset')) parcels.coded.subset else ParcelsCodedSubset(parcels.coded)
Main(control, parcels.coded.subset)
cat('done\n')

