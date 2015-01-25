Features <- function(feature.set.name) {
    # return data frame with columns feature.name, transformation
    # feature.set.names in 
    #   { act, actLog, ct, ctLog, t, tLog
    #    ,bestNN, pcaNN
    #    ,id, prices
    #    ,best15[census|city|zip]
    #    }
    # where a --> assesment
    #       c --> census
    #       t --> taxroll
    # where suffix Log --> transform size predictors into log domain
    

    Predictor <- function(name, from, kind) {
        result <- list(name = name, from = from, kind = kind)
        class(result) <- 'Predictor'
        result
    }

    Name.Predictor <- function(x) x$name
    From.Predictor <- function(x) x$from
    Kind.Predictor <- function(x) x$kind

    #                             Name                          From          Kind
    predictors <- list( Predictor('improvement.value',          'assessment', 'log')
                       ,Predictor('land.value',                 'assessment', 'log')
                       ,Predictor('effective.year.built',       'taxroll',    'none')
                       ,Predictor('factor.has.pool',            'taxrool',    'none')
                       ,Predictor('factor.is.new.construction', 'taxroll',    'none')
                       ,Predictor('year.built',                 'taxroll',    'none')
                       ,Predictor('zip5.has.industry',          'taxroll',    'none')
                       ,Predictor('zip5.has.park',              'taxroll',    'none')
                       ,Predictor('zip5.has.retail',            'taxroll',    'none')
                       ,Predictor('zip5.has.school',            'taxroll',    'none')
                       ,Predictor('avg.commute.time',           'census',     'none')
                       ,Predictor('census.tract.has.industry',  'census',     'none')
                       ,Predictor('census.tract.has.park',      'census',     'none')
                       ,Predictor('census.tract.has.retail',    'census',     'none')
                       ,Predictor('census.tract.has.school',    'census',     'none')
                       ,Predictor('fraction.owner.occupied',    'census',     'none')
                       ,Predictor('median.household.income',    'census',     'none')
                       ,Predictor('fraction.improvement.value', 'assessment', 'none')
                       ,Predictor('land.square.footage',        'taxroll',    'log')
                       ,Predictor('living.area',                'taxroll',    'log')
                       ,Predictor('land.value',                 'taxroll',    'log')
                       ,Predictor('basement.square.feet',       'taxroll',    'log1p')
                       ,Predictor('bathrooms',                  'taxroll',    'log1p')
                       ,Predictor('bedrooms',                   'taxroll',    'log1p')
                       ,Predictor('fireplace.number',           'taxroll',    'log1p')
                       ,Predictor('parking.spaces',             'taxroll',    'log1p')
                       ,Predictor('stories.number',             'taxroll',    'log1p')
                       ,Predictor('total.rooms',                'taxroll',    'log1p')
                       )

    pca.features <- c( 'median.household.income'  # the order matters
                      ,'land.square.footage'
                      ,'living.area'
                      ,'basement.square.feet'
                      )
    PCA <- function(n) pca.features[1:n]

    Predictors <- function(which, transform) {
        # return data frame
        browser()
        selected.predictors <-
            switch( which
                   ,act =
                   ,actLog = Filter( function(x) 
                                        From.Predictor(x) == 'assessment' ||
                                        From.Predictor(x) == 'census' ||
                                        From.Predictor(x) == 'taxroll'
                                    ,predictors
                                    )
                   ,ct =
                   ,ctLog =  Filter( function(x) 
                                        From.Predictor(x) == 'census' ||
                                        From.Predictor(x) == 'taxroll'
                                    ,predictors
                                    )
                   ,t =
                   ,tLog =   Filter( function(x) 
                                        From.Predictor(x) == 'taxroll'
                                    ,predictors
                                    )

                   ,stop(paste('bad which', which))
                   )
        feature.name <- Map(Name.Predictor, selected.predictors)
        transformation <-
            switch( transform
                   ,plain = Map(function(x) 'none', selected.predictors)
                   ,log  = Map(Kind.Predictor, selected.predictors)
                   ,stop(paste('bad transform', transform))
                   )
        result <- data.frame( stringsAsFactors = FALSE
                             ,'feature.name' = feature.name
                             ,'transformation' = transformation
                             )
        result
    }

    # BODY STARTS HERE
    switch( feature.set.name
           ,id = c( 'recordingDate'
                   ,'saleDate'
                   ,'apn'
                   ,'census.tract'
                   ,'zip5'
                   ,'property.city'
                   )

           ,price = c( 'price'
                      ,'price.log'
                      )

           ,act = Predictors('act', 'plain')   # formerly always
           ,actLog = Predictors('act', 'log')
           ,ct = Predictors('ct', 'plain')     # formerly alwaysNoAssessment
           ,ctLog = Predictors('ct', 'log')
           ,t = Predictors('t', 'plain')       # formerly alwaysNoCensus
           ,tLog = Predictors('t', 'log') 

           ,best01 = Best(01)
           ,best02 = Best(02)
           ,best03 = Best(03)
           ,best04 = Best(04)
           ,best05 = Best(05)
           ,best06 = Best(06)
           ,best07 = Best(07)
           ,best08 = Best(08)
           ,best09 = Best(09)
           ,best10 = Best(10)
           ,best11 = Best(11)
           ,best12 = Best(12)
           ,best13 = Best(13)
           ,best14 = Best(14)
           ,best15 = Best(15)
           ,best16 = Best(16)
           ,best17 = Best(17)
           ,best18 = Best(18)
           ,best19 = Best(19)
           ,best20 = Best(20)

           ,pca01 = PCA(01)
           ,pca02 = PCA(02)
           ,pca03 = PCA(03)
           ,pca04 = PCA(04)

           ,best15census = best15census
           ,best15city = best15city
           ,best15zip = best15zip
           ,stop(paste('bad feature.set.name', feature.set.name))
           )
}
UnitTest <- function() {
    Test <- function(feature.set.name) {
        if (TRUE) {
            df <- Features(feature.set.name)
            print(feature.set.name)
            print(df)
        }
    }
    browser()
    Test('id')
    Test('price')
    Test('act')
    Test('actLog')
    Test('ct')
    Test('ctLog')
    Test('t')
    Test('tLog')
    Test('best04')
    Test('PCA03')
    Test('best15census')
    Test('best15city')
    Test('best15zip')
}

UnitTest()

if (FALSE)  # OLD CODE
{
Predictors2 <- function(predictors.name, predictors.form) {
    # return character vector of predictor names
    # predictor.name : chr, of one always | alwaysNoassessment
    # predictor.form : chr, one of level | log
    #                  where log means use the log of size variables

    # taxonomy of features
    # house
    # - from the assessment
    # - not from the assessment
    # location
    # - derived from a decenial census
    # - derived from a zip code

    # features are further divided into
    # - size (log(price) ~ log(size) could reasonably be linear)
    #   . always positive (so that log(x) is always defined
    #   . always non-negative (so that log1p(x) is always defined)
    # - not size

    # always := the feature is present in every transaction in subset1

    path.best.features <- paste0(Directory('working'), 'e-features-lcv2.txt')
    best.features <- readLines(con = path.best.features)
    pca.features <- c( 'median.household.income'  # the order matters
                      ,'land.square.footage'
                      ,'living.area'
                      ,'basement.square.feet'
                      )

    always.house.assessment.size.positive <- 
        c( 'improvement.value'
          ,'land.value'
          )
    always.house.assessment.size.non.negative <- 
        c(
          )
    always.house.assessment.not.size <- 
        c( 'fraction.improvement.value')
    always.house.not.assessment.size.positive <- 
        c( 'land.square.footage'
          ,'living.area')
    always.house.not.assessment.size.non.negative <- 
        c( 'basement.square.feet'
          ,'bathrooms'
          ,'bedrooms'
          ,'fireplace.number'
          ,'parking.spaces'
          ,'stories.number'
          ,'total.rooms'
          )
    always.house.not.assessment.not.size <- 
        c( 'effective.year.built'
          ,'factor.has.pool'
          ,'factor.is.new.construction'
          ,'year.built'
          )
    always.location.census <- 
        c( 'avg.commute.time'
          ,'census.tract.has.industry'
          ,'census.tract.has.park'
          ,'census.tract.has.retail'
          ,'census.tract.has.school'
          ,'fraction.owner.occupied'
          ,'median.household.income'
          )
    always.location.zip <- 
        c( 'zip5.has.industry'
          ,'zip5.has.park'
          ,'zip5.has.retail'
          ,'zip5.has.school'
          )
    identification <- 
        c( 'recordingDate'
          ,'saleDate'
          ,'apn'
          ,'census.tract'
          ,'zip5'
          ,'property.city'
          )
    price <- 
        c( 'price'
          ,'price.log'
          )



    Log <- function(v) {
        # list of names in log space
        sapply(v, function(name) sprintf('%s.log', name))
    }
    Log1p <- function(v) {
        # list of names in log1p space
        sapply(v, function(name) sprintf('%s.log1p', name))
    }
    IsPositiveInteger <- function(s) {
        maybe.number <- as.integer(s)
        result <- (!is.na(maybe.number)) && maybe.number > 0
        result
    }

    #cat('Predictors2 args:', predictors.name, predictors.form, '\n'); browser()
    result.named <-
        if (predictors.name == 'price') {
            price
        } else if (predictors.name == 'identification') {
            identification
        } else if (predictors.name == 'always' && predictors.form == 'level') {
            c( always.house.not.assessment.size.positive
              ,always.house.not.assessment.size.non.negative
              ,always.house.not.assessment.not.size
              ,always.house.assessment.size.positive
              ,always.house.assessment.size.non.negative
              ,always.house.assessment.not.size
              ,always.location.census
              ,always.location.zip
              )
        } else if (predictors.name == 'always' && predictors.form == 'log') {
            c( Log(always.house.not.assessment.size.positive)
              ,Log1p(always.house.not.assessment.size.non.negative)
              ,always.house.not.assessment.not.size
              ,Log(always.house.assessment.size.positive)
              ,Log1p(always.house.assessment.size.non.negative)
              ,always.house.assessment.not.size
              ,always.location.census
              ,always.location.zip
              )
        } else if (predictors.name == 'alwaysNoAssessment' && predictors.form == 'level') {
            c( always.house.not.assessment.size.positive
              ,always.house.not.assessment.size.non.negative
              ,always.house.not.assessment.not.size
              ,always.location.census
              ,always.location.zip
              )
        } else if (predictors.name == 'alwaysNoAssessment' && predictors.form == 'log') {
            c( Log(always.house.not.assessment.size.positive)
              ,Log1p(always.house.not.assessment.size.non.negative)
              ,always.house.not.assessment.not.size
              ,always.location.census
              ,always.location.zip
              )
        } else if (predictors.name == 'alwaysNoCensus' && predictors.form == 'level') {
            c( always.house.not.assessment.size.positive
              ,always.house.not.assessment.size.non.negative
              ,always.house.not.assessment.not.size
              ,always.location.zip
              )
        } else if (predictors.name == 'alwaysNoCensus' && predictors.form == 'log') {
            c( Log(always.house.not.assessment.size.positive)
              ,Log1p(always.house.not.assessment.size.non.negative)
              ,always.house.not.assessment.not.size
              ,always.location.zip
              )
        } else if (substr(predictors.name, 1, 4) == 'best' && 
                   IsPositiveInteger(substr(predictors.name, 5, 6)) &&
                   substr(predictors.name, 7, 9) == 'census' &&
                   predictors.form == 'level') {
            # bestNNcensus, level
            n <- as.integer(substr(predictors.name, 5, 6))
            stopifnot(n >= 1)
            stopifnot(n <= 24)
            result <- c( best.features[1:n]
                        ,'census.tract'
                        )
            result
        } else if (substr(predictors.name, 1, 4) == 'best' && 
                   IsPositiveInteger(substr(predictors.name, 5, 6)) &&
                   substr(predictors.name, 7, 9) == 'city' &&
                   predictors.form == 'level') {
            # bestNNcity, level
            n <- as.integer(substr(predictors.name, 5, 6))
            result <- c( best.features[1:n]
                        ,'property.city'
                        )
            result
        } else if (substr(predictors.name, 1, 4) == 'best' && 
                   IsPositiveInteger(substr(predictors.name, 5, 6)) &&
                   substr(predictors.name, 7, 9) == 'zip' &&
                   predictors.form == 'level') {
            # bestNNzip, level
            n <- as.integer(substr(predictors.name, 5, 6))
            result <- c( best.features[1:n]
                        ,'zip5'
                        )
            result
        } else if (substr(predictors.name, 1, 4) == 'best' && 
                   IsPositiveInteger(substr(predictors.name, 5, 6)) &&
                   predictors.form == 'level') {
            # bestNN, level
            n <- as.integer(substr(predictors.name, 5, 6))
            result <- best.features[1:n]
            result
        } else if (substr(predictors.name, 1, 3) == 'pca' && 
                   IsPositiveInteger(substr(predictors.name, 4, 5)) &&
                   predictors.form == 'level') {
            n <- as.integer(substr(predictors.name, 4, 6))
            result <- pca.features[1:n]
            result
        } else {
            print(predictors.name)
            print(predictors.form)
            stop('bad arguments')
        }
    result.list <- unname(result.named)
    result.vector <- sapply(result.list, function(x) x)
} 

Predictors2Test <- function() {
    # unit test
    # for now, simply test that everything runs to completion
    verbose <- TRUE
    Test <- function(predictors.name, predictors.form = NULL) {
        browser()
        value <- Predictors2(predictors.name, predictors.form)
        if (verbose) {
            cat(predictors.name, predictors.form, '\n')
            cat(sprintf( 'predictors.name %s predictors.form %s\n'
                        ,predictors.name
                        ,as.character(predictors.form)
                        )
            )
            print(value)
            cat('number of features:', length(value), '\n')
            cat('\n')
            browser()
        }
    }
    Test('best01zip', 'level')
    Test('best15zip', 'level')
    Test('price')
    Test('identification')
    Test('always', 'level')
    Test('always', 'log')
    Test('alwaysNoAssessment', 'level')
    Test('alwaysNoAssessment', 'log')
    Test('alwaysNoCensus', 'level')
    Test('alwaysNoCensus', 'log')
}
#Predictors2Test()
}
