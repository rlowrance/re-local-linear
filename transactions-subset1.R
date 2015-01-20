# transactions-subset1.R
# Create output file WORKING/transactions-subset1.RData


source('DirectoryLog.R')
source('DirectoryWorking.R')
source('Libraries.R')

source('ReadTransactions.R')

source('DEEDC.R')
source('SCODE.R')
source('TRNTP.R')

require(methods)  # avoid error 'could not find function "hasArg" when running via Rscript

Control <- function() {
    # set control variables
    me <- 'transactions-subset1'

    log <- DirectoryLog()
    working <- DirectoryWorking()

    control <- list( path.out.log = paste0(log, me, '.log')
                    ,path.out.rdata = paste0(working, 'transactions-subset1.RData')
                    ,path.in.transactions = paste0(working, 'transactions.RData')
                    ,testing = FALSE
                    # used in selection of OK transactions
                    ,max.sale.amount = 85e6  # source: Wall Street Journal
                    ,max.percentile = 99  
                    ,min.total.rooms = 1
                    ,required.num.buildings = 1
                    ,required.num.units = 1
                    )
    control
}

TransactionDate <- function(df) {
    # create transaction.date field (possibly NA)
    # ARGS:
    # df : data.frame with features SALE.DATE and RECORDING.DATE
    # RESULT: vector of transaction dates, some possible NA

    # If SALE.DATE is present and valid, use it for the transaction date.
    # Otherwise use the RECORDING.DATE less then average difference between
    # sale dates and recording dates
    #browser()

    ToDate <- function(v) {
        # Value: vector of adjusted sales dates
        c <- as.character(v)
        c2 <- gsub("00$", "15", c) # move dates like 1991-01-00 to 1991-01-15
        dates <- as.Date(c2, "%Y%m%d")
        dates
    }
    
    # convert dates encoded as character strings to date class objects
    adjusted.sale.date <- ToDate(df$SALE.DATE)
    adjusted.recording.date <- ToDate(df$recording.date.recoded)
    
    # mean days between known sales dates and their corresponding recording dates
    # NOTE: the recording date is always present
    adjusted.sale.date.present <- !is.na(adjusted.sale.date)
    diff <- adjusted.recording.date[adjusted.sale.date.present] - adjusted.sale.date[adjusted.sale.date.present] 
    mean.diff <- mean(diff)

    transaction.date <- as.Date(ifelse(is.na(adjusted.sale.date),
                                         adjusted.recording.date - mean.diff,
                                         adjusted.sale.date),
                                  origin="1970-01-01")
    result <- list( transaction.date = transaction.date
                   ,mean.diff = mean.diff
                   )
    result
}

# SUPPORT FUNCTIONS FOR SELECTOR FUNCTIONS

AccumulateCounts <- function(v, Classifier, field.name) {
    counts <- list()
    lapply( unique(v)
           ,function(one.v) {
               code.name <- Classifier(code = one.v)
               count <- sum(v[!is.na(v)] == one.v)  # some elements of v may be missing
               Printf('%s code %s (%s) occurs %d times\n'
                      ,field.name
                      ,one.v
                      ,code.name
                      ,count)
               counts[[code.name]] <<- count
           }
           )
    counts
}
PositiveNotHuge <- function(v, max.percentile) {
    # return selector vector for entries in v > 0 and <= 99th percentile of values in v
    stopifnot(max.percentile == 99)  # this value is hard-coded into the program
    q <- quantile(v, probs=seq(.95, 1, .01))
    quintile.99 <- q[5]
    selector <- (v > 0) & (v <= quintile.99)
    result <- list( selector = selector
                   ,max.before.selection = max(v)
                   ,max.after.selection = max(v[selector])
                   ,quintile.99 = quintile.99
                   )
}
SelectNotHuge <- function(column.name, control, df) {
    pnh <- PositiveNotHuge(df[[column.name]], control$max.percentile)
    selector <- pnh$selector
    result <- list( selector = selector
                   ,info = list( num.selected = sum(selector)
                                ,quintile.99 = pnh$quintile.99
                                ,max.before.selection = pnh$max.before.selection
                                ,max.after.selection = pnh$max.after.selection
                                )
                   )
}
SelectAll<- function(column.name, control, df) {
    selector <- rep(TRUE, nrow(df))
    result <- list( selector = selector
                   ,info = list( num.selected = sum(selector)
                                ,max.value = max(df[[column.name]])
                                )
                   )
}

# SELECTOR FUNCTIONS
# Each returns a list
# $selector: logical vector, TRUE iff the corresponding observations should be retained
# $info    : object, saved for reporting purposes

IsOneBuilding <- function(control, df) {
    is.selected <- df$NUMBER.OF.BUILDINGS == control$required.num.buildings
    result <- list( selector = is.selected
                   ,info = sum(is.selected)
                   )
}
IsOneParcel <- function(control, df) {
    # determine whether sale is for all of one parcel
    # return list $selector $info.name $info.value
    # ARGS
    # df : data.frame with features MULTI.APN.FLAG.CODE and MULTI.APN.COUNT
    # RETURNS logical vector TRUE when observation is OK

    # NA means not coded as having multiple APNs
    is.not.multiple.apn <- is.na(df$MULTI.APN.FLAG.CODE)

    has.one.apn <- df$MULTI.APN.COUNT <= 1  # most values are zero

    is.selected <- is.not.multiple.apn & has.one.apn
    result <- list( selector = is.selected
                   ,info = list( num.multi.apn.flag.code = sum(is.not.multiple.apn)
                                ,num.one.apn = sum(has.one.apn)
                                )
                   )
}
OkAssessedValue <- function(control, df) {
    #The value of the property is estimated by the tax assessor. It's broken down into the land 
    #value and the improvement value. 

    # accept zero values and values not exceeding the $85 million max sales price
    not.zero <- 
        (df$TOTAL.VALUE.CALCULATED > 0) & 
        (df$LAND.VALUE.CALCULATED > 0) & 
        (df$IMPROVEMENT.VALUE.CALCULATED > 0)

    max.value <- control$max.sale.amount
    not.too.large <- 
        df$TOTAL.VALUE.CALCULATED < max.value & 
        df$LAND.VALUE.CALCULATED < max.value & 
        df$IMPROVEMENT.VALUE.CALCULATED < max.value
            
    is.selected <- not.zero & not.too.large
    result <- list( selector = is.selected
                   ,info = list( num.not.zero = sum(not.zero)
                                ,num.not.too.large = sum(not.too.large)
                                )
                   )
}
OkDocumentTypeCode <- function(control, df) {
    # determine which deed types are valid
    # ARGS:
    # df : data.frame with DOCUMENT.TYPE.CODE field
    # RETURNS: logical vector of observations considered valid

    # Valid codes are for grant deed and trust deeds

    dtc <- df$DOCUMENT.TYPE.CODE

    info <- AccumulateCounts( v = dtc
                             ,Classifier = DEEDC
                             ,field.name = 'document.type.code'
                             )

    is.grant.deed <- DEEDC(dtc, 'grant.deed')        # sale or transfer

    # earlier a deed of trust was considered to be a sale
    #is.deed.of.trust <- DEEDC(dtc, 'deed.of.trust')  # gives mortgage lender a lien on the property
    #is.sale <- is.grant.deed | is.deed.of.trust

    is.sale <- is.grant.deed 
    result <- list( selector = is.sale
                   ,info = info
                   )
}
OkEffectiveYearBuilt <- function(control, df) {
    # accept any positive value
    # NOTE; could make sure effective year built is not before year built
    selector <- df$EFFECTIVE.YEAR.BUILT > 0  # is the year known?
    result <- list( selector = selector
                   ,info = sum(selector)
                   )
}
OkGeocoding <- function(control, df) {
    # require both latitude and longitude
    # missing values have a zero
    selector <- (df$G.LATITUDE != 0) & (df$G.LONGITUDE != 0)  # 0 ==> not available
    result <- list( selector = selector
                   ,info = sum(selector)
                   )
}
OkLandSquareFootage <- function(control, df) {
    # accept land size with up to 99th percentile
    # some land sizes are huge and some are zero
    SelectNotHuge( column.name = 'LAND.SQUARE.FOOTAGE'
                  ,control = control
                  ,df = df
                  )
}
OkLivingSquareFeet <- function(control, df) {
    # accept positive and up to 99th percentile
    SelectNotHuge( column.name = 'LIVING.SQUARE.FEET'
                  ,control = control
                  ,df = df
                  )
}
OkRecordingDate <- function(control, df) {
    # there is a recorded date
    is.missing.recording.date <- is.na(df$recording.date.recoded)
    result <- list( selector = !is.missing.recording.date
                   ,info = sum(is.missing.recording.date)
                   )
}
OkSaleAmount <- function(control, df) {
    # determine which sale amounts are valid
    # ARGS
    # df : data.fram with feature SALE.AMOUNT
    # RETURNS logical vector, TRUE, for observations with valid sale amounts
    
    # A valid sale amount is positive and less than $85 million
    # $85 million is believed to be the highest price every recorded in Los Angeles
    # for a residential real estate transaction
    is.valid.sale.amount <- (df$SALE.AMOUNT > 0) & (df$SALE.AMOUNT <= control$max.sale.amount)
    result <- list( selector = is.valid.sale.amount
                   ,info = sum(is.valid.sale.amount)
                   )
}
OkSaleCode <- function(control, df) {
    # determine valid sales code (financial consideration)
    # ARG
    # df : data.frame with feature SALE.CODE
    # RETURNS logical vector, TRUE, if sales code is valid

    sc <- df$SALE.CODE
    info<- AccumulateCounts( v = sc
                            ,Classifier = SCODE
                            ,field.name = 'SALE.CODE'
                            )
    is.sale.price.full <- SCODE(sc, 'sale.price.full')
    result <- list( selector = is.sale.price.full
                   ,info = info
                   )
}
OkTotalRooms <- function(control, df) {
    # allow 0 bedrooms, 0 bathrooms (could be an outhouse), but require at least one room
    selector <- df$TOTAL.ROOMS > control$min.total.rooms
    result <- list( selector = selector
                   ,info = sum(selector)
                   )
}
OkTransactionTypeCode <- function(control, df) {
    # determine valid transaction types
    # ARG:
    # df: data.frame containing TRANSACTION.TYPE.CODE field
    # RETURNS logical vector, TRUE, when observation considered valid

    # A valid transaction type is a resale or new construction

    ttc <- as.numeric(df$TRANSACTION.TYPE.CODE)

    info<- AccumulateCounts( v = ttc
                            ,Classifier = TRNTP
                            ,field.name = 'transaction.type.code'
                            )

    is.resale <- TRNTP(ttc, 'resale') 
    is.new.construction <- TRNTP(ttc, 'new.construction')
    is.resale.or.new.construction <- is.resale | is.new.construction
    result <- list( selector = is.resale.or.new.construction
                   ,info.name = 'transaction.type.code'
                   ,info= info
                   )
}
OkUnitsNumber <- function(control, df) {
    # require exactly one unit (otherwise, don't know what the features are for)
    selector <- df$UNITS.NUMBER == control$required.num.units
    result <- list( selector = selector
                   ,info = sum(selector)
                   )
}
OkUniversalBuildingSquareFeet <- function(control, df) {
    # REVISED: accept all
    # Some buildings are huge
    SelectNotHuge( column.name = 'UNIVERSAL.BUILDING.SQUARE.FEET'
                  ,control = control
                  ,df = df
                  )
}
OkYearBuilt <- function(control, df) {
    # accept any positive value
    selector <- df$YEAR.BUILT > 0  # is the year known?
    result <- list( selector = selector
                   ,info = sum(selector)
                   )
}

FormSubset <- function(control, df) {
    # form the subset we are interested in
    # Return list $subset1.df $info
    # ARGS
    # df : data.frame with all rows

    info <- list()  # accumulate info useful for reporting in this list

    # determine transaction date
    td <- TransactionDate(df)
    df$transaction.date <- td$transaction.date
    info$mean.days.between.sale.and.recording.dates <- td$mean.diff

    # determine observations to exclude based on values of certain features

    nrow.df <- nrow(df)
    info$num.transactions.al.sfr = nrow.df

    cat('number of observations before checking values', nrow.df, '\n')

    Do <- function(test.name, f) {
        # create a selection vector and accumulate reporting info
        # f(control, df) --> list $selector $info
        f.result <- f(control = control, df = df)

        selector.vector <- f.result$selector
        num.excluded <- nrow.df - sum(selector.vector)
        Printf(' test %20s by itself would exclude %d observations\n', test.name, num.excluded)
        print(f.result$info)

        info[[test.name]] <<- f.result$info

        selector.vector
    }
    
    is.one.building <- Do('one.building', IsOneBuilding)
    is.one.parcel <- Do('one.parcel', IsOneParcel)

    ok.assessed.value <- Do('assessed.value', OkAssessedValue)
    ok.document.type.code <- Do('DOCUMENT.TYPE.CODE', OkDocumentTypeCode)

    ok.effective.year.built <- Do('effective.year.built', OkEffectiveYearBuilt)
    ok.geocoding <- Do('geocoding', OkGeocoding)

    ok.land.square.footage <- Do('land.square.footage', OkLandSquareFootage)
    ok.living.square.feet <- Do('living.square.feet', OkLivingSquareFeet)

    ok.recording.date <- Do('recorded.date', OkRecordingDate)
    ok.sale.amount <- Do('sale.amount', OkSaleAmount)

    ok.sale.code <- Do('sale.code', OkSaleCode)
    ok.total.rooms <- Do('total.rooms', OkTotalRooms)

    ok.transaction.type.code <- Do('TRANSACTION.TYPE.CODE', OkTransactionTypeCode)
    ok.units.number <- Do('units.number', OkUnitsNumber)

    ok.universal.building.square.feet <- Do('building.square.feet', OkUniversalBuildingSquareFeet)
    ok.year.built <- Do('year.built', OkYearBuilt)



    # determine all observations excluded
    all.good <- 
        is.one.building &
        is.one.parcel &

        ok.assessed.value &
        ok.document.type.code &

        ok.effective.year.built &
        ok.geocoding &

        ok.land.square.footage &
        ok.living.square.feet &

        ok.recording.date &
        ok.sale.amount & 
        
        ok.sale.code &
        ok.total.rooms &

        ok.transaction.type.code &
        ok.units.number &

        ok.universal.building.square.feet &
        ok.year.built 

    total.num.excluded <- nrow.df - sum(all.good)
    Printf(' all fields combined excluded %d transactions\n', total.num.excluded)
    info$total.num.excluded = total.num.excluded

    result <- list( subset1 = df[all.good, ]
                   ,info = info
                   )
    result
}

RecodeRecordingDate <- function(recordingDate) {
    # replace YYYYMM00 with YYYYMM15 where recordingDate is an int
    day <- recordingDate %% 100
    result <- ifelse(day == 0, recordingDate + 15, recordingDate)
    result
}

Main <- function(control, transactions) {
    #cat('starting Main\n') ; browser()

    # recoded RECORDING.DATE
    transactions$recording.date.recoded <- RecodeRecordingDate(transactions$RECORDING.DATE)

    # form the subset we are interested in
    form.subset <- FormSubset(control, transactions)
    subset1 <- form.subset$subset1
    info <- form.subset$info
    #str(subset1)
    print(summary(subset1))

    # eliminate duplicate transactions
    num.with.dups = nrow(subset1)
    subset1.unique <- unique(subset1)
    num.dropped = num.with.dups - nrow(df)
    info$num.with.possible.duplicates <- num.with.dups
    info$num.duplicates.dropped <- num.dropped


    cat('number of duplicate observations eliminated', num.dropped, '\n')
    Printf('about to write %d subset1 transactions\n', nrow(subset1.unique))
    stopifnot(num.dropped == 0)

    str(info)

    # write output file
    transactions.subset1 = subset1.unique
    save( info
         ,transactions.subset1
         ,control
         ,file = control$path.out.rdata
         )

    str(info)

    str(control)
    if (control$testing)
        cat('DISCARD OUTPUT; TESTING\n')
}

# EXECUTION STARTS HERE
control <- Control()
InitializeR(duplex.output.to = control$path.out.log)
str(control)
transactions <-
    if (exists('transactions')) {
        transactions
    } else {
        cat('reading transactions\n')
        #debug(ReadDeedsAl)
        ReadTransactions(path = control$path.in.transactions)
    }


Main(control, transactions)
cat('done\n')
