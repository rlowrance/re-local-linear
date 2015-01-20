# Create output files WORKING/transactions-subset1-train.RData
#                     WORKING/transactions-subset1-test.RData
#
# The test data consists of about 1,000 randomly-chosen samples from each year
# The training data consists of all the other samples

source('Directory.R')
source('Libraries.R')

source('ReadTransactionsSubset1.R')

require(methods)  # avoid error 'could not find function "hasArg" when running via Rscript

Control <- function() {
    # set control variables
    me <- 'transactions-subset1-train'

    log <- Directory('log')
    working <- Directory('working')

    testing <- TRUE
    testing <- FALSE
    control <- list( path.out.log = paste0(log, me, '.log')
                    ,path.out.train.rdata = paste0(working, 'transactions-subset1-train.RData')
                    ,path.out.test.rdata = paste0(working, 'transactions-subset1-test.RData')
                    ,path.in.transactions.subset1 = paste0(working, 'transactions-subset1.RData')
                    ,fraction.samples.per.month = 0.02
                    ,random.seed = 123
                    ,testing = testing
                    )
    control
}

Main <- function(control, transactions.subset) {
    #cat('starting Main\n') ; browser()

    sale.year <- SplitDate(transactions.subset$transaction.date)$year
    sale.month <- SplitDate(transactions.subset$transaction.date)$month

    sale.year.min <- min(sale.year, na.rm = TRUE)
    sale.year.max <- max(sale.year, na.rm = TRUE)
    Printf('sale.year min %d max %d\n', sale.year.min, sale.year.max)


    # pick 2% of transactions from each month as the test set
    sample.all <- NULL
    for (year in 1984:2009) {
        for (month in 1:12) {
            in.month <- sale.year == year & sale.month == month
            selected <- which(in.month)
            sample.for.month <- sample( x = selected
                                       ,size = round(length(selected) * control$fraction.samples.per.month)
                                       ,replace = FALSE
                                       )
            Printf( 'year %d month %2d has %d observations of which %d are in test set\n'
                   ,year
                   ,month
                   ,sum(in.month)
                   ,length(sample.for.month)
                   )
            sample.all <- c(sample.all, sample.for.month)
        }
    }

    Printf('randomly selected %d transactions to be in test set\n', length(sample.all))

    test <- transactions.subset[sample.all,]
    train <- transactions.subset[-sample.all,]

    info <- list( num.test = nrow(test)
                 ,num.train = nrow(train)
                 ,sample.all = sample.all
                 )


    # write output files
    save( test
         ,control
         ,info
         ,file = control$path.out.test.rdata
         )
    save( train
         ,control
         ,info
         ,file = control$path.out.train.rdata
         )

    str(info)

    str(control)
    if (control$testing)
        cat('DISCARD OUTPUT; TESTING\n')
}

# EXECUTION STARTS HERE
control <- Control()
InitializeR( duplex.output.to = control$path.out.log
            ,random.seed = control$random.seed
            )
str(control)
transactions.subset <-
    if (exists('transactions.subset')) {
        transactions.subset
    } else {
        cat('reading transactions subset\n')
        #debug(ReadDeedsAl)
        ReadTransactionsSubset1(path = control$path.in.transactions.subset)
    }


Main(control, transactions.subset)
cat('done\n')
