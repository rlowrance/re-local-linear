# deeds-al-g.R
# main program to create file OUTPUT/deeds-al-g.RData, hold many features of arms-length grant deeds.
# Record layout for the input is in 1080_Record_layout.csv
# Save just the deeds info, not the info that is also in the payroll file (except for the APN)

source('DEEDC.R')
source('Directory.R')
source('InitializeR.R')
source('PRICATCODE.R')

require(methods)  # make hasArg() function available

Control <- function() {
    # return list of values that control the script
    #cat('start Control\n'); browser()
    me <-'deeds-al-g'
    
    input <- Directory('input')
    log <- Directory('log')
    working <- Directory('working')
    
    volume1 <- 'corelogic-deeds-090402_07/'
    volume2 <- 'corelogic-deeds-090402_09/'
    
    Deeds <- function(n) {
        # return path to deeds
        paste0( input
               ,if (n <= 4) volume1 else volume2
               ,sprintf('CAC06037F%d.zip', n)
               )
    }
    
    control <- list(
         path.in.deeds = list(Deeds(1), Deeds(2), Deeds(3), Deeds(4), Deeds(5), Deeds(6), Deeds(7), Deeds(8))
        ,path.out.log = paste0(log, me, '.log')
        ,path.out.deeds = paste0(working, me, '.RData')
        ,testing = FALSE
        )
    control
}
ReadDeedsFile <- function(control, num) {
    # Return dataframe containing all deeds from input file num
    # ARGS:
    # num: number of input file; in {1, 2, ..., 8}

    # read a deeds file
    # Note: In file 5, data record 945 has an NA value for APN.FORMATTED
    #cat('start ReedDeedsFile', num, '\n'); browser()
    path <- control$path.in.deeds[[num]]
    len <- nchar(path)
    filename <- paste0(substr(path, len-13, len-4), '.txt')
    
    # NOTE: Don't convert strings to factors, because the other input
    # files may have different values than this file
    # read.table cannot read a zip file directly
    df <- read.table(
         file = unz(path, filename)
        ,header=TRUE
        ,sep="\t"
        ,quote=""
        ,comment.char=""
        ,stringsAsFactors=FALSE
        ,na.strings=""
        ,nrows=if(control$testing) 1000 else -1
        )
    
    sprintf('Read %d observations\n from file %s\n in zip %s\n', nrow(df), filename, path)
    
    # keep just APN and info unique to current sale
    # (so don't keep info that is in the taxroll file)
    keeps <- c( 'APN.FORMATTED'
               ,'APN.UNFORMATTED'
               ,'APN.SEQUENCE.NUMBER'
               ,'DOCUMENT.YEAR'
               ,'SALE.AMOUNT'
               ,'MORTGAGE.AMOUNT'
               ,'SALE.DATE'
               ,'RECORDING.DATE'
               ,'DOCUMENT.TYPE.CODE'
               ,'TRANSACTION.TYPE.CODE'
               ,'SALE.CODE'
               ,'MULTI.APN.FLAG.CODE'
               ,'MULTI.APN.COUNT'
               ,'TITLE.COMPANY.CODE'
               ,'MORTGAGE.DATE'
               ,'MORTGAGE.LOAN.TYPE.CODE'
               ,'MORTGAGE.DEED.TYPE.CODE'
               ,'MORTGAGE.TERM.CODE'
               ,'MORTGAGE.TERM'
               ,'MORTGAGE.DUE.DATE'
               ,'MORTGAGE.ASSUMPTION.AMOUNT'
               ,'X2ND.MORTGAGE.AMOUNT'
               ,'X2ND.MORTGAGE.LOAN.TYPE.CODE'
               ,'X2ND.MORTGAGE.DEED.TYPE.CODE'
               ,'PARTIAL.INTEREST.INDICATOR.FLAG'
               ,'OWNERSHIP.TRANSFER.PERCENTAGE'
               ,'PRI.CAT.CODE'
               ,'MORTGAGE.INTEREST.RATE.TYPE.CODE'
               ,'SELLER.CARRY.BACK.FLAG'
               ,'PRIVATE.PARTY.LENDER.FLAG'
               ,'CONSTRUCTION.LOAN.FLAG'
               ,'RESALE.NEW.CONSTRUCTION.CODE'
               ,'INTER.FAMILY.FLAG'
               ,'CASH.MORTGAGE.PURCHASE.CODE'
               ,'FORCLOSURE.CODE'
               ,'REFI.FLAG.CODE'
               ,'EQUITY.FLAG.CODE'
               )

    if (FALSE) {
        # test that we spelled field names correctly
        lapply(keeps, function(one.keep) {
               cat(one.keep, '\n')
               df[one.keep]  # this will fail if one keep is not a column
               }
        )
    }
    kept <- df[keeps]

    # track original source
    kept$deed.file.number=rep(num, nrow(df))
    kept$deed.record.number=1:nrow(df)
    
    kept
}
ReadAll <- function(control) {
    # Return all the arms-length deeds files into one big data.frame
    # ARGS:
    # control : list of control values
    df <- NULL
    for (file.number in 1:8) {
        deeds<- ReadDeedsFile(control, file.number)
        df <- rbind(df, deeds)
        #cat('after file.number', file.number, '\n'); browser()
    }
    df
}
Main <- function(control) {
    #cat('start Main\n'); browser()
    
    # write control variables
    InitializeR(duplex.output.to = control$path.out.log)
    str(control)
    
    # Read all the deeds
    all <- ReadAll(control)
    
    # Retain only observations coded as arms-length and as grant deeds
    is.arms.length <- PRICATCODE(all$PRI.CAT.CODE, 'arms.length.transaction')
    is.grant.deed <- DEEDC(all$DOCUMENT.TYPE.CODE, 'grant.deed')
    is.keeper <- is.arms.length & is.grant.deed
    deeds.al.grant <- all[is.keeper,]
    str(deeds.al.grant)

    # print record counts
    count <- list( num.all.deeds = nrow(all)
                  ,num.is.arms.length = sum(is.arms.length)
                  ,num.is.grant.deed  = sum(is.grant.deed)
                  ,num.is.arms.length.and.grant.deed = sum(is.keeper)
                  )
    str(count)

    # Write RData
    save(deeds.al.grant, count, control, file = control$path.out.deeds)#

    
    #write control variables
    str(control)
    if (control$testing)
        cat('DISCARD OUTPUT: TESTING\n')
}

Main(Control())
cat('done\n')
