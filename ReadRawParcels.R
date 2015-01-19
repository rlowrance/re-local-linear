ReadRawParcels <- function(nrows, path.to.input.directory, verbose = FALSE) {
    # return data frame containing all are first nrows of all parcels
    #cat('start ReadRawParcels\n'); browser()

    ReadParcelsFile <- function(num) {
        # return dataframe containing all features in a taxroll file
        #cat('start ReadParcelsFile', num, '\n'); browser()

        path <- paste0( path.to.input.directory
                       ,'corelogic-taxrolls-090402_05/'
                       ,sprintf('CAC06037F%d.zip', num)
                       )

        # The path is to a zip file. It has one file in it. Construct the file name.
        len <- nchar(path)
        filename <- paste0(substr(path, len - 13, len - 4), '.txt')

        df <- read.table( file = unz(path, filename)
                         ,header=TRUE
                         ,sep="\t"
                         ,quote=""
                         ,comment=""
                         ,stringsAsFactors=FALSE
                         ,na.strings=""
                         ,nrows=nrows
                         )

        if (verbose) {
            sprintf('Read %d observations\n from file %s\n in zip %s\n', nrow(df), filename, path)
        }

        # track original source
        df$parcel.file.number <- rep(num,nrow(df))
        df$parcel.record.number <- 1:nrow(df)

        df
    }

    # BODY STARTS HERE

    df <- NULL
    for (file.number in 1:8) {
        parcels <- ReadParcelsFile(file.number)
        df <- rbind(df, parcels)
    }
    df
}
