ReadParcelsCoded <- function(path) {
    # return data.frame parcels.code in RData at path
    #cat('start ReadParcelsCoded', path, '\n'); browser()
    parcels.coded <- NULL
    loaded <- load(path)
    stopifnot(!is.null(parcels.coded))
    parcels.coded
}
