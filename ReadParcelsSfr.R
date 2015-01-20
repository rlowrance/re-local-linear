# ReadParcelsSfr.R
# function to read all the parcels from an RData file
ReadParcelsSfr <- function(path) {
    #cat('start ReadParcelsSfr', path, '\n'); browser()
    parcels.sfr <- NULL
    loaded <- load(path)  # may load more than parcels.sfr
    stopifnot(!is.null(parcels.sfr))
    parcels.sfr
}
