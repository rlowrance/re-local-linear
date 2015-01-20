ReadCensus <- function(path) {
    #cat('start ReadCensus', path, '\n'); browser()
    census <- NULL
    loaded <- load(path)  # may load more than parcels.sfr
    stopifnot(!is.null(census))
    census
}
