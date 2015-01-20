ReadDeedsAlG <- function(path) {
    #cat('start ReadDeedsAlG', path, '\n'); browser()
    deeds.al.g <- NULL
    loaded <- load(path)  # may load more than parcels.sfr
    stopifnot(!is.null(deeds.al.grant))
    deeds.al.grant
}
