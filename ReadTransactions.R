ReadTransactions <- function(path) {
    # return data.frame with all the transaction in the path
    transactions <- NULL
    loaded <- load(path)  # may load more than transactions
    str(loaded)
    stopifnot(!is.null(transactions))
    transactions
}
