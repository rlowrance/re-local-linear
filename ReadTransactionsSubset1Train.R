ReadTransactionsSubset1Train <- function(path) {
    # return transactions.al.sfr.subset1 object
    train <- NULL
    loaded <- load(path)
    stopifnot(!is.null(train))
    train
}
