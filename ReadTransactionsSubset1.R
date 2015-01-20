ReadTransactionsSubset1 <- function(path) {
    # return transactions.al.sfr.subset1 object
    transactions.subset1 <- NULL
    loaded <- load(path)
    stopifnot(!is.null(transactions.subset1))
    transactions.subset1
}
