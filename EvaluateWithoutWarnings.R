EvaluateWithoutWarnings <- function(expr) {
    # return value of expression, completely ignoring any warnings
    current.warn.option <- getOption('warn')
    options(warn = -1)  # no warning given, even at the top level
    
    result <- force(expr)
    
    options(warn = current.warn.option)
    
    result
}