source('EvaluateWithoutWarnings.R')
ZipN <- function(zipcodes.char) {
    # convert vector of 9-digit zipcodes encoded as characters into 9 and 5 digit numeric zip codes
    zip9 <- EvaluateWithoutWarnings(as.numeric(zipcodes.char))
    zip5 <- round(zip9 / 10000)
    list( zip5 = zip5
         ,zip9 = zip9
         )
}
