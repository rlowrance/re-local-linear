Directory <- function(name) {
    # return string containing path to the directory with the informal name
    working <- '../data/working/'
    switch( name
           ,cells = paste0(working, 'e-cv-cells/')
           ,log = paste0(working, 'log/')
           ,working = working
           ,stop(paste('bad name', name))
           )
}
