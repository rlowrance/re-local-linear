Directory <- function(name) {
    # return string containing path to the directory with the informal name
    root <- '../'  # root of project
    switch( name

           ,cells = paste0(root, 'data/working/e-cv-cells/')
           ,input = paste0(root, 'data/input/')
           ,log = paste0(root, 'data/working/log/')
           ,working = paste0(root, 'data/working/')

           ,stop(paste('bad name', name))
           )
}
