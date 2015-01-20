# define all S3 methods
AsCharacter <- function(x, ...) UseMethod('AsCharacter')
Command     <- function(x, ...) UseMethod('Command')
Cpu         <- function(x, ...) UseMethod('Cpu')
Path        <- function(x, ...) UseMethod('Path')
SystemCpu   <- function(x, ...) UseMethod('SystemCpu')
UserCpu     <- function(x, ...) UseMethod('UserCpu')
Wallclock   <- function(x, ...) UseMethod('Wallclock')
