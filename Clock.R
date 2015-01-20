# Clock S3 class
# constructor: Clock(), initialize and restart
# methods
#  Reset(x)     --> reset timer
#  UserCpu(x)   --> return user cpu time in seconds from last restart
#  SystemCpu(x) --> return system cpu time in seconds from last restart
#  Cpu(x)       --> return total cpu time in seconds from last restart
#  WallCpu(x)   --> return wallclock time in seconds from last restart

Clock <- function() {
    result <- proc.time()
    class(result) <- 'Clock'
    result
}

Elapsed <- function(x) { # helper function
    proc.time() - x
}

# Methods
Reset.Clock <- function(x) {
    x <- proc.time()
}
UserCpu.Clock <- function(x) {
    e <- Elapsed(x)
    e[['user.self']] + e[['user.child']]
}
SystemCpu.Clock <- function(x) {
    e <- Elapsed(x) 
    e[['sys.self']] + e[['sys.child']]
}
Cpu.Clock <- function(x) {
    UserCpu(x) + SystemCpu(x)
}
Wallclock.Clock <- function(x) {
    e <- Elapsed(x)
    e[['elapsed']]
}

ClockTest <- function() {
    # unit test
    v <- function(name, value) {
        if (TRUE) {
            Printf('%s = %f\n', name, value)
        }
    }
    c <- Clock()
    Sys.sleep(1)  # sleep for 1 second
    v('user cpu', UserCpu(c))
    v('system cpu', SystemCpu(c))
    v('cpu', Cpu(c))
    v('wallclock', Wallclock(c))
    browser()
}

#ClockTest()

# OLD BELOW ME
if (FALSE) {
Clock <- function() {
    # return a new clock object and access functions for it:
    # clock <- Clock()
    #
    # clock$Reset()    : restart clock at zero
    # clock$Usercpu()  : return elapsed user cpu time in seconds from last reset
    # clock$Systemcpu(): return elapsed system cpu time in seconds from last reset
    # clock$Cpu()      : return elapsed total cpu time in seconds from last reset
    # clock$Wallclock(): return elapsed wall clock time in seconds from last reset
    #
    # ref: www.ats.ucla.edu/stat/r/faq/timing_code.htm
    initial.clock <- NULL

    Reset <- function() {
        initial.clock <<- proc.time()
    }

    Difference <- function() {
        proc.time() - initial.clock
    }

    Usercpu <- function() {
        e <- Difference()
        e[['user.self']] + e[['user.child']]
    }

    Systemcpu <- function() {
        e <- Difference()
        e[['sys.self']] + e[['sys.child']]
    }

    Cpu <- function() {
        Usercpu() + Systemcpu()
    }

    Wallclock <- function() {
        e <- Difference()
        e[['elapsed']]
    }

    Reset()
    
    list( Reset = Reset
         ,Usercpu = Usercpu
         ,Systemcpu = Systemcpu
         ,Cpu = Cpu
         ,Wallclock = Wallclock
         )
}

Clock.test <- function() {
    c <- Clock()
    # make sure that all the function run to completion
    stopifnot(c$Reset() >= 0)
    stopifnot(c$Usercpu() >= 0)
    stopifnot(c$Systemcpu() >= 0)
    stopifnot(c$Cpu() >= 0)
    stopifnot(c$Wallclock() >= 0)
}

Clock.test()
}
