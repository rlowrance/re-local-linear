'''Demonstrate possible bug in iPython'''

import sys


class Logger(object):
    # from stack overflow: how do i duplicat sys stdout to a log file in python

    def __init__(self, logfile_path, logfile_mode='w'):
        self.terminal = sys.stdout
        clean_path = logfile_path.replace(':', '-')
        self.log = open(clean_path, logfile_mode)
        pass

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

log_path = '/tmp/Logger-test.txt'

sys.stdout = Logger(logfile_path=log_path)

print "abc"  # should print on std out and log file
