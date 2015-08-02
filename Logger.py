import sys
import pdb

if False:
    pdb.set_trace()  # avoid warning message from pyflakes


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

if False:
    # usage example
    sys.stdout = Logger('path/to/log/file')
    # now print statements write on both stdout and the log file
