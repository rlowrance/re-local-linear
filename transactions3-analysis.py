'''analyze columns in a transaction3 csv file

INPUT FILE: specified on command line via --in
 INPUT/transactions3-al-g-sfr.csv

OUTPUT FILE: specified on command line via --out
'''

import numpy as np
import pandas as pd
import pdb
from pprint import pprint
import sys

from Bunch import Bunch
from directory import directory
from Logger import Logger
from Report import Report
import parse_command_line


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python transactions3-analysis.py --csv RELPATH --txt RELPATH [--test]'
    print ' --csv RELPATH: path to input file, a transaction3-format csv file'
    print ' --txt RELPATH: path to output file, which will contain a text report'
    print ' --test       : run in test mode'
    print 'where'
    print ' RELPATH is a path relative to the WORKING directory'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (5, 6):
        usage('invalid number of arguments')

    pcl = parse_command_line.ParseCommandLine(argv)
    arg = Bunch(
        base_name=argv[0].split('.')[0],
        csv=pcl.get_arg('--csv'),
        test=pcl.has_arg('--test'),
        txt=pcl.get_arg('--txt'),
    )
    if arg.csv is None:
        usage('missing --csv')
    if arg.txt is None:
        usage('missing --txt')

    debug = False

    return Bunch(
        arg=arg,
        debug=debug,
        path_in=directory('working') + arg.csv,
        path_out=directory('working') + arg.txt,
        test=arg.test,
    )


def analyze(transactions, column_name, report):
    values = transactions[column_name]
    if values.dtype not in (np.dtype('int64'), np.dtype('float64'), np.dtype('object')):
        print column_name, type(values), values.dtype
        pdb.set_trace()
    report.append(' ')
    report.append(column_name)
    ndframe = values.describe()
    report.append(ndframe)


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(base_name=control.arg.base_name)
    print control

    transactions = pd.read_csv(control.path_in,
                               nrows=100 if control.arg.test else None,
                               )
    report = Report()
    report.append('Analysis of transactions3 file: ' + control.arg.csv)
    report.append('shape is ' + str(transactions.shape))
    report.append(' ')
    column_names = sorted(transactions.columns)
    for column_name in column_names:
        analyze(transactions, column_name, report)
    pdb.set_trace()
    report.write(control.path_out)

    print control
    if control.test:
        print 'DISCARD OUTPUT: test'
    print 'done'

    return


if __name__ == '__main__':
    if False:
        # avoid pyflakes warnings
        pdb.set_trace()
        pprint()
        parse_command_line()
        pd.DataFrame()
        np.array()

    main(sys.argv)
