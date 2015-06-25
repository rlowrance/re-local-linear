# rescale certain features in the input file
# invocations: rescale.py --in  PATHIN --out PATHOUT
# where
#   both paths are to pickled files that contain a pandas DataFrame

import datetime
#import pandas as pd
import cPickle as pickle
import pdb
import sys
import warnings

from Bunch import Bunch
from directory import directory
from Logger import Logger
import parse_command_line


def make_control(argv):
    script_name = argv[0]

    b = Bunch(debugging=False,
              testing=False,
              now=datetime.datetime.now(),
              base_name=script_name.split('.')[0],
              me=script_name,
              arg_in=parse_command_line.get_arg(argv, '--in'),
              arg_out=parse_command_line.get_arg(argv, '--out'))
    return b


def rescale(df):
    # return pd.DataFrame with some features rescaled to [0,1]
    # the rescaled value of x is
    #  (x - min(x)) / (max(x) - min(x))
    # approach: mutate certain columns of df
    not_rescaled = ('SALE.DATE',
                    'zip5',
                    'apn.recoded',
                    'CENSUS.TRACT',
                    'RECORDING.DATE',
                    'YEAR.BUILT',
                    'EFFECTIVE.YEAR.BUILT',
                    'sale.date',
                    'zip5.has.industry',
                    'zip5.has.park',
                    'zip5.has.retail',
                    'zip5.has.school',
                    'census.tract.has.industry',
                    'census.tract.has.park',
                    'census.tract.has.retail',
                    'census.tract.has.school',
                    'has.pool',
                    'is.new.construction',
                    'SALE.AMOUNT',
                    'sale.month',
                    'sale.year',
                    'PROPERTY.CITY',
                    'sale.python_date')
    for column_name in df.columns:
        if column_name in not_rescaled:
            print 'not rescaling ', column_name
            continue
        print 'rescaling ', column_name
        column_values = df[column_name]
        min_value = column_values.min()
        max_value = column_values.max()
        print ' min: ', min_value, ' max: ', max_value
        rescaled_values = (column_values - min_value) / (max_value - min_value)
        df[column_name] = rescaled_values
        print ' rescaled min: ', \
            rescaled_values.min(), ' max: ', rescaled_values.max()

    return df


def main():
    warnings.filterwarnings('error')  # turn warnings into errors
    control = make_control(sys.argv)
    path = \
        directory('log') + \
        control.base_name + '.' + control.now.isoformat('T') + '.log'
    sys.stdout = Logger(logfile_path=path)  # print x now logs and prints x
    print control

    # read input
    f = open(control.arg_in, 'rb')
    df = pickle.load(f)
    f.close()

    rescaled_df = rescale(df)

    # write output
    f = open(control.arg_out, 'wb')
    pickle.dump(rescaled_df, f)
    f.close()

    print control
    print 'done'

if __name__ == '__main__':
    if False:
        pdb.set_trace()  # avoid warning from pyflakes
    main()
