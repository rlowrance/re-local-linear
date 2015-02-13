# create files
# WORKING/transactions-subset2.pickle
# WORKING/transactions-subset2-counts.csv

# import built-ins and libraries
import numpy as np
import pandas as pd
import pdb
import sys
import cPickle as pickle
import datetime

# import my stuff
from directory import directory
from features import features
from year_month_day import year_month_day
from Logger import Logger
from SCODE import SCODE
from TRNTP import TRNTP


class Control(object):
    def __init__(self):
        me = 'transactions-subset2'
        working = directory('working')
        log = directory('log')

        self.path_out_pickle = working + me + '.pickle'
        self.path_out_counts = working + me + '-counts.csv'
        self.path_out_log = log + me + '.log'
        self.path_in_data = working + 'transactions.csv'

        self.testing = False
        self.debugging = False


def feature_names(feature_name_set):
    '''Return list of feature names (dropping transformations).'''
    d = features(feature_name_set)
    result = []
    for k in d.iterkeys():
        result.append(k)
    return result


class Selector(object):
    def __init__(self, df):
        self.df = df
        self.num_observations = df.shape[0]
        self.count_dropped = {}
        self.all_selected = [True] * self.num_observations

    def select(self, description, selector_function):
        selector_vector = selector_function(self.df)
        self.count_dropped[description] = \
            self.num_observations - sum(selector_vector)
        self.all_selected &= selector_vector

    def report(self):
        for k, v in self.count_dropped.iteritems():
            print 'selector', k, 'dropped', v


def set_certain_column_types(path_in_csv):
    '''
    Return dictionary specifying some column types.

    These were found by running pd.read_csv and examining the stdout.
    '''

    # found by running pd.read_csv on all the rows and examing stdout
    indices_with_mixed_types = (26,
                                32, 38,
                                48, 49,
                                50, 51,
                                61, 65, 68,
                                71, 75, 76, 79,
                                80, 89,
                                93, 94,
                                150, 156, 157,
                                164,
                                174)

    column_names = pd.read_csv(path_in_csv,
                               nrows=1).columns

    # each of the mixed type column should be read an Python objects
    d = {}
    for column_name_index in indices_with_mixed_types:
        column_name = column_names[column_name_index]
        d[column_name] = np.object

    return d


def create_new_features(df, trntp):
    '''Mutate df to add features.'''

    def add_splits_of_sale_date():
        sale_year, sale_month, sale_day = year_month_day(df['SALE.DATE'])
        df['sale.year'] = sale_year
        df['sale.month'] = sale_month
        df['sale.day'] = sale_day

    def add_fraction_improvement_value():
        lvc = df['LAND.VALUE.CALCULATED']
        ivc = df['IMPROVEMENT.VALUE.CALCULATED']
        fraction_improvement_value = ivc / (ivc + lvc)
        df['fraction.improvement.value'] = fraction_improvement_value

    def add_has_pool():
        df['has.pool'] = df['POOL.FLAG'] == 'Y'
        # which is equivalent to this logic (ported from R subset1)
        if False:
            pf = df['POOL.FLAG']
            POOL_FLAG_isnull = pf.isnull()
            POOL_FLAG_isY = pf == 'Y'
            # R: ifelse(is.na(POOL.FLAG), FALSE, POOL.FLAG == 'Y')
            has_pool = np.where(POOL_FLAG_isnull, False, POOL_FLAG_isY)
            print has_pool

    def add_is_new_construction():
        ttc = df['TRANSACTION.TYPE.CODE']
        df['is.new.construction'] = trntp.is_new_construction(ttc)

    add_splits_of_sale_date()
    add_fraction_improvement_value()
    add_has_pool()
    add_is_new_construction()


def main():
    control = Control()
    sys.stdout = Logger(logfile_path=control.path_out_log)

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    mixed_types = set_certain_column_types(control.path_in_data)
    # NOTE: need 1000 records to read enough POOL.FLAG to force type to object
    # if this isn't done, add_has_pool fails
    df = pd.read_csv(control.path_in_data,
                     nrows=10000 if control.testing else None,
                     dtype=mixed_types)

    if True:
        for column_name in df.columns:
            print 'all column name', column_name
            print df[column_name].describe()
            print

    scode = SCODE()
    trntp = TRNTP()

    create_new_features(df, trntp)  # mutate df

    # drop transactions with certain conditions:
    # unlike subset1, do NOT screen out very large values for
    #  LAND.SQUARE.FOOTAGE
    #  LIVING.SQUARE.FEET
    #  SALE.AMOUNT
    #  TOTAL.VALUE.CALCULATED
    #  UNIVERSAL.BUILDING.SQUARE.FEET
    # unlike subset1, do NOT screen out zero values in
    #  G.LATITUDE
    #  G.LONGITUDE
    # unlike subset1, DO screen out
    #  sale_day == 0 (subset1 guessed that the day was 15)

    def resale_or_new_construction(x):
        v = x['TRANSACTION.TYPE.CODE']
        return trntp.is_resale_or_new_construction(v)

    selector = Selector(df)
    selector.select('sale date >= 2003',
                    lambda x: x['sale.year'] >= 2003)
    selector.select('sale day != 0',
                    lambda x: x['sale.day'] != 0)
    selector.select('1 building',
                    lambda x: x['NUMBER.OF.BUILDINGS'] > 0)
    selector.select('1 apn',
                    lambda x: x['MULTI.APN.COUNT'] <= 1)
    selector.select('land value > 0',
                    lambda x: x['LAND.VALUE.CALCULATED'] > 0)
    selector.select('improvement value > 0',
                    lambda x: x['IMPROVEMENT.VALUE.CALCULATED'] > 0)
    selector.select('effective year built > 0',
                    lambda x: x['EFFECTIVE.YEAR.BUILT'] > 0)
    selector.select('land square footage > 0',
                    lambda x: x['LIVING.SQUARE.FEET'] > 0)
    selector.select('sale amount > 0',
                    lambda x: x['SALE.AMOUNT'] > 0)
    selector.select('full sale price',
                    lambda x: scode.is_sale_price_full(x['SALE.CODE']))
    selector.select('total rooms > 0',
                    lambda x: x['TOTAL.ROOMS'] > 0)
    selector.select('resale or new construction only',
                    resale_or_new_construction)
    selector.select('1 unit',
                    lambda x: x['UNITS.NUMBER'] == 1)
    selector.select('universal building square feet > 0',
                    lambda x: x['UNIVERSAL.BUILDING.SQUARE.FEET'] > 0)
    selector.select('year built > 0',
                    lambda x: x['YEAR.BUILT'] > 0)
    selector.report()

    # determine feature names we actually use

    ids = set(features('id').keys())
    prices = set(features('prices').keys())
    act = set(features('act').keys())
    features_used = list(ids | prices | act)
    for used_column in features_used:
        print 'keeping column', used_column

    # keep just rows the were selected and the columns we actually use

    subset = df.ix[selector.all_selected, features_used]

    # add date feature sale.date
    years = subset['sale.year'].values
    months = subset['sale.month'].values
    days = subset['sale.day'].values

    # build list of python dates
    dates = []
    for i in xrange(len(years)):
        date = datetime.date(int(years[i]),
                             int(months[i]),
                             int(days[i]))
        dates.append(date)

    date_series = pd.Series(dates)
    if date_series.isnull().any():
        raise ValueError('null date')
    # date_series index is wrong, so fix it
    date_series.index = subset.index
    subset['sale.python_date'] = date_series  # NOTE: aligns indices

    # describe all columns in the subset
    print 'subset.shape', subset.shape
    if True:
        for column_name in subset.columns:
            print 'subset column name', column_name
            print subset[column_name].describe()
            print

    subset.to_pickle(control.path_out_pickle)

    # read it back in and test sale.python_date for nulls
    if False:
        f = open(control.path_out_pickle, 'rb')
        d = pickle.load(f)
        print (d)
        python_date = d['sale.python_date']
        if python_date.isnull().any():
            print python_date
            raise ValueError('null sale.python_date')

    # write record counts
    counts = pd.DataFrame({'file_name': ['all'],
                           'record_count': [subset.shape[0]]})
    # counts = pd.DataFrame(dd)
    counts.to_csv(control.path_out_counts)

    # read, if debugging
    if control.debugging:
        subset = None
        f = open(control.path_out_pickle, 'rb')
        subset = pickle.load(f)
        f.close()
        pdb.set_trace()  # investigate column sale.datetime
        print 'investigate sale.datetime'

    # log the control variables
    for k, v in control.__dict__.iteritems():
        print 'control', k, v

    print 'done'


if __name__ == '__main__':
    main()
