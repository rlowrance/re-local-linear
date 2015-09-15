'''create joined files of test and training transactions

INPUT FILES
 INPUT/corelogic-deeds-090402_07/CAC06037F1.zip ...
 INPUT/corelogic-deeds-090402_09/CAC06037F1.zip ...
 INPUT/corelogic-taxrolls-090402_05/CAC06037F1.zip ...
 INPUT/geocoding.txv
 INPUT/neighborhood-data/census.csv

OUTPUT FILES
 WORKING/transactions-subset3-all-samples.csv
 WORKING/transactions-subset3-subset-all.csv
 WORKING/transactions-subset3-subset-test.pickle  OR .csv
 WORKING/transactions-subset3-subset-train.pickle

NOTES:
    1. The deeds file has the prior sale info, which we could use
       to create more transactions. We didn't, because we only have
       census data from the year 2000, and we use census tract
       features, so its not effective to go back before sometime
       in 2002, when the 2000 census data became public.
'''

import collections
import csv
import datetime
import numpy as np
import pandas as pd
import cPickle as pickle
import pdb
from pprint import pprint
import random
from sklearn import cross_validation
import sys
import time
import zipfile


from Bunch import Bunch
import census
import deeds
from directory import directory
from Logger import Logger
import parse_command_line
import parcels


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage  : python transactions_subset3.py [--just TAG] [--test]'
    print ' TAG   : run only a portion of the analysis (used during development)'
    print ' --test: run in test mode'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) not in (1, 2, 3, 4):
        usage('invalid number of arguments')

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()

    def cache_path(name):
        return directory('working') + base_name + '-cache-' + name + '.pickle'

    test = parse_command_line.has_arg(argv, '--test')
    debug = False
    cache = True

    return Bunch(
        debug=debug,
        test=test,
        cache=cache,
        just=parse_command_line.default(argv, '--just', None),
        path_cache_base=directory('working') + base_name + '-cache-',
        path_cache_deeds_g_al=directory('working') + base_name + '-cache-deeds-g-al.csv',
        path_cache_parcels=directory('working') + base_name + '-cache-parcels.csv',
        path_in_census=directory('input') + 'neighborhood-data/census.csv',
        path_in_geocoding=directory('input') + 'geocoding.tsv',
        path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
        path_out_subset=directory('working') + base_name + '-subset-all.csv',
        path_out_test=directory('working') + base_name + '-subset-test.csv',
        path_out_train=directory('working') + base_name + '-subset-train.csv',
        path_all_samples=directory('working') + base_name + '-all-samples.csv',
        dir_deeds_a=directory('input') + 'corelogic-deeds-090402_07/',
        dir_deeds_b=directory('input') + 'corelogic-deeds-090402_09/',
        dir_parcels=directory('input') + 'corelogic-taxrolls-090402_05/',
        now=str(now),
        max_sale_price=85e6,  # according to Wall Street Journal
        random_seed=123,
        fraction_test=0.1,
    )


def read_census(control):
    'return dataframe'
    print 'reading census'
    df = pd.read_csv(control.path_in_census, sep='\t')
    return df


def read_geocoding(control):
    'return dataframe'
    print 'reading geocoding'
    df = pd.read_csv(control.path_in_geocoding, sep='\t')
    return df


def best_apn(df, feature_formatted, feature_unformatted):
    '''return series with best apn

    Algo for the R version
     use unformatted, if its all digits
     otherwise, use formatted, if removing hyphens makes a number
     otherwise, use NaN
    '''
    formatted = df[feature_formatted]
    unformatted = df[feature_unformatted]
    if False:
        print unformatted.head()
        print formatted.head()
    if np.dtype(unformatted) == np.int64:
        # every entry is np.int64, because pd.read_csv made it so
        return unformatted
    if np.dtype(unformatted) == np.object:
        return np.int64(unformatted)
    print 'not expected'
    pdb.set_trace()


def parcels_derived_features(parcels_df):
    'return dict containing sets with certain features of the location'
    def make_tracts(mask_function):
        mask = mask_function(parcels_df)
        subset = parcels_df[mask]
        r = set(int(item)
                for item in subset[parcels.census_tract]
                if not np.isnan(item))
        return r

    def make_zips(mask_function):

        def truncate(zip):
            'convert possible zip9 to zip5'
            return zip / 10000.0 if zip > 99999 else zip

        mask = mask_function(parcels_df)
        subset = parcels_df[mask]
        r = set(int(truncate(item))
                for item in subset[parcels.zipcode]
                if not np.isnan(item))
        return r

    tracts = {
        'has_commercial': make_tracts(parcels.mask_commercial),
        'has_industry': make_tracts(parcels.mask_industry),
        'has_park': make_tracts(parcels.mask_park),
        'has_retail': make_tracts(parcels.mask_retail),
        'has_school': make_tracts(parcels.mask_school),
    }
    zips = {
        'has_commercial': make_zips(parcels.mask_commercial),
        'has_industry': make_zips(parcels.mask_industry),
        'has_park': make_zips(parcels.mask_park),
        'has_retail': make_zips(parcels.mask_retail),
        'has_school': make_zips(parcels.mask_school),
    }
    return {
        'tracts': tracts,
        'zips': zips
    }


def make_subset(all_samples, control):
    def names(s):
        for name in all_samples.columns:
            if s in name:
                print name

    def below(percentile, series):
        quantile_value = series.quantile(percentile / 100.0)
        r = series < quantile_value
        return r

    a = all_samples
    # set mask value in m to True to keep the observation
    m = {}
    m['one building'] = a[parcels.n_buildings] == 1
    m['one APN'] = a[deeds.has_multiple_apns + '_deed'].isnull()  # NaN => not a multiple APN
    m['assessment total > 0'] = a[parcels.assessment_total] > 0
    m['assessment land > 0'] = a[parcels.assessment_land] > 0
    m['assessment improvement > 0'] = a[parcels.assessment_improvement] > 0
    m['assessment total < max'] = a[parcels.assessment_total] < control.max_sale_price
    m['effective_year_built > 0'] = a[parcels.effective_year_built] > 0
    m['year_built > 0'] = a[parcels.year_built] > 0
    m['effective year >= year built'] = a[parcels.effective_year_built] >= a[parcels.year_built]
    m['latitude known'] = a['G LATITUDE'] != 0
    m['longitude known'] = a['G LONGITUDE'] != 0
    m['land size'] = below(99, a[parcels.land_size])
    m['living size'] = below(99, a[parcels.living_size])
    m['recording date present'] = ~a[deeds.recording_date + '_deed'].isnull()  # ~ => not
    m['price > 0'] = a[deeds.price + '_deed'] > 0
    m['price < max'] = a[deeds.price + '_deed'] < control.max_sale_price
    m['full price'] = deeds.mask_full_price(a, deeds.sale_code + '_deed')
    m['rooms > 0'] = a[parcels.n_rooms] > 0
    m['new or resale'] = deeds.mask_new_construction(a) | deeds.mask_resale(a)
    m['units == 1'] = a[parcels.n_units] == 1

    print 'effect of conditions individually'
    for k, v in m.iteritems():
        removed = len(a) - sum(v)
        print '%30s removed %6d samples (%3d%%)' % (k, removed, 100.0 * removed / len(a))

    mm = reduce(lambda a, b: a & b, m.values())
    total_removed = len(a) - sum(mm)
    print 'in combination, removed %6d samples (%3d%%)' % (total_removed, 100.0 * total_removed / len(a))

    r = a[mm]
    return r


def just_derived(control):
    pdb.set_trace()
    start = time.time()
    df = pd.read_csv(control.path_cache_parcels,
                     nrows=1000 if control.test else None)
    print 'read parcels; time: ', time.time() - start
    print 'parcels shape:', df.shape
    r = parcels_derived_features(df)
    return r


def just_timing(control):
    '''report timing for long-IO operations

    SAMPLE OUTPUT
    read all parcel files : 103 sec
    dump parcels to pickle: 338
    read pickle           : 760
    write parcels to csv  :  87
    read csv engine python: 218
    read csv engine c     :  38
    '''
    if False:
        start = time.time()
        parcels = parcels.read(directory('input'), control.test)
        print 'read parcels:', time.time() - start

    path_csv = '/tmp/parcels.csv'
    path_pickle = '/tmp/parcels.pickle'

    if False:
        start = time.time()
        f = open(path_pickle, 'wb')
        pickle.dump(parcels, f)
        f.close()
        print 'dump parcels in pickle form:', time.time() - start

    if False:
        start = time.time()
        f = open(path_pickle, 'rb')
        pickle.load(f)
        f.close()
        print 'load parcels from pickle file:', time.time() - start

    if False:
        start = time.time()
        parcels.to_csv(path_csv)
        print 'write parcels to csv:', time.time() - start

    start = time.time()
    pd.read_csv(path_csv, engine='python')
    print 'read parcels from csv file, parser=python:', time.time() - start

    start = time.time()
    pd.read_csv(path_csv, engine='c')
    print 'read parcels from csv file, parser=c:', time.time() - start


def read_and_write(read_function, write_path, control):
    start = time.time()
    df = read_function(directory('input'), control.test)
    print 'secs to read:', time.time() - start
    start = time.time()
    df.to_csv(write_path)
    print 'secs to write:', time.time() - start


def reduce_census(census_df):
    'return dictionary: key=census_trace, value=(avg commute, median hh income, fraction owner occupied)'

    def get_census_tract(row):
        fips_census_tract = float(row[census.fips_census_tract])
        census_tract = int(fips_census_tract % 1000000)
        return census_tract

    def get_avg_commute(row):
        def mul(factor):
            return (factor[0] * float(row[census.commute_less_5]) +
                    factor[1] * float(row[census.commute_5_to_9]) +
                    factor[2] * float(row[census.commute_10_to_14]) +
                    factor[3] * float(row[census.commute_15_to_19]) +
                    factor[4] * float(row[census.commute_20_to_24]) +
                    factor[5] * float(row[census.commute_25_to_29]) +
                    factor[6] * float(row[census.commute_30_to_34]) +
                    factor[7] * float(row[census.commute_35_to_39]) +
                    factor[8] * float(row[census.commute_40_to_44]) +
                    factor[9] * float(row[census.commute_45_to_59]) +
                    factor[10] * float(row[census.commute_60_to_89]) +
                    factor[11] * float(row[census.commute_90_or_more]))
        n = mul((1., ) * 12)
        wsum = mul((2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 52.5, 75.0, 120.0))
        return None if n == 0 else wsum / n

    def get_median_household_income(row):
        mhi = float(row[census.median_household_income])
        return mhi

    def get_fraction_owner_occupied(row):
        total = float(row[census.occupied_total])
        owner = float(row[census.occupied_owner])
        return None if total == 0 else owner / total

    d = {}
    # first row has explanationss for column names
    labels = census_df.loc[0]
    if False:
        print 'labels'
        for i in xrange(len(labels)):
            print ' ', labels.index[i], labels[i]
    for row_index in xrange(1, len(census_df)):
        if False:
            print row_index
        row = census_df.loc[row_index]
        if False:
            print row
        ct = get_census_tract(row)
        if ct in d:
            print 'duplicate census tract', ct
            pdb.set_trace()
        ac = get_avg_commute(row)
        mhi = get_median_household_income(row)
        foo = get_fraction_owner_occupied(row)
        if ac is not None and mhi is not None and foo is not None:
            d[ct] = (ac, mhi, foo)
    return d


def make_census_reduced_df(d):
    'convert d[census_tract]=(avg commute, med hh inc, fraction owner occ) to dataframe'
    df = pd.DataFrame({'census_tract': [k for k in d.keys()],
                       'avg_commute': [d[k][0] for k in d.keys()],
                       'fraction_owner_occupied': [d[k][2] for k in d.keys()],
                       'median_household_income': [d[k][1] for k in d.keys()]
                       })
    return df


def just_cache(control):
    'consolidate the parcels and deeds files into 2 csv files'
    # goal: speed up testing but don't use in production

    print 'deeds g al'
    read_and_write(deeds.read_g_al, control.path_cache_base + 'deeds-g-al.csv', control)

    print 'parcels'
    read_and_write(parcels.read, control.path_cache_base + 'parcels.csv', control)


def just_parcels(control):
    print 'parcels'
    read_and_write(parcels.read, control.path_cache_base + 'parcels.csv', control)


def just_subset(control):
    print 'subset'
    all_samples = pd.read_csv(control.path_all_samples,
                              nrows=10000 if control.test else None)
    subset = make_subset(all_samples, control)
    print 'subset shape', subset.shape


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    if control.just:
        if control.just == 'derived':
            just_derived(control)
        elif control.just == 'timing':
            just_timing(control)
        elif control.just == 'cache':
            just_cache(control)
        elif control.just == 'parcels':
            just_parcels(control)
        elif control.just == 'subset':
            just_subset(control)
        else:
            assert False, control.just
        pdb.set_trace()
        print 'DISCARD RESULTS; JUST', control.just
        sys.exit(1)

    # create dataframes
    census_df = read_census(control)
    deeds_g_al_df = deeds.read_g_al(directory('input'),
                                    10000 if control.test else None)
    geocoding_df = read_geocoding(control)
    parcels_df = parcels.read(directory('input'),
                              10000 if control.test else None)

    print 'len census', len(census_df)
    print 'led deeds g al', len(deeds_g_al_df)
    print 'len geocoding', len(geocoding_df)
    print 'len parcels', len(parcels_df)

    derived_features = parcels_derived_features(parcels_df)
    tract_features = derived_features['tracts']
    zip_features = derived_features['zips']

    def print_lens(d):
        for k, v in d.iteritems():
            print ' key %s len %d' % (k, len(v))

    print 'tract_features'
    print_lens(tract_features)
    print 'zip_features'
    print_lens(zip_features)

    parcels_sfr_df = parcels_df[parcels.mask_sfr(parcels_df)]
    print 'len parcels sfr', len(parcels_sfr_df)

    # augment parcels and deeds to include a better APN
    new_column_parcels = best_apn(parcels_sfr_df, parcels.apn_formatted, parcels.apn_unformatted)
    parcels_sfr_df.loc[:, parcels.best_apn] = new_column_parcels  # generates an ignorable warning

    new_column_deeds = best_apn(deeds_g_al_df, deeds.apn_formatted, deeds.apn_unformatted)
    deeds_g_al_df.loc[:, deeds.best_apn] = new_column_deeds

    # reduce census
    census_reduced = reduce_census(census_df)
    census_reduced_df = make_census_reduced_df(census_reduced)

    # join the files
    dp = deeds_g_al_df.merge(parcels_sfr_df, how='inner',
                             left_on=deeds.best_apn, right_on=parcels.best_apn,
                             suffixes=('_deed', '_parcel'))
    dpg = dp.merge(geocoding_df, how='inner', left_on=parcels.best_apn, right_on='G APN')
    dpgc = dpg.merge(census_reduced_df, how='inner',
                     left_on=parcels.census_tract + '_parcel', right_on='census_tract')

    print 'names of column in dpgc dataframe'
    for name in dpgc.columns:
        print ' ', name

    print 'merge analysis'
    print ' input sizes'

    def ps(name, value):
        s = value.shape
        print '  %20s shape (%d, %d)' % (name, s[0], s[1])

    ps('deeds_g_al_df', deeds_g_al_df)
    ps('parcels_sfr_df', parcels_sfr_df)
    ps('geocoding_df', geocoding_df)
    ps('census_reduced_df', census_reduced_df)
    print ' output sizes'
    ps('dp', dp)
    ps('dpg', dpg)
    ps('dpgc', dpgc)

    dpgc.to_csv(control.path_all_samples)

    # form subset by retaining only reasonable values
    subset = make_subset(dpgc, control)
    subset.to_csv(control.path_out_subset)

    # split into test and train
    rs = cross_validation.ShuffleSplit(len(subset),
                                       n_iter=1,
                                       test_size=control.fraction_test,
                                       train_size=None,
                                       random_state=control.random_seed)
    assert len(rs) == 1
    for train_index, test_index in rs:
        print 'len train', len(train_index), 'len test', len(test_index)
        assert len(train_index) > len(test_index)
        train = subset.iloc[train_index]
        test = subset.iloc[test_index]
    train.to_csv(control.path_out_train)
    test.to_csv(control.path_out_test)

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
