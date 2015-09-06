'''create joined files of test and training transactions

INPUT FILES
 INPUT/corelogic-deeds-090402_07/CAC06037F1.zip ...
 INPUT/corelogic-deeds-090402_09/CAC06037F1.zip ...
 INPUT/corelogic-taxrolls-090402_05/CAC06037F1.zip ...
 INPUT/geocoding.txv
 INPUT/neighborhood-data/census.csv

OUTPUT FILES
 WORKING/transactions_subset3.pickle
 WORKING/transactions_subset3_test.pickle
 WORKING/transactions_subset3_train.pickle

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
import sys
import time
import zipfile


from Bunch import Bunch
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
        path_out_all=directory('working') + base_name + '.pickle',
        path_out_test=directory('working') + base_name + '-test.pickle',
        path_out_train=directory('working') + base_name + '-train.pickle',
        dir_deeds_a=directory('input') + 'corelogic-deeds-090402_07/',
        dir_deeds_b=directory('input') + 'corelogic-deeds-090402_09/',
        dir_parcels=directory('input') + 'corelogic-taxrolls-090402_05/',
        now=str(now),
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


def read_deeds_g_al(control):
    'return df containing all the grant, arms-length deeds'
    def read_deed(dir, file_name):
        z = zipfile.ZipFile(dir + file_name)
        assert len(z.namelist()) == 1
        if False:
            for archive_member_name in z.namelist():
                f = z.open(archive_member_name)
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    print row
        for archive_member_name in z.namelist():
            print 'opening deeds archive member', archive_member_name
            f = z.open(archive_member_name)
            try:
                # line 255719 has an stray " that messes up the parse
                skiprows = (255718,) if archive_member_name == 'CAC06037F3.txt' else None
                nrows = 10000 if control.test else None
                df = pd.read_csv(f, sep='\t', nrows=nrows,  skiprows=skiprows)
                # df = pd.read_csv(f, sep='\t', nrows=10000 if control.test else None)
            except:
                print 'exception', sys.exc_info()[0]
                print 'exception', sys.exc_info()
                pdb.set_trace()
                sys.exit(1)
            deeds = Deeds(df)
            mask_keep = deeds.mask_arms_length() & deeds.mask_grant()
            keep = df[mask_keep]
            return keep

    print 'reading deeds g al'
    df1 = read_deed(control.dir_deeds_a, 'CAC06037F1.zip')
    df2 = read_deed(control.dir_deeds_a, 'CAC06037F2.zip')
    df3 = read_deed(control.dir_deeds_a, 'CAC06037F3.zip')
    df4 = read_deed(control.dir_deeds_a, 'CAC06037F4.zip')
    df5 = read_deed(control.dir_deeds_b, 'CAC06037F5.zip')
    df6 = read_deed(control.dir_deeds_b, 'CAC06037F6.zip')
    df7 = read_deed(control.dir_deeds_b, 'CAC06037F7.zip')
    df8 = read_deed(control.dir_deeds_b, 'CAC06037F8.zip')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
    return df


def read_parcels(control):
    'return df containing all parcels (not just single family residences)'
    def read_parcels(dir, file_name):
        z = zipfile.ZipFile(dir + file_name)
        assert len(z.namelist()) == 1
        if False:
            for archive_member_name in z.namelist():
                f = z.open(archive_member_name)
                reader = csv.reader(f, delimiter='\t')
                for row in reader:
                    print row
        for archive_member_name in z.namelist():
            f = z.open(archive_member_name)
            try:
                nrows = 1000 if control.test else None
                df = pd.read_csv(f, sep='\t', nrows=nrows)
            except:
                print 'exception', sys.exc_info()[0]
            return df

    print 'reading parcels'
    df1 = read_parcels(control.dir_parcels, 'CAC06037F1.zip')
    df2 = read_parcels(control.dir_parcels, 'CAC06037F2.zip')
    df3 = read_parcels(control.dir_parcels, 'CAC06037F3.zip')
    df4 = read_parcels(control.dir_parcels, 'CAC06037F4.zip')
    df5 = read_parcels(control.dir_parcels, 'CAC06037F5.zip')
    df6 = read_parcels(control.dir_parcels, 'CAC06037F6.zip')
    df7 = read_parcels(control.dir_parcels, 'CAC06037F7.zip')
    df8 = read_parcels(control.dir_parcels, 'CAC06037F8.zip')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
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
        return unformatted
    if np.dtype(unformatted) == np.object:
        return np.int64(unformatted)
    print 'not expected'
    pdb.set_trace()


def parcels_derived_features(parcels):
    'return dict containing sets with certain features of the location'
    def make_tracts(mask):
        subset = parcels.df[mask]
        r = set(int(item)
                for item in subset[Parcels.name_census_tract]
                if not np.isnan(item))
        return r

    def make_zips(mask):
        def truncate(zip):
            'convert possible zip9 to zip5'
            return zip / 10000.0 if zip > 99999 else zip

        subset = parcels.df[mask]
        r = set(int(truncate(item))
                for item in subset[Parcels.name_zipcode]
                if not np.isnan(item))
        return r

    pdb.set_trace()
    tracts = {
        "has_commercial": make_tracts(parcels.mask_commercial()),
        "has_industry": make_tracts(parcels.mask_industry()),
        "has_park": make_tracts(parcels.mask_park()),
        "has_retail": make_tracts(parcels.mask_retail()),
        "has_school": make_tracts(parcels.mask_school()),
    }
    zips = {
        "has_commercial": make_zips(parcels.mask_commercial()),
        "has_industry": make_zips(parcels.mask_industry()),
        "has_park": make_zips(parcels.mask_park()),
        "has_retail": make_zips(parcels.mask_retail()),
        "has_school": make_zips(parcels.mask_school()),
    }
    return {
        "tracts": tracts,
        "zips": zips
    }


def just_derived(control):
    pdb.set_trace()
    start = time.time()
    df = pd.read_csv(control.path_cache_parcels,
                     nrows=10000 if control.test else None)
    parcels = Parcels(df)
    print 'read parcels; time: ', time.time() - start
    print 'parcels shape:', parcels.df.shape
    r = parcels_derived_features(parcels)
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
        parcels = read_parcels(control)
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
    df = read_function(control)
    print 'secs to read:', time.time() - start
    start = time.time()
    df.to_csv(write_path)
    print 'secs to write:', time.time() - start


def just_cache(control):
    'consolidate the parcels and deeds files into 2 csv files'
    # goal: speed up testing but don't use in production

    print 'deeds g al'
    read_and_write(read_deeds_g_al, control.path_cache_base + 'deeds-g-al.csv', control)

    print 'parcels'
    read_and_write(read_parcels, control.path_cache_base + 'parcels.csv', control)


def just_parcels(control):
    print 'parcels'
    read_and_write(read_parcels, control.path_cache_base + 'parcels.csv', control)


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
        else:
            assert False, control.just
        pdb.set_trace()
        print 'DISCARD RESULTS; JUST', control.just
        sys.exit(1)

    # create dataframes
    census_df = read_census(control)
    deeds_g_al_df = read_deeds_g_al(control)
    geocoding_df = read_geocoding(control)
    parcels_df = read_parcels(control)

    print 'len census', len(census_df)
    print 'led deeds g al', len(deeds_g_al_df)
    print 'len geocoding', len(geocoding_df)
    print 'len parcels', len(parcels_df)
    pdb.set_trace()

    tract_features, zip_features = parcels_derived_features(parcels)

    parcels_sfr_df = parcels_df[parcels.mask_sfr(parcels_df)]

    # augment parcels and deeds to include a better APN
    bestapn = 'BestApn'
    parcels[bestapn] = best_apn(parcels_df, parcel.apn_formatted, parcel.apn_unformatted)
    deeds_g_al[bestapn] = best_apn(deeds_g_al_df, deed.apn_formatted, deed.apn_unformatted)


    # reduce parcels to just SFR parcels
    parcels_sfr = parcels[parcels.mask_sfr(parcels)]

    # join the files
    pdb.set_trace()
    dp = deeds_g_al.merge(parcels, how='inner', on='BestApn', suffices=('_deed', '_parcel'))
    dgp = dp.merge(geocoding, how='inner', left_on='BestApn', right_on='G APN')
    cdgp = dp.merge(census, how='inner', left_on='BestApn', right_on='apn')

    # TODO add zip-code based features
    # TODO add census-based features


    # TODO add in zip-code based features
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
