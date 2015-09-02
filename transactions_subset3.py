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
import corelogic
from directory import directory
from Logger import Logger
import parse_command_line


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage: python transactions_subset3.py [--just TAG]'
    print ' TAG: run only a portion of the analysis (used during development)'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    print argv
    if len(argv) > 3:
        usage()

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()

    def cache_path(name):
        return directory('working') + base_name + '-cache-' + name + '.pickle'

    test = False
    debug = False
    cache = True

    return Bunch(
        debug=debug,
        test=test,
        cache=cache,
        just=parse_command_line.default(argv, '--just', None),
        path_cache_base=directory('working') + base_name + '-cache-',
        path_cache_census=cache_path('census'),
        path_cache_deeds=cache_path('deeds'),
        path_cache_geocoding=cache_path('geocoding'),
        path_cache_parcels=cache_path('parcels'),
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
            keep = df[corelogic.is_grant_deed(df) & corelogic.is_arms_length(df)]
            return keep

    print 'reading deeds g al'
    df1 = read_deed(control.dir_deeds_a, 'CAC06037F1.zip')
    if False and control.test:
        return df1
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
                df = pd.read_csv(f, sep='\t', nrows=10000 if control.test else None)
            except:
                print 'exception', sys.exc_info()[0]
            keep = df[corelogic.is_sfr(df)]
            return keep

    print 'reading parcels'
    df1 = read_parcels(control.dir_parcels, 'CAC06037F1.zip')
    if False and control.test:
        return df1
    df2 = read_parcels(control.dir_parcels, 'CAC06037F2.zip')
    df3 = read_parcels(control.dir_parcels, 'CAC06037F3.zip')
    df4 = read_parcels(control.dir_parcels, 'CAC06037F4.zip')
    df5 = read_parcels(control.dir_parcels, 'CAC06037F5.zip')
    df6 = read_parcels(control.dir_parcels, 'CAC06037F6.zip')
    df7 = read_parcels(control.dir_parcels, 'CAC06037F7.zip')
    df8 = read_parcels(control.dir_parcels, 'CAC06037F8.zip')
    df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8])
    return df


def best_apn(unformatted, formatted):
    '''return series with best apn

    Algo for the R version
     use unformatted, if its all digits
     otherwise, use formatted, if removing hyphens makes a number
     otherwise, use NaN
    '''
    if False:
        print unformatted.head()
        print formatted.head()
    if np.dtype(unformatted) == np.int64:
        return unformatted
    if np.dtype(unformatted) == np.object:
        return np.int64(unformatted)
    print 'not expected'
    pdb.set_trace()


class Cache(object):
    def __init__(self, read_from_file, path_cache, control):
        self._read_from_file = read_from_file
        self._path_cache = path_cache
        self._control = control

    def load(self):
        'if cache is present, use it; otherwise read file'
        try:
            f = open(self._path_cache, 'rb')
            data = pickle.load(f)
            f.close()
            return data
        except:
            data = self._read_from_file(self._control)
            self.save(data)
            return data

    def save(self, data):
        pdb.set_trace()
        f = open(self._path_cache, 'wb')
        pickle.dump(data, f)
        f.close()


def parcels_derived_features(parcels, control):
    'return dataframes containing features of the census tract and zip code'
    pdb.set_trace()
    census = parcels['CENSUS TRACT']
    zipcode = parcels['PROPERTY ZIPCODE']


def just_derived(control):
    pdb.set_trace()
    parcels = Cache(read_parcels, control.path_cache_parcels, control).load()
    r = parcels_derived_features(parcels, control)


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


def just_cache(control):
    'consolidate the parcels and deeds files into 2 csv files'
    # goal: speed up testing but don't use in production
    def read_and_write(read_function, write_path):
        start = time.time()
        df = read_function(control)
        print 'secs to read:', time.time() - start
        start = time.time()
        df.to_csv(write_path)
        print 'secs to write:', time.time() - start

    print 'deeds g al'
    read_and_write(read_deeds_g_al, control.path_cache_base + 'deeds-g-al.csv')

    print 'parcels'
    read_and_write(read_parcels, control.path_cache_base + 'parcels.csv')




def main(argv):
    pdb.set_trace()
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
        else:
            assert False, control.just
        pdb.set_trace()
        print 'DISCARD RESULTS; JUST', control.just
        sys.exit(1)

    census = read_census(control)
    deeds_g_al = read_deeds_g_al(control)
    geocoding = read_geocoding(control)
    parcels = read_parcels(control)

    print 'len census', len(census)
    print 'led deeds g al', len(deeds_g_al)
    print 'len geocoding', len(geocoding)
    print 'len parcels', len(parcels)

    tract_features, zip_features = make_geo_features(parcels)

    # augment parcels and deeds to include a better APN
    parcels['BestAPN'] = best_apn(parcels['APN UNFORMATTED'], parcels['APN FORMATTED'])
    deeds_g_al['BestAPN'] = best_apn(deeds_g_al['APN UNFORMATTED'], deeds_g_al['APN FORMATTED'])

    # create zip-code based features

    # create census-tract based features

    # reduce parcels to just SFR parcels
    parcels_sfr = parcels[corelogic.is_sfr(parcels)]

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
