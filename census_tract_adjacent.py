'''determine census tracts that are nearyby a given census tract

INPUT FILES
 INPUT/brown-edu/tract00co/nlist_2000.csv
 INPUT/brown-edu/tract00co/tract_2000.csv

OUTPUT FILE
 WORKING/census_tract_adjacent.pickle,  a dict
'''

import collections
import datetime
import numpy as np
import operator
import pandas as pd
import cPickle as pickle
from pprint import pprint  # for debugging
import random
import pdb
import sys

# import my stuff
from Bunch import Bunch
from directory import directory
from Logger import Logger
import parse_command_line


def usage(msg=None):
    if msg is not None:
        print msg
    print 'usage: python census_nearby.py'
    sys.exit(1)


def make_control(argv):
    # return a Bunch

    if len(argv) not in (1,):
        usage()

    random.seed(123456)
    base_name = argv[0].split('.')[0]
    now = datetime.datetime.now()

    dir_in = directory('input') + 'brown-edu/tract00co/'

    test = False
    debug = False

    return Bunch(
        debug=debug,
        test=test,
        path_log=directory('log') + base_name + '.' + now.isoformat('T') + '.log',
        path_in_nlist=dir_in + 'nlist_2000.csv',
        path_in_tract=dir_in + 'tract_2000.csv',
        path_out=directory('working') + base_name + '.pickle',
        now=str(now),
        california='60',
        los_angeles='370',
    )


def main(argv):
    control = make_control(argv)
    sys.stdout = Logger(control.path_log)
    print control

    nlist = pd.read_csv(control.path_in_nlist)
    tract = pd.read_csv(control.path_in_tract)

    print 'nlist columns', nlist.columns
    print 'tract columns', tract.columns

    assert (tract.FID == tract.NID).all()

    pdb.set_trace()
    fid = nlist.FID
    nid = nlist.NID
    next_ids = collections.defaultdict(set)
    for i in xrange(len(nlist)):
        next_ids[fid[i]].add(nid[i])
    pdb.set_trace()
    print 'len next_ids', len(next_ids)

    # select just los angeles county rows
    x = tract.STATEA == control.california
    y = tract.COUNTYA == control.los_angeles
    lac = tract[x & y]

    # map id -> tract (str)
    tracta = lac.TRACTA
    fid = lac.FID
    tract_of = {fid.iloc[i]: int(tracta.iloc[i])
                for i in xrange(len(fid))}

    # determine adjacent census tracts
    adjacent = {}
    pdb.set_trace()
    for i in xrange(len(lac)):
        # according to ashpd.ca.gov
        # adjacent to 1011.10 are
        #  1011.22, 1012.10, 1012.20, 1014.00, 1031.02, 1034.00
        # adjacent to 1011.22 are
        #  1011.10, 1012.20, 1013.00, 1031.01, 1031.02, 9800.26
        print lac.iloc[i]
        tracta = int(lac.TRACTA.iloc[i])
        fid = lac.FID.iloc[i]
        adj = set()
        for adjacent_id in next_ids[fid]:
            if adjacent_id != fid:
                if adjacent_id in tract_of:
                    # missing, if the adjancent tract is in another county
                    adj.add(tract_of[adjacent_id])
        if tracta in ('101110', '101122'):
            print 'adjacent to', tracta
            print 'are', sorted(list(adj))
            pdb.set_trace()
        adjacent[tracta] = adj
    pdb.set_trace()

    print 'writing', control.path_out
    f = open(control.path_out, 'wb')
    pickle.dump(adjacent, f)
    f.close()

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
        operator.add()

    main(sys.argv)
