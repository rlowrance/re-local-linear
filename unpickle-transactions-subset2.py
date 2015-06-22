# program: convert pickled transactions file to csv

from directory import directory

import cPickle
import pdb

if False:
    pdb.set_trace()

path_in = directory('working') + 'transactions-subset2.pickle'
f = open(path_in, 'rb')
data = cPickle.load(f)
f.close()

# data is a pandas.core.frame.DataFrame
path_out = directory('working') + 'transactions-subset2.csv'
data.to_csv(path_out)

print "ok"
