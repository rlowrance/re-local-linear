# create files for chart-05
#  python chart-05.py makefile
#   src/chart-05.makefile
#  python chart-05.py data
#   data/working/chart-05.data
#  python chart-05.py txt SUFFIX
#   data/working/chart-05-SUFFIX.txt
# where
#  SUFFIX in
#   median-root-median-squared-errors

import pdb
import sys


from chart_02_04_05 import chart


def main():
    model = 'med'
    predictors = ['act', 'actlog', 'ct', 'ctlog']
    responses = ['price', 'logprice']
    year = '2008'

    args = sys.argv
    chart(model=model,
          predictors=predictors,
          responses=responses,
          year=year,
          chart_id='05',
          suffix=args[1],
          specific=args[2] if len(args) == 3 else None)

if __name__ == '__main__':
    if False:
        pdb.set_trace()
    main()
