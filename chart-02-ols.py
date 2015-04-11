# create files for chart-02
#  python chart-02.py makefile
#   src/chart-02.makefile
#  python chart-02.py data
#   data/working/chart-02.data
#  python chart-02.py txt SUFFIX
#   data/working/chart-02-SUFFIX.txt
# where
#  SUFFIX in
#   mean-root-mean-squared-errors  (NOT RUN)
#   median-root-median-squared-errors

import sys
from chart_02_04_05 import chart


def main():
    model = 'ols'
    predictors = ['act', 'actlog', 'ct', 'ctlog']
    responses = ['price', 'logprice']
    year = '2008'

    args = sys.argv
    chart(model=model,
          predictors=predictors,
          responses=responses,
          year=year,
          chart_id='02',
          suffix=args[1],
          specific=args[2] if len(args) == 3 else None)

if __name__ == '__main__':
    main()
