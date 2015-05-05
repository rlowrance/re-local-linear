# create files for chart-02-*
# invocations and files created
#  python chart-02X.py makefile -> src/chart-02X.makefile
#  python chart-02X.py data     -> data/working/chart-02X.data
#  python chart-02X.py txt      -> data/working/chart-02X.txt
#  python chart-02X.py txtY     -> data/working/chart-02X-Y.txt

import sys


from Bunch import Bunch
from chart_02_template import chart


def main():
    specs = \
        Bunch(metric='mean-wi10',
              title='Mean of Fraction Within 10 Percent of Actual From Folds',
              model='ols',
              training_periods=['30', '60', '90', '120', '150', '180',
                                '210', '240', '270', '300', '330', '360'],
              feature_sets=['ct', 'ctlog', 't', 'tlog'],
              responses=['price', 'logprice'],
              year='2003on')
    chart(specs=specs,
          argv=sys.argv)


if __name__ == '__main__':
    main()
