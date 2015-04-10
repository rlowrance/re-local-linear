# create files for chart-04
#  python chart-02.py makefile
#   src/chart-02.makefile
#  python chart-02.py data
#   data/working/chart-02.data
#  python chart-02.py txt SUFFIX
#   data/working/chart-02-SUFFIX.txt
# where
#  SUFFIX in
#   mean-root-mean-squared-errors
#   median-root-median-squared-errors

import sys
import chart_02_04


def main():
    args = sys.argv
    chart_02_04.chart_02_04(chart_id='04',
                            suffix=args[1],
                            specific=args[2] if len(args) == 3 else None)

if __name__ == '__main__':
    main()
