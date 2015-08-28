'create ege_week.makefile'
import datetime
import pdb
from pprint import pprint
import random


lines = []


def make_lines(date, model, training_days, hp, system):
    verbose = True
    if verbose:
        print date, model, training_days, hp, system
    target_file_prefix = '%s-%s-%03d' % (date, model, training_days)
    target_file_middle = ('-%02d' % hp if model == 'rf' else
                          '-%s-%s' % ('lin', 'lin') if model == 'lr' else  # use lin-lin as representative
                          None)
    target_file_suffix = '-0.pickle'  # use fold 0 as a representative of all the folds
    target_file = target_file_prefix + target_file_middle + target_file_suffix
    target_path = '../data/working/ege_week/%s' % target_file
    lines.append('%s += %s' % (system, target_path))
    lines.append('%s: ege_week.py ../data/working/transactions-subset2.pickle' % target_path)
    recipe_prefix = '\t~/anaconda/bin/python ege_week.py %s --model %s --td %d' % (
        date, model, training_days)
    recipe_suffix = '' if model == 'lr' else ' --hp %s' % hp
    lines.append(recipe_prefix + recipe_suffix)
    if verbose:
        pprint(lines[len(lines) - 3: len(lines)])
    lines.append(' ')


def make_hps(model, test):
    if model == 'lr':
        return (None,)
    assert model == 'rf'
    return (4,) if test else range(1, 27, 1)

test = False
just_2009 = True
random.seed(123)

if test:
    systems = ('roy', 'judith')
else:
    # run 10 jobs on my system and 6 on judiths's
    roy = 10 * ['roy']
    judith = 6 * ['judith']
    systems = roy + judith
    random.shuffle(systems)

dates = [datetime.date(2009, 02, 15)]
if not just_2009:
    for year in (2004, 2005, 2006, 2007, 2008):
        for month in (2, 5, 8, 11):
            dates.append(datetime.date(year, month, 15))

system_index = 0
for date in dates:
    for model in ('lr', 'rf'):
        for training_days in (7,) if test else range(7, 365, 7):
            for hp in make_hps(model, test):
                make_lines(date, model, training_days, hp, systems[system_index])
                system_index = (system_index + 1) % len(systems)

for system in systems:
    lines.append(' ')
    lines.append('system_%s: $(%s)' % (system, system))


f = open('ege_week.makefile', 'w')
for line in lines:
    f.write(line)
    f.write('\n')
f.close()

if __name__ == '__main__':
    if False:
        pdb.set_trace()
