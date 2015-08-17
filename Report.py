'Report writer'


class Report(object):
    def __init__(self):
        self.lines = []

    def append(self, line):
        self.lines.append(line)
        print line

    def extend(self, lines):
        for line in lines:
            self.append(line)

    def write(self, path):
        f = open(path, 'w')
        for line in self.lines:
            f.write(line)
            f.write('\n')
        f.close()
