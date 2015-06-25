# functions to parse the command line


def get_arg(argv, tag):
    # return value past the tag or None
    for i in xrange(len(argv)):
        if argv[i] == tag:
            return argv[i + 1]
    return None


def has_arg(argv, tag):
    for i in xrange(len(argv)):
        if argv[i] == tag:
            return True
    return False
