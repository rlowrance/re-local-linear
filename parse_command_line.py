# functions to parse the command line


def get_arg(argv, tag):
    '''return value or list of values past the tag, or None

    --tag v1 --tag        ==> return v1 as a string
    --tag v1 v2 v3 --tag  ==> return v1 v2 v3 as list of string
    --tag v1 v2 v3        ==> return v1 v2 v3 as list of string
    --tag --tag           ==> return []
                          ==> return None
    '''
    for i in xrange(len(argv)):
        if argv[i] == tag:
            result = []
            i += 1
            while i < len(argv) and argv[i][:2] != '--':
                result.append(argv[i])
                i += 1
            return result[0] if len(result) == 1 else result
    return None


def has_arg(argv, tag):
    'return True iff argv contains the tag'
    for i in xrange(len(argv)):
        if argv[i] == tag:
            return True
    return False


def default(argv, tag, default_value):
    actual = get_arg(argv, tag)
    return default_value if actual is None else actual


if __name__ == '__main__':
    def a(s):
        'convert string to list of strings'
        return s.split(' ')

    # test get_arg
    assert get_arg(a('--x a b c'), '--x') == ['a', 'b', 'c']
    assert get_arg(a('--x a'), '--x') == 'a'
    assert get_arg(a('--x a b --x'), '--x') == ['a', 'b']
    assert get_arg(a('--x --y'), '--y') == []
    assert get_arg(a('--y'), '--x') is None

    # test has_arg
    assert has_arg(a('--x'), '--x')
    assert not has_arg(a('--x'), '--y')

    # default
    assert default(a('--x'), 'x', 'def') == 'def'

    print 'ok'
