from nose.tools import assert_equal
import re


def get_children(key, tree):
    return dict((k, v) for k, v in tree.iteritems()
                if re.match('^%s.$' % key, k))


def test_get_children():
    tree = {'': None, 'a': 1, 'a1': 4, 'a2': ['hey', None], 'b': 5}
    assert_equal(get_children('a', tree), {'a2': ['hey', None], 'a1': 4})
    assert_equal(get_children('b', tree), {})
