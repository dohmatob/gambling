from nose.tools import assert_true, assert_equal, assert_false


def test_preflop():
    players = [Player(), Player()]
    poker = Poker(norminal=1)
    poker.do_round()
