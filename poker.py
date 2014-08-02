from copy import deepcopy
import random
from nose.tools import assert_true, assert_equal, assert_false
import numpy as np


def clone(obj):
    """Quick and dirty technology for cloning an object instance."""
    c = object.__new__(obj.__class__)
    c.__dict__ = deepcopy(obj.__dict__)
    return c


class BasePlayer(object):
    def __init__(self, name='Bob', account=1000.):
        self.name = name
        self.account = account
        self.card = None
        self.flop = None
        self.norminal = False
        self.rnd = 0
        self.env = {}
        self.cards = list('JJQQ')
        self.norminal = False
        self.opponent_last_action = None
        self.history = ""

    def _do(self, action):
        """Execute an action."""
        if action == "call":
            self.history += "K"
            return "call"
        elif action == "fold":
            self.history += "F"
            return "fold"
        elif action == "check":
            self.history += "C"
            return "check"
        elif action == "raise":
            self.history += "R"
            return "raise"
        else:
            assert 0

    def _can_check(self):
        return self.rnd == 0 and self.norminal

    def _can_call(self):
        return self.account >= self.env['bet']

    def _can_raise(self):
        return self.account >= self.env["bet"] + 1.

    def act(self, env):
        """Invoked by dealer, demanding player's action."""
        self.env.update(env)
        if self._can_call():
            return self._do("call")
        else:
            return self._do("fold")

    def opponent_acted(self, action):
        """Invoked by dealer to notify player about opponent's last action."""
        self.opponent_last_action = action

        if action == "call":
            self.history += "k"
        elif action == "fold":
            self.history += "f"
        elif action == "check":
            self.history += "c"
        elif action == "raise":
            self.history += "r"
        else:
            assert 0

    def show_card(self):
        """Reveal private (hole) card to the public."""
        self.history += ".%s" % (self.card)
        return self.card

    def opponent_card(self, ocard):
        """Invoked by dealer to let this player know their opponent's
        private card."""
        self.history += ocard

    def take_card(self, card):
        """Invoked by dealer to give player's private card, face-down."""
        self.card = card

    def see_flop(self, flop):
        """Invoked by dealer (at beginning of 2nd round) to let player
        know what the flop is."""
        self.flop = flop
        self.history += flop

    def collect_money(self, amnt):
        """Invoked by dealer to let player collect gains."""
        self.account += amnt

    def new_rnd(self, norminal=False):
        """Invoked by dealer at beginning of each round."""
        self.rnd += 1
        self.norminal = norminal

    def end_rnd(self):
        """Invoked by dealer to tell player the round is ending."""
        self.history += "."

    def end_game(self, gain):
        """Invoked by dealer to tell player the game is ending."""
        self.account += gain
        self.history += "+"

    def new_game(self):
        """Invoked by dealer at beginning of each game."""
        self.flop = None  # flop has not yet been made public


class Dull(BasePlayer):
    def act(self, env):
        self.env.update(env)
        p = random.random()
        if p < .5:
            return self._do("fold")
        else:
            if self._can_check():
                return self._do("check")
            else:
                return self._do("call")


class Crazy(BasePlayer):
    def act(self, env):
        self.env.update(env)
        p = random.random()
        if p < .6 and self._can_call():
            return self._do("call")
        elif p < .8 and self._can_raise:
            return self._do("raise")
        else:
            return self._do("fold")


class Wiseman(BasePlayer):
    """The strategy of such a player is the following:
    At the beginning of a game, the player flips a coin and if it lands
    heads, he guesses that his private card matches the unknown flop.
    Otherwise, he recons his private card is different from the unknown
    flop. Once the flop is revealed. He changes his guess accordingly.

    He acts as follows: If his guess is "match", then he always raises.
    Otherwise he "checks" 50% of the time whenever he can, else he
    "calls" if possible, or elase he "folds" and ends the game.

    """

    def new_game(self):
        super(Wiseman, self).new_game()
        self.hit = random.random() > .5

    def act(self, env):
        if self.flop is not None:
            self.hit = (self.flop == self.card)
        self.env.update(env)
        if self.hit:
            return self._do("raise")
        elif random.random() > .5 and self._can_check():
            return self._do('check')
        elif self._can_call():
            return self._do('call')
        else:
            return self._do("fold")


class Poker(object):
    def __init__(self, players, capital=10., n_games=1):
        self.players = players
        self.check_names()
        self.capital = capital
        self.n_games = n_games
        self.accounts = np.ones(len(players)) * capital
        self.bet = 1.
        self.pot = 0.
        self.potted = np.zeros_like(self.accounts)
        self.norminal_player = 0
        self.rnd = 0
        self.cards = list('JJQQ')
        self.winner = None
        self.flop = None
        self.check_names()
        self.subgames = []

    def check_names(self):
        assert len(set([player.name.lower() for player in self.players])
                   ) == len(self.players)

    def empty_pot(self):
        assert self.potted.sum() == self.pot
        if self.winner is not None:
            self.accounts[self.winner] += self.pot
        else:
            for p, _ in enumerate(self.players):
                self.debit(p, -self.potted[p])
        self.pot *= 0.
        self.potted *= 0.

    def debit(self, p, amnt):
        assert self.accounts[p] >= amnt
        self.accounts[p] -= amnt
        self.pot += amnt
        self.potted[p] += amnt

    def can_check(self, p):
        return self.norminal_player == p

    def can_call(self, p):
        return self.accounts[p] >= self.bet

    def can_raise(self, p):
        return self.accounts[p] >= self.bet + 1.

    def handle_check(self, p):
        assert self.can_check(p)
        self.rnd += 1

    def handle_fold(self, p):
        pass

    def handle_call(self, p):
        pass

    def handle_raise(self, p):
        pass

    def do_flop(self):
        pass

    def show_flop(self):
        print "\t\tDealer: 'The flop is '%s'." % self.flop
        for player in self.players:
            player.see_flop(self.flop)

    def new_rnd(self):
        self.bet = 1.
        for p, player in enumerate(self.players):
            player.new_rnd(norminal=(p == self.norminal_player))
            self.debit(p, 1)

        print "\tRound %i: %s is norminal. Pot = %g$" % (
            self.rnd + 1, self.players[self.norminal_player].name, self.pot)
        if self.rnd == 1.:
            self.show_flop()
        self.rnd += 1

    def share_cards(self):
        self.cards = list('JJQQ')
        random.shuffle(self.cards)
        for player in self.players:
            player.take_card(self.cards.pop())
        self.flop = self.cards.pop()

    def new_game(self):
        self.norminal_player = (random.random() > .5)
        self.rnd = 0
        self.winner = None
        self.pot *= 0.
        self.potted *= 0.
        self.share_cards()
        for player in self.players:
            player.new_game()

    def end_rnd(self):
        if self.rnd == 2:
            print "\t\tDealer: 'Show your private cards, now!'"
            pcards = [player.show_card() for player in self.players]
            for p, pcard in enumerate(pcards):
                print "\t\t%s? '%s'." % (self.players[p].name, pcard)
                self.players[p].opponent_card(pcards[(p + 1) % 2])
            if pcards[0] == self.flop:
                self.winner = 0
            elif pcards[1] == self.flop:
                self.winner = 1

    @property
    def history(self):
        return self.players[0].history

    def play_rnd(self):
        self.new_rnd()
        p = self.norminal_player
        for p in [p, (p + 1) % 2]:
            oppenent = (p + 1) % 2
            player = self.players[p]
            action = player.act(dict(bet=self.bet, pot=self.pot))
            if action == "fold":
                self.handle_fold(p)
            elif action == "call":
                self.handle_call(p)
            elif action == "check":
                self.handle_check(p)
            elif action == "raise":
                self.handle_raise(p)
            else:
                assert 0
            print "\t\t%s? %s (pot = %g$)." % (self.players[p].name, action,
                                              self.pot)
            self.players[oppenent].opponent_acted(action)
            if action == "fold":
                self.winner = oppenent
                return False

        self.end_rnd()

        self.norminal_player = (self.norminal_player + 1) % 2
        return True

    def end_game(self):
        if self.winner is None:
            gain = 0.
            print "\t\tDealer: 'Guys, its a draw. Have your stakes back.'"
            self.accounts += self.potted
        else:
            gain = self.potted[(self.winner + 1) % 2]
            print "\t\tDealer: 'Winner is: %s (wins %g$ net)!'" % (
                self.players[self.winner].name, gain)
            self.accounts[self.winner] += self.pot
        for p, player in enumerate(self.players):
            player.end_game(gain * (2. * (p == self.winner) - 1.))
        print "_" * 80

    def play_game(self):
        self.new_game()
        while self.rnd < 2:
            if not self.play_rnd():
                break

        assert np.min(self.accounts) >= 0.
        self.end_game()
        assert self.accounts.sum() == self.capital * len(self.players)

    def game_over(self):
        pass

    def run(self):
        for g in xrange(self.n_games):
            print "Game %i/%i: %s has %g$ net. %s has %g$ net." % (
                g + 1, self.n_games, self.players[0].name,
                self.accounts[0], self.players[1].name, self.accounts[1])
            self.play_game()
        print ("End of games. %s leaves with capital %g$ net. %s"
               " leaves with capital %g$ net." % (self.players[0].name,
                                                  self.accounts[0],
                                                  self.players[1].name,
                                                  self.accounts[1]))
        return self


def test_player_methods():
    p = BasePlayer()
    for method in ['act', 'show_card', 'take_card', 'see_flop', 'new_rnd']:
        assert_true(hasattr(p, method))


def test_player_show_card():
    p = BasePlayer()
    card = 'J'
    p.take_card(card)
    assert_equal(card, p.show_card())


def test_player_see_flop():
    p = BasePlayer()
    flop = 'Q'
    p.see_flop(flop)
    assert_equal(flop, p.flop)


def test_player_act():
    p = BasePlayer()
    assert_true(p.act(dict(pot=2, bet=1., rnd=0.)) in ["fold", "check",
                                                       "call", "raise"])


def test_player_new_rnd():
    p = BasePlayer()
    p.new_rnd(norminal=True)
    assert_true(p.norminal)
    p.new_rnd(norminal=False)
    assert_false(p.norminal)


def test_player_collect_money():
    capital = 100.
    p = BasePlayer(account=capital)
    assert_equal(p.account, capital)
    p.collect_money(+5)
    assert_equal(p.account, capital + 5)


def test_poker_methods():
    poker = Poker([BasePlayer(name="a"), BasePlayer(name="b")])
    for method in ["handle_check", "do_flop", "handle_call", "handle_raise",
                   "can_check", "can_call", "can_raise", "game_over",
                   "play_rnd", "play_game", "new_game", "new_rnd", "run"]:
        assert_true(hasattr(poker, method),
                    msg="Poker doesn't implement method '%s'" % method)


def test_poker_run():
    poker = Poker([Crazy(name="Alice"), Dull(name="Bob")], n_games=2)
    poker.run()


if __name__ == "__main__":
    import sys
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    capital = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.
    players = [Wiseman(name="Alice"), Crazy(name="Bob")]
    poker = Poker(players, capital=capital, n_games=n_games)
    poker.run()
    print poker.players[0].history
