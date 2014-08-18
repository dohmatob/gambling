import random
import numpy as np


class Player(object):
    def __init__(self):
        self.card = None  # private (hidden) card
        self.flop_card = None  # public hard to be made known by dealer
        self.norminal = False

    def take_card(self, card):
        """Take "hole" (face-down) card from dealer."""
        self.card = card

    def see_flop(self, flop_card):
        """See flop unrevealed by dealer to everybody."""
        self.flop_card = flop_card

    def show_card(self):
        """Reveal the hidden (private) card."""
        return self.card

    def act(self, env):
        return "flop"

    def new_round(self, norminal=False):
        self.norminal = norminal

    def can_check(self, env):
        return env["rnd"] == 1 and self.norminal


class Dull(Player):
    def act(self, env):
        if self.can_check(env):
            return "check"
        else:
            return "call"


class Crazy(Player):
    def act(self, _):
        p = random.random()
        return "call" if p < .6 else ("raise" if p < .8 else "fold")


class Poker(object):
    """Simplified Texas Hold'em Poker as explained in the article:
    http://www.contrib.andrew.cmu.edu/~qihangl/firstsummerpaper_finaldraft.pdf,
    page 2.

    """

    def __init__(self, player1, player2, capital=100., n_games=1):
        self.players = [player1, player2]
        self.n_games = n_games
        self.capital = capital  # initial capital of each player
        self.pot = 0.
        self.accounts = np.ones(len(self.players)) * capital
        self.norminal_player = 0
        self.potted = np.zeros_like(self.accounts)
        self.winner = None  # who won last game
        self.net_winner = None  # who walks out the house with net profit
        self.bet = 1.  # current bet
        self.rnd = 0  # round #
        self.cards = list('JJQQ')  # deck
        random.shuffle(self.cards)

        # deal
        for player in self.players:
            player.take_card(self.cards[-1])
            self.cards = self.cards[:-1]
        assert len(self.cards) == 2, self.cards
        self.flop_card = self.cards[-2]
        self.final_card = self.cards[-1]

    def new_round(self):
        """Starts a new round of betting."""
        # everybody puts 1$ into pot
        self.accounts -= 1.
        self.potted += 1.
        self.pot += 2.

        self.rnd += 1
        print ("\tRound %i: Norminal player: player %i" % (
                self.rnd, self.norminal_player + 1))
        print "\tEach player puts 1$ into pot. Pot = %g$." % self.pot

        self.norminal_player += 1
        self.norminal_player %= 2
        for p, player in enumerate(self.players):
            player.new_round(norminal=(p == self.norminal_player))

    def finish_game(self):
        """Finishes a game, updating the winner's account accordingly."""
        if self.winner is not None:
            self.accounts[self.winner] += self.pot
            print "\t\tDealer: 'winner is player %i; they win %g$'" % (
                self.winner + 1, self.pot)
        else:
            print ("\t\tDealer: 'Oops! Looks like we got a draw! Take"
                   " your stakes back'")
            self.accounts += self.potted

    def handle_fold(self, p):
        """Handle player p's "fold" action."""
        self.winner = (p + 1) % 2
        self.finish_game()
        return True

    def handle_call(self, p):
        """Handle player p's "call" action."""
        self.potted[p] += self.bet
        self.pot += self.bet
        self.accounts[p] -= self.bet

    def handle_raise(self, p):
        """Handle player p's "raise" action."""
        self.bet += 1
        self.pot += self.bet
        self.potted[p] += self.bet
        self.accounts[p] -= self.bet

    def can_check(self, p):
        return self.rnd == 1 and p == self.norminal_player

    def handle_check(self, p):
        assert self.can_check(p)

    def do_flop(self):
        """Reveal a public card to all players."""
        print "\t\tDealer: 'The flop is %s. We'll move to the next round'" % (
            self.flop_card)
        for player in self.players:
            player.see_flop(self.flop_card)

    def play_round(self):
        """
        Player a round.

        Returns
        -------
        False if game must end; True if game continues.
        """
        assert self.rnd in [0, 1]
        self.new_round()
        only_raises = True
        p = self.norminal_player
        while only_raises:
            player = self.players[p]
            action = player.act(dict(rnd=self.rnd))
            if action == "fold":
                print "\t\tPlayer %i: fold (pot = %g$)" % (p + 1, self.pot)
                self.handle_fold(p)
                return False
            elif action == "call":
                self.handle_call(p)
                only_raises = False
            elif action == "raise":
                self.handle_raise(p)
            elif action == "check":
                self.handle_check(p)
                print "\t\tPlayer %i: check (pot = %g$)" % (p + 1,
                                                            self.pot)
                if self.rnd == 1:
                    break
                continue
            else:
                raise RuntimeError("Unknown action '%s' for player %i" % (
                        action, p + 1))
            print "\t\tPlayer %i: %s (pot = %g$)" % (p + 1, action,
                                                     self.pot)
            p = (p + 1) % 2

        # post-treatment for this round
        if self.rnd == 1:
            self.do_flop()
        elif self.rnd == 2:
            print "\t\tDealer: 'Show your private cards now!'"
            card1, card2 = [player.show_card() for player in self.players]
            print "\t\tPlayer 1: %s" % card1
            print "\t\tPlayer 2: %s" % card2
            if card1 == self.flop_card:
                self.winner = 0
            if card2 == self.flop_card:
                self.winner = 1
            self.finish_game()

        return True

    def initialize_game(self):
        self.rnd = 0.
        self.pot = 0.
        self.bet = 1.
        self.potted *= 0.
        self.norminal_player = 0

    def play_game(self):
        """Play a game."""
        self.initialize_game()
        for _ in xrange(2):
            if not self.play_round():
                break

    def run(self):
        """Run poker."""
        for g in xrange(self.n_games):
            if np.min(self.accounts) <= 0.:
                break
            print ("Game %i/%i: player 1 has %g$ net; player 2 has %g$ "
                   "net." % (g + 1, self.n_games, self.accounts[0],
                             self.accounts[1]))
            self.play_game()
            assert self.accounts.sum() == self.capital * len(self.players)
            print "_" * 80

        print ("Player 2 ends with %g$ net. Player 2 ends with %g$ net." % (
               self.accounts[0], self.accounts[1]))
        self.net_winner = np.argmax(self.accounts)
        gain = self.accounts[self.net_winner] - self.capital
        print "Net winner is: Player %i (+%g$ net)." % (self.net_winner, gain)

if __name__ == "__main__":
    import sys
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 10.
    capital = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.
    players = [Crazy(), Dull()]
    poker = Poker(*players, capital=capital, n_games=n_games)
    poker.run()
