MOVES = ['fold', 'check', 'call', 'raise']


class Game(object):
    def __init__(self, limit=10, verbose=2):
        self.limit = limit
        self.verbose = verbose

    def can_check(self, state, player):
        """Check that given player can "check" in given state."""
        return player == state["norminal"] and state["rnd"] < 1.

    def can_call(self, state, _):
        """Check that given player can "call" in given state."""
        return state["pot"] + state["bet"] <= self.limit

    def can_raise(self, state, _):
        """Check that given player can "raise" in given state."""
        return state["pot"] + state["bet"] + 1 <= self.limit

    def can_fold(self, *unused):
        """Check that given player can "fold" in given state."""
        return True  # since player can always fold

    def can_move(self, state, player, move):
        """Check where given player can make given move in given state."""
        assert move in MOVES, move
        if move == "call":
            return self.can_call(state, player)
        elif move == "raise":
            return self.can_raise(state, player)
        elif move == "fold":
            return self.can_fold(state, player)
        elif move == "check":
            return self.can_check(state, player)
        else:
            RuntimeError("Shouldn't be here!")

    def do_move(self, state, p, move):
        if p > 0:
            if move in ["check", "fold"]:
                return False
            elif move in ['raise', 'call']:
                if move == "raise":
                    state['bet'] += 1
                state["pot"] += state['bet']
                assert state["pot"] <= self.limit
                if move == "raise":
                    return True
                else:
                    return p == state['norminal']
        else:
            return True
            # RuntimeError("Shouldn't be here!")

    def move_id(self, state, p, move):
        if p == 0:
            return state["chance_move"]
        else:
            if move.lower() == "call":
                move = "kall"
            if p == 1:
                move = move.upper()
            else:
                move = move.lower()
            return move[0]

    def label(self, state, p, move):
        i = str(self.move_id(state, p, move))
        if self.verbose > 1:
            i += " " + str(dict((k, state[k]) for k in ["rnd", "bet", "pot"]))
        return i

    def react(self, state, p, move):
        """Let a player react to the others action."""
        stop = not self.do_move(state, p, move)
        if self.verbose:
            print state['padding'][:-1] + "+-" + self.label(state, p, move)

        state["padding"] += " "
        if not stop:
            if p > 0:
                q = 3 - p
            else:
                q = 1
            moves = self.get_moves(state, q)
            for cnt, m in enumerate(moves):
                state_ = state.copy()
                if self.verbose:
                    print state["padding"] + "|"
                if cnt == len(moves) - 1:
                    state_["padding"] += " "
                else:
                    state_["padding"] += "|"
                self.react(state_, q, m)
        else:
            # enter second round
            if move != "fold":
                state['rnd'] += 1
                state['norminal'] = 3 - state["norminal"]
                if state['rnd'] < 2:
                    return self.play_rnd(state)
                return False

    def get_moves(self, state, p):
        if p > 0.:
            return [move for move in ['check', "fold", "call", "raise"]
                    if self.can_move(state, p, move)]
        else:  # chance (player 0)
            if state["rnd"] == 0:
                return [(x, y) for x in 'JQ' for y in 'JQ']
            elif state["rnd"] == 1:
                return [x for x in 'JQ' if
                        state['chance_move'].count(x) < 2]
            else:
                assert 0

    def play_rnd(self, state):
        """Play a new round."""
        state["bet"] = 1
        moves = self.get_moves(state, 0)
        for cnt, move in enumerate(moves):
            state_ = state.copy()
            state_["chance_move"] = move
            if self.verbose:
                print state["padding"] + "|"
            if cnt == len(moves) - 1:
                state_["padding"] += " "
            else:
                state_["padding"] += "|"
            self.react(state_, 0, move)

        state['rnd'] += 1
        return True

    def play_game(self):
        """Player a new game."""
        state = dict(bet=1, pot=0, norminal=1, rnd=0, padding="")
        if self.verbose:
            print "^%s" % ("" if self.verbose < 2 else
                           dict((k, state[k]) for k in ["bet", "pot", "rnd"]))
        self.play_rnd(state)

Game(limit=10, verbose=1).play_game()
