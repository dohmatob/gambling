MOVES = ['fold', 'check', 'call', 'raise']


class Game(object):
    def __init__(self, limit=10):
        self.limit = limit

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

    def do_move(self, state, move):
        if move in ["check", "fold"]:
            return False
        elif move in ['raise', 'call']:
            if move == "raise":
                state['bet'] += 1
            state["pot"] += state['bet']
            assert state["pot"] <= self.limit  # double-check move was possible
            return move == "raise"
        else:
            RuntimeError("Shouldn't be here!")

    def move_id(self, p, move):
        if move == "kall":
            move = "call"
        if p:
            move = move.upper()
        return move[0]

    def react(self, state, p, move):
        """Let a player react to the others action."""
        stop = not self.do_move(state, move)
        print state['padding'][:-1] + "+-%s %s" % (
            self.move_id(p, move), dict((k, state[k])
                                        for k in ["rnd", "bet", "pot"]))
        state["padding"] += " "
        if not stop:
            q = (p + 1) % 2
            moves = [m for m in ["fold", "call", "raise"]
                     if self.can_move(state, q, m)]
            for cnt, m in enumerate(moves):
                state_ = state.copy()
                print state["padding"] + "|"
                if cnt == len(moves) - 1:
                    state_["padding"] += " "
                else:
                    state_["padding"] += "|"
                self.react(state_, q, m)
        else:
            # entre second round
            if move != "fold":
                state['rnd'] += 1
                if state['rnd'] < 2:
                    state['norminal'] = 1 - state["norminal"]
                    # print state['padding'] + "|"
                    # print state['padding'] + "+-new rnd"
                    # state["padding"] += " "
                    return self.play_rnd(state)
                return False

    def play_rnd(self, state):
        """Play a new round."""
        state["bet"] = 1
        moves = [m for m in ['check', "fold", "call", "raise"]
                 if self.can_move(state.copy(), state["norminal"], m)]
        for cnt, move in enumerate(moves):
            state_ = state.copy()
            print state["padding"] + "|"
            if cnt == len(moves) - 1:
                state_["padding"] += " "
            else:
                state_["padding"] += "|"
            self.react(state_, state['norminal'], move)

        state['rnd'] += 1
        return True

    def play_game(self):
        """Player a new game."""
        state = dict(bet=1, pot=0, norminal=0, rnd=0, padding="")
        print "^ %s" % (dict((k, state[k]) for k in ["bet", "pot", "rnd"]))
        self.play_rnd(state)

Game(limit=500).play_game()
