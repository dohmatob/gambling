# Author: DOHMATOB Elvis Dopgima

from nose.tools import assert_true, assert_false, assert_equal

MOVES = ['fold', 'check', 'call', 'raise']


class PokerTree:
    def __init__(self, max_rounds=1, limit=2):
        self.max_rounds = max_rounds
        self.limit = limit
        self.tree_dict = {}
        self.node_count = 0.

    def init_state(self, **kwargs):
        """Creates game state, overriding defaults with kwargs."""
        state = {}
        state["player"] = kwargs.get('player', 0)
        state["bet_size"] = kwargs.get('bet_size', 1)
        state["pot"] = kwargs.get('pot', 0)
        state["middle"] = kwargs.get('middle', False)
        state["round_ended"] = kwargs.get('round_ended', False)
        state["hand_ended"] = kwargs.get('hand_ended', False)
        state["round"] = kwargs.get('round', 0)
        state["padding"] = kwargs.get('padding', "")
        state["move"] = kwargs.get('move', ".")  # move that took us here
        state["node_id"] = kwargs.get('node_id', None)
        return state

    def can_start_round(self, state):
        """Determines whether a new round maybe started at the given state."""
        if not state['round_ended']:
            return False
        return state['round'] + 1 < self.max_rounds and not state['hand_ended']

    def start_round(self, state):
        """Invoked to start a new round.

        Notes
        -----
        Public signals can only be revealed inside this function.
        """
        state['round'] += 1

    def end_round(self, state):
        """Ends current round of game and increment round counter."""
        state['round_ended'] = True

    def end_hand(self, state):
        """Ends current game."""
        self.end_round(state)
        state['hand_ended'] = True

    def handle_move(self, state, move):
        """Updates game state by effecting the given move."""
        if move == "check":
            self.end_round(state)
        elif move == "call":
            state['pot'] += state['bet_size']
            if state["middle"]:
                self.end_round(state)
        elif move == "fold":
            self.end_hand(state)
        elif move == "raise":
            state['bet_size'] += 1
            state['pot'] += state['bet_size']
        else:
            assert 0

        state['middle'] = True
        state['player'] = 1 - state['player']

    def can_check(self, state):
        """Checks whether current player can "check" in given state."""
        if state["round"] < self.max_rounds:
            # can't "check" in the middle of a round
            return not state["middle"]
        else:
            # can't check in last round
            return False

    def can_fold(self, state):
        """Checks whether current player can "fold" in given state."""
        # can always fold
        return True

    def can_raise(self, state):
        """Checks whether current player can "raise" in given state."""
        # can raise if and only if new bet would not exceed limit
        return state["bet_size"] + 1 <= self.limit

    def can_call(self, state):
        """Checks whether current player can "call" in given state."""
        assert state['bet_size'] <= self.limit  # ensure game's not garbage
        # can always call
        return True

    def can(self, state, move):
        """Checks whether current player can make given move in given state."""
        return getattr(self, "can_%s" % move)(state)

    def get_moves(self, state):
        """Returns the possible moves of current player in current state."""
        return [move for move in MOVES if self.can(state, move)]

    def get_move_id(self, state, move):
        """Returns the player-wise identifier for the given move."""
        if move == "call":
            move = "kall"
        if state['player'] == 1:  # XXX swapped with 0
            move = move.upper()
        elif state['player'] == 0:
            move = move.lower()
        else:
            assert 0
        return move[0]

    def fight(self, state):
        """Players take turns playing current round until it ends."""
        if state["round"] == 0. and not state['middle']:
            print " .(node_id=%i,bet_size=%g$,pot=%g$,limit=$%g)" % (
                self.node_count, state['bet_size'], state['pot'],
                self.limit)
        moves = self.get_moves(state)
        state['padding'] += " "
        for cnt, move in enumerate(moves):
            print state['padding'] + "|"
            state_ = self.init_state(**state)
            if cnt == len(moves) - 1:
                state_['padding'] += " "
            else:
                state_['padding'] += "|"
            mid = self.get_move_id(state_, move)
            self.handle_move(state_, move)
            self.node_count += 1
            state_['node_id'] = self.node_count
            print state_['padding'][:-1] + (
                "+-" + "%s(node_id=%i,round=%i,bet_size=%g$,pot=%g$)%s" % (
                    mid, state_['node_id'], state_["round"],
                    state_['bet_size'],
                    state_['pot'], "/" if state_["hand_ended"] else ""))
            # continue ?
            if state_['round_ended']:
                if self.can_start_round(state_):
                    self.start_round(state_)
                    self.fight(state_)
            else:
                self.fight(state_)

    def __call__(self):
        self.node_count = 0
        self.tree_dict.clear()
        self.fight(self.init_state())


def test_check_ends_round():
    pt = PokerTree()
    for rnd in [0, 1]:
        state = pt.init_state(**{'round': rnd})
        pt.handle_move(state, "check")
        assert_true(state['round_ended'])


def test_call_ends_round_if_started():
    pt = PokerTree()
    for rnd in [0, 1]:
        for middle in [True, False]:
            state = pt.init_state(**{'round': rnd, 'middle': middle})
            pt.handle_move(state, "call")
            if middle:
                assert_true(state['round_ended'])
            else:
                assert_false(state['round_ended'])
                assert_equal(state['round'], rnd)


def test_fold_ends_round_and_hand():
    pt = PokerTree()
    for rnd in [0, 1]:
        state = pt.init_state(**{'round': rnd})
        pt.handle_move(state, "fold")
        assert_true(state['hand_ended'])
        assert_true(state['round_ended'])


def test_raise_never_ends_hand():
    pt = PokerTree()
    for rnd in [0, 1]:
        for middle in [True, False]:
            state = pt.init_state(round=rnd, middle=middle, bet_size=1, pot=0)
            pt.handle_move(state, "raise")
            assert_false(state['hand_ended'])


def test_end_hand_ends_round():
    pt = PokerTree()
    for rnd in [0, 1]:
        state = pt.init_state(round=rnd)
        pt.end_hand(state)
        assert_true(state['round_ended'])


def test_can_check():
    pt = PokerTree()
    for round in [0, 1]:
        for middle in [True, False]:
            state = pt.init_state(round=round, middle=middle)
            cc = pt.can_check(state)
            if round > 0:
                assert_false(cc)
            else:
                if middle:
                    assert_false(cc)
                else:
                    assert_true(cc)


def test_can_fold_always():
    pt = PokerTree()
    for round in [0, 1]:
        for middle in [True, False]:
            state = pt.init_state(round=round, middle=middle)
            assert_true(pt.can_fold(state))


def test_can_call_always():
    pt = PokerTree()
    for round in [0, 1]:
        for middle in [True, False]:
            state = pt.init_state(round=round, middle=middle)
            assert_true(pt.can_call(state))


def test_can_raise():
    for round in [0, 1]:
        for middle in [True, False]:
            for bet_size in xrange(1, 5):
                for limit in xrange(bet_size, bet_size + 5):
                    pt = PokerTree(limit=limit)
                    state = pt.init_state(round=round, middle=middle,
                                       bet_size=bet_size)
                    cr = pt.can_raise(state)
                    if bet_size + 1 <= limit:
                        assert_true(cr)
                    else:
                        assert_false(cr)


def test_get_moves():
    for round in [0, 1]:
        for middle in [True, False]:
            for bet_size in xrange(1, 5):
                for limit in xrange(bet_size, bet_size + 5):
                    pt = PokerTree(limit=limit)
                    state = pt.init_state(round=round, middle=middle,
                                       bet_size=bet_size, limit=limit)
                    moves = pt.get_moves(state)
                    assert_true('call' in moves)
                    assert_true('fold' in moves)
                    if round > 0:
                        assert_false('check' in moves)
                    else:
                        if middle:
                            assert_false('check' in moves)
                        else:
                            assert_true('check' in moves)
                    if bet_size + 1 <= limit:
                        assert_true('raise' in moves)


def test_get_move_id():
    pt = PokerTree()
    for player in [0, 1]:
        state = pt.init_state(player=player)
        for move in MOVES:
            mid = pt.get_move_id(state, move)
            ul = [mid.lower(), mid.upper()]
            assert_equal(mid, ul[player == 1])


def test_fight():
    for max_rounds in [1, 2]:
        pt = PokerTree(max_rounds=max_rounds, limit=2)
        pt()


def test_can_start_round():
    pt = PokerTree()
    assert_false(pt.can_start_round(pt.init_state(round_ended=False)))
    assert_false(pt.can_start_round(pt.init_state(hand_ended=True)))
    pt.max_rounds = 2
    assert_true(pt.can_start_round(pt.init_state(round_ended=True)))


if __name__ == '__main__':
    pt = PokerTree(limit=2, max_rounds=1)
    pt()
