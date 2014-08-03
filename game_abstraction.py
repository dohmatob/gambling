from nose.tools import assert_true, assert_false, assert_equal

MOVES = ['fold', 'check', 'call', 'raise']
TREE = {}


def init_state(**kwargs):
    """Creates game state, overriding defaults with kwargs."""
    state = {}
    state = state.copy()
    state["player"] = kwargs.get('player', 0)
    state["bet"] = kwargs.get('bet', 1)
    state["pot"] = kwargs.get('pot', 0)
    state["limit"] = kwargs.get('limit', 4)
    state["middle"] = kwargs.get('middle', False)
    state["stage_ended"] = kwargs.get('stage_ended', False)
    state["game_ended"] = kwargs.get('game_ended', False)
    state["stage"] = kwargs.get('stage', 0)
    state["max_stages"] = kwargs.get('max_stages', 1)
    state["padding"] = kwargs.get('padding', "")
    state["move"] = kwargs.get('move', ".")
    return state


def can_start_stage(state):
    """Determines whether a new stage maybe started at the given state."""
    if not state['stage_ended']:
        return False
    return state['stage'] + 1 < state['max_stages'] and not state['game_ended']


def start_stage(state):
    """Invoked to start a new stage.

    Notes
    -----
    Public signals can only be revealed inside this function.
    """
    state['stage'] += 1


def end_stage(state):
    """Ends current stage of game and increment stage counter."""
    state['stage_ended'] = True


def end_game(state):
    """Ends current game."""
    end_stage(state)
    state['game_ended'] = True


def handle_move(state, move):
    """Updates game state by effectiving the given move."""
    if move == "check":
        end_stage(state)
    elif move == "call":
        state['pot'] += state['bet']
        if state["middle"]:
            end_stage(state)
    elif move == "fold":
        end_game(state)
    elif move == "raise":
        state['bet'] += 1
        state['pot'] += state['bet']
    else:
        assert 0

    state['middle'] = True
    state['player'] = 1 - state['player']


def can_check(state):
    """Checks whether current player can "check" in given state."""
    if state["stage"] < state["max_stages"]:
        # can't "check" in the middle of a round
        return not state["middle"]
    else:
        # can't check in last round
        return False


def can_fold(state):
    """Checks whether current player can "fold" in given state."""
    # can always fold
    return True


def can_raise(state):
    """Checks whether current player can "raise" in given state."""
    # can raise if and only if new bet would not exceed limit
    return state["bet"] + 1 <= state['limit']


def can_call(state):
    """Checks whether current player can "call" in given state."""
    assert state['bet'] <= state['limit']  # ensure game is not garbage
    # can always call
    return True


def can(state, move):
    """Checks whether current player can make given move in given state."""
    return eval("can_%s(state)" % move)


def get_moves(state):
    """Returns the possible moves of current player in current state."""
    return [move for move in MOVES if can(state, move)]


def get_move_id(state, move):
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


def test_check_ends_stage():
    for stage in [0, 1]:
        state = init_state(stage=stage)
        handle_move(state, "check")
        assert_true(state['stage_ended'])


def test_call_ends_stage_if_started():
    for stage in [0, 1]:
        for middle in [True, False]:
            state = init_state(stage=stage, middle=middle)
            handle_move(state, "call")
            if middle:
                assert_true(state['stage_ended'])
            else:
                assert_false(state['stage_ended'])
                assert_equal(state['stage'], stage)


def test_fold_ends_stage_and_game():
    for stage in [0, 1]:
        state = init_state(stage=stage)
        handle_move(state, "fold")
        assert_true(state['game_ended'])
        assert_true(state['stage_ended'])


def test_raise_never_ends_game():
    for stage in [0, 1]:
        for middle in [True, False]:
            state = init_state(stage=stage, middle=middle, bet=1, pot=0)
            handle_move(state, "raise")
            assert_false(state['game_ended'])


def test_end_game_ends_stage():
    for stage in [0, 1]:
        state = init_state(stage=stage)
        end_game(state)
        assert_true(state['stage_ended'])


def test_can_check():
    for stage in [0, 1]:
        for middle in [True, False]:
            state = init_state(stage=stage, middle=middle)
            cc = can_check(state)
            if stage > 0:
                assert_false(cc)
            else:
                if middle:
                    assert_false(cc)
                else:
                    assert_true(cc)


def test_can_fold_always():
    for stage in [0, 1]:
        for middle in [True, False]:
            state = init_state(stage=stage, middle=middle)
            assert_true(can_fold(state))


def test_can_call_always():
    for stage in [0, 1]:
        for middle in [True, False]:
            state = init_state(stage=stage, middle=middle)
            assert_true(can_call(state))


def test_can_raise():
    for stage in [0, 1]:
        for middle in [True, False]:
            for bet in xrange(1, 5):
                for limit in xrange(bet, bet + 5):
                    state = init_state(stage=stage, middle=middle, bet=bet,
                                       limit=limit)
                    cr = can_raise(state)
                    if bet + 1 <= limit:
                        assert_true(cr)
                    else:
                        assert_false(cr)


def test_get_moves():
    for stage in [0, 1]:
        for middle in [True, False]:
            for bet in xrange(1, 5):
                for limit in xrange(bet, bet + 5):
                    state = init_state(stage=stage, middle=middle, bet=bet,
                                       limit=limit)
                    moves = get_moves(state)
                    assert_true('call' in moves)
                    assert_true('fold' in moves)
                    if stage > 0:
                        assert_false('check' in moves)
                    else:
                        if middle:
                            assert_false('check' in moves)
                        else:
                            assert_true('check' in moves)
                    if bet + 1 <= limit:
                        assert_true('raise' in moves)


def test_get_move_id():
    for player in [0, 1]:
        state = init_state(player=player)
        for move in MOVES:
            mid = get_move_id(state, move)
            ul = [mid.lower(), mid.upper()]
            assert_equal(mid, ul[player == 1])


def test_fight():
    for max_stages in [1, 2]:
        state = init_state(limit=2, max_stages=max_stages)
        fight(state)
        TREE.clear()


def test_can_start_stage():
    assert_false(can_start_stage(init_state(stage_ended=False)))
    assert_false(can_start_stage(init_state(game_ended=True)))
    assert_true(can_start_stage(init_state(stage_ended=True, max_stages=2,
                                           stage=0)))


def fight(state):
    """Players take turns playing current round until it ends."""
    if state["stage"] == 0. and not state['middle']:
        print " .(bet=%g$,pot=%g$,limit=$%g)" % (
            state['bet'], state['pot'], state['limit'])
        key = (state['padding'], None)
        TREE[key] = state
    moves = get_moves(state)
    state['padding'] += " "
    for cnt, move in enumerate(moves):
        print state['padding'] + "|"
        state_ = init_state(**state)
        if cnt == len(moves) - 1:
            state_['padding'] += " "
        else:
            state_['padding'] += "|"
        mid = get_move_id(state_, move)
        key = (state_['padding'], mid, state["stage"])
        assert key not in TREE
        TREE[key] = state_
        handle_move(state_, move)
        assert state_['limit'] == state['limit']
        print state_['padding'][:-1] + (
            "+-" + "%s(stage=%i,bet=%g$,pot=%g$)%s" % (
                mid, state_["stage"], state_['bet'], state_['pot'],
                "/" if state_["game_ended"] else ""))
        # continue ?
        if state_['stage_ended']:
            if can_start_stage(state_):
                start_stage(state_)
                fight(state_)
        else:
            fight(state_)


if __name__ == '__main__':
    TREE.clear()
    state = init_state(limit=2, max_stages=2)
    if can_start_stage(state) or 1:
        fight(state)
