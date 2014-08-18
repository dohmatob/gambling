from nose.tools import assert_true, assert_false, assert_equal

MOVES = ['fold', 'check', 'call', 'raise']


def init_state(**kwargs):
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
    state["padding"] = kwargs.get('padding', "")
    return state


def end_stage(state):
    state['stage_ended'] = True
    state['stage'] += 1


def end_game(state):
    end_stage(state)
    state['game_ended'] = True


def handle_move(state, move):
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
    if state['stage'] == 1:
        # can only "check" in first round
        return False
    elif state['stage'] == 0:
        # can't "check" in the middle of a round
        return not state['middle']
    else:
        assert 0


def can_fold(state):
    # can always fold
    return True


def can_raise(state):
    # can raise if and only if new bet would not exceed limit
    return state["bet"] + 1 <= state['limit']


def can_call(state):
    assert state['bet'] <= state['limit']  # ensure game is not garbage
    # can always call
    return True


def can(state, move):
    return eval("can_%s(state)" % move)


def get_moves(state):
    return [move for move in MOVES if can(state, move)]


def get_move_id(state, move):
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
        assert_equal(state['stage'], stage + 1)


def test_call_ends_stage_if_started():
    for stage in [0, 1]:
        for middle in [True, False]:
            state = init_state(stage=stage, middle=middle)
            handle_move(state, "call")
            if middle:
                assert_true(state['stage_ended'])
                assert_equal(state['stage'], stage + 1)
            else:
                assert_false(state['stage_ended'])
                assert_equal(state['stage'], stage)


def test_fold_ends_stage_and_game():
    for stage in [0, 1]:
        state = init_state(stage=stage)
        handle_move(state, "fold")
        assert_true(state['game_ended'])
        assert_true(state['stage_ended'])
        assert_equal(state['stage'], stage + 1)


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
        assert_equal(state['stage'], stage + 1)


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
            assert_equal(mid, ul[player == 0])


def fight(state):
    if not state['middle']:
        print " .(bet=%g$,pot=%g$, limit=$%g)" % (
            state['bet'], state['pot'], state['limit'])
    moves = get_moves(state)
    state['padding'] += " "
    for cnt, move in enumerate(moves):
        print state['padding'] + "|"
        state_ = init_state(**state)
        if cnt == len(moves) - 1:
            state_['padding'] += " "
        else:
            state_['padding'] += "|"
        handle_move(state_, move)
        assert state_['limit'] == state['limit']
        print state_['padding'][:-1] + "+-" + "%s(bet=%g$,pot=%g$)%s" % (
            get_move_id(state_, move), state_['bet'], state_['pot'],
            "/" if state_["stage_ended"] else "")
        if not state_['stage_ended']:
            fight(state_)


fight(init_state(limit=3))
