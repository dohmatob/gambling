from nose.tools import assert_equal

MOVES = ['fold', 'check', 'call', 'raise']
CHANCE_MOVES = ['JJ', 'JQ', 'QJ', 'QQ']


def init_state(player=0, pot=0, bet_size=1., accounts={1: 10., 2: 10},
               middle=False, potted={1: 0, 2: 0}, move=None, norminal=1,
               stage=0, max_stages=1):
    return dict(player=player, pot=pot, bet_size=bet_size, move=move,
                accounts=accounts.copy(), middle=middle, stage=stage,
                potted=potted.copy(), norminal=norminal,
                max_stages=max_stages)


def dead(state):
    if state["move"] == "fold":
        return True
    if state['stage'] == state['max_stages']:
        return True


def can(state, move):
    if state['player'] == 0:
        return move in CHANCE_MOVES
    if move == "fold":
        return True
    if move == "call":
        return state["accounts"][state["player"]] >= state['bet_size']
    if move == "raise":
        return state["accounts"][state["player"]] >= 2 * state['bet_size']
    if move == "check":
        if state["player"] != state["norminal"]:
            return False
        if state["middle"]:
            return False
        return True
    else:
        assert 0


def end_stage(state):
    state['stage_ended'] = True
    state['stage'] += 1


def moves(state):
    if state['player'] > 0:
        return [move for move in MOVES if can(state, move)]
    else:
        assert 0


def handle_fold(state):
    pass


def handle_check(state):
    end_stage(state)


def handle_call(state):
    state['pot'] += state['bet_size']
    state['potted'][state['player']] += state['bet_size']
    state['accounts'][state['player']] -= state['bet_size']


def handle_raise(state):
    state['bet_size'] *= 2.
    handle_call(state)


def next_player(state):
    if state['player'] == 0:
        state['player'] = state["norminal"]
    else:
        state['player'] = 1 + state["player"] % 2


def handle_move(state, move):
    if not can(state, move):
        return False
    state['move'] = move
    eval("handle_%s" % move)(state)
    next_player(state)
    return True


def play(state):
    for move in moves(state):
        s = init_state(**state)
        if handle_move(s, move):
            yield s


def fight(state):
    if not dead(state):
        for s in play(state):
            fight(state)


def test_bet():
    state = init_state(player=1)
    handle_move(state, 'call')
    assert_equal(state['pot'], 1.)
    handle_move(state, 'raise')
    assert_equal(state['pot'], 3.)
    assert_equal(state['potted'][1], 1.)
    assert_equal(state['potted'][2], 2.)


if __name__ == "__main__":
    state = init_state(player=1)
    fight(state)
