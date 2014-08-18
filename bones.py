"""
Basic heads-up limit Texas Hold'em

The game has a maximum of 2 rounds. In each round, some public signals
(possible no signal, in a non-final round) are revealed by the dealer
and a stage game played. Let's first define the following betting
actions:

----------------------------------------------------------------------
| ACTION |            DESCRIPTION                |   WHEN POSSIBLE   |
|--------------------------------------------------------------------|
| fold   | throwaway one's cards, thus  ending   | always            |
|        | the game with th opponent as winner   |                   |
|--------------------------------------------------------------------|
| call   | bet the same amount as the running    | always            |
|        | bet amount                            |                   |
|--------------------------------------------------------------------|
| raise  | increase the running bet amount by 1  | if new bet would  |
|        | and bet the result                    | be less than some |
|        |                                       | hard-limit        |
|--------------------------------------------------------------------|
| check  | bet nothing and move to the next      | if first player   |
|        |                                       | for this  round   |
----------------------------------------------------------------------

"""

MOVES = ['fold', 'kall', 'check', 'raise']


def can(move, info):
    if move == "fold":
        return True
    elif move == "kall":
        return True
    elif move == "raise":
        return info['bet'] + 1. <= info['limit']
    elif move == "check":
        return not (info["stage_started"] or info['last_stage'])
    else:
        assert 0


def get_moves(info):
    return [move for move in MOVES if can(move, info)]


def do_move(move, info):
    if move == "fold":
        info["stage_ended"] = True
        info["game_ended"] = True
    elif move == "kall":
        info["pot"] += info["bet"]
        if info["stage_started"]:
            info["stage_ended"] = True
    elif move == "raise":
        info['bet'] += 1
        info["pot"] += info["bet"]
    elif move == "check":
        info['stage_ended'] = True
    else:
        assert 0


def go(info):
    if not info['stage_started']:
        print " ."
    info['player'] = 1 - info['player']
    moves = get_moves(info)
    info['padding'] += " "
    for cnt, move in enumerate(moves):
        info_ = info.copy()
        info_['stage_started'] = True
        print info_['padding'] + "|"
        if cnt == len(moves) - 1:
            info_['padding'] += " "
        else:
            info_['padding'] += "|"
        do_move(move, info_)
        print info_['padding'][:-1] + " +-" + move
        print info_
        if not info_['stage_ended']:
            go(info_)


info = dict(padding="", bet=1., pot=0., limit=2,
            stage_started=False, player=0, last_stage=False,
            stage_ended=False)
go(info)
