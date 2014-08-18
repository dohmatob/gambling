def gen_stage_seqs(limit=2):
    yield ''
    yield 'c'
    for k in ['', 'k']:
        for n in xrange(limit):
            for c in ['f', 'k']:
                yield k + 'r' * n + c


def _gen_game_seqs(max_stages, limit=2):
    if max_stages == 1:
        for sigma in gen_stage_seqs(limit=limit):
            yield sigma
    elif max_stages > 1:
        for sigma in _gen_game_seqs(max_stages - 1, limit=limit):
            yield sigma
            if sigma and sigma[-1] not in ['f', 'c']:
                for future in gen_stage_seqs(limit=limit):
                    if future:
                        yield sigma + "." + future


def gen_game_seqs(*args, **kwargs):
    for seq in _gen_game_seqs(*args, **kwargs):
        if not (len(seq) > 1 and seq.endswith('c')):
            yield seq


def count(s, l):
    if s > 1:
        return 2 * (2 * l * ((2 * l) ** (s - 1) + 1) + 1)
    else:
        return 4 * l + 2

s, l = (2, 4)
seqs = gen_game_seqs(s, limit=l)
for sigma in seqs:
    print "'%s'" % sigma
# assert len(seqs) == len(set(seqs)) == count(s, l)
print count(s, l)
