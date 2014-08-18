def move_id(move, norminal):
    if norminal == 1:
        return move.upper()
    else:
        return move.lower()


def everythx(rnd=0, limit=2, consumed_limit=0, norminal=1, max_rnds=1):
    assert norminal > 0
    if rnd >= max_rnds:
        yield ""
        return
    yield move_id("f", norminal)
    if rnd < max_rnds - 1:
        for s in everythx(rnd=rnd + 1, limit=limit, norminal=1 + norminal % 2,
                          max_rnds=max_rnds, consumed_limit=consumed_limit):
            yield move_id("c", norminal) + ("." + s if s else "")
    for n in xrange(limit - consumed_limit):
        start = "K" if norminal == 1 else "k"
        middle = "rR" if norminal == 1 else "Rr"
        for end in "fk":
            aux = start + middle * n + end
            if not end[-1].lower() == 'f':
                for s in everythx(rnd=rnd + 1, limit=limit,
                                  consumed_limit=consumed_limit + n,
                                  norminal=1 + norminal % 2,
                                  max_rnds=max_rnds):
                    yield aux + ("." + s if s else "")
            else:
                yield aux


for s in everythx(limit=2, max_rnds=2):
    print s
