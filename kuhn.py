rules = []
for perm, proba in zip([(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)],
                       [1. / 6] * 6):
    head = [((perm, proba), 1)]
    rules.append(([head, ("pass", 2), ("pass", 3)], (1., None),
                  [head, ("pass", 2), ("bet", 1), (
