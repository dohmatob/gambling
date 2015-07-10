import itertools
import re


ALPHABET = "abcdefghijklm"

with open("kuhn.txt", "r") as fd:
    lines = fd.read().splitlines()
    fd.close()
    n_lines = int(lines[0])
    lines = lines[1:]
    assert n_lines == len(lines)
    for line in lines:
        line = line.split("*")
        size = int(line[0])
        line = line[1:]
        assert size == len(line) - 1
        if size == 0:
            n, k = map(int, re.match("\((.+), (.+)\)", line[0]).groups())
            for cards in itertools.combinations(ALPHABET[:n], k):
                print cards
        print size
