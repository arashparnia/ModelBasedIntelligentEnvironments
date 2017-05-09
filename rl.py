# https://github.com/NathanEpstein/reinforce

import random
import bisect
import collections


def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def choice(population, weights):
    assert len(population) == len(weights)
    cdf_vals = cdf(weights)
    x = random.random()
    idx = bisect.bisect(cdf_vals, x)
    return population[idx]


weights = [0.6, 0.5, 0.5, 0.9, 0.4]
population = '12345'
counts = collections.defaultdict(int)
for i in range(100):
    counts[choice(population, weights)] += 1

import operator

x = counts
sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
print(sorted_x)

# % test.py
# defaultdict(<type 'int'>, {'A': 3066, 'C': 2964, 'B': 3970})






exit(0)


def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"




from reinforce.learn import MarkovAgent

observations = [
    {'state_transitions': [
        {'state': 'low', 'action': 'climb', 'state_': 'mid'},
        {'state': 'mid', 'action': 'climb', 'state_': 'high'},
        {'state': 'high', 'action': 'sink', 'state_': 'mid'},
        {'state': 'mid', 'action': 'sink', 'state_': 'low'},
        {'state': 'low', 'action': 'sink', 'state_': 'bottom'}
    ],
        'reward': 0
    },
    {'state_transitions': [
        {'state': 'low', 'action': 'climb', 'state_': 'mid'},
        {'state': 'mid', 'action': 'climb', 'state_': 'high'},
        {'state': 'high', 'action': 'climb', 'state_': 'top'},
    ],
        'reward': 0
    }
]

trap_states = [
    {
        'state_transitions': [
            {'state': 'bottom', 'action': 'sink', 'state_': 'bottom'},
            {'state': 'bottom', 'action': 'climb', 'state_': 'bottom'}
        ],
        'reward': 0
    },
    {
        'state_transitions': [
            {'state': 'top', 'action': 'sink', 'state_': 'top'},
            {'state': 'top', 'action': 'climb', 'state_': 'top'},
        ],
        'reward': 1
    },
]

mark = MarkovAgent(observations + trap_states)
mark.learn()

print(mark.policy)
# {'high': 'climb', 'top': 'sink', 'bottom': 'sink', 'low': 'climb', 'mid': 'climb'}
# NOTE: policy in top and bottom states is chosen randomly (doesn't affect state)
