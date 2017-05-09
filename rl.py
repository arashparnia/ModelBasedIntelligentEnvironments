# https://github.com/NathanEpstein/reinforce

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
