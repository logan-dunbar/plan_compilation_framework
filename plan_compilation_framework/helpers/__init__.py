from collections import namedtuple

Transition = namedtuple('Transition', ['s', 'a', 'r_p', 's_p', 'd', 'a_d'])
