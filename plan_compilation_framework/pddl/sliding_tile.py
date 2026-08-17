"""Sliding-tile (n-puzzle) PDDL domain and instance generator.

The 15-puzzle of the ECML'22 experiments is not in the IPC benchmark set, so
it is generated here: a 4x4 board, tiles t1..t15, one blank; `move` slides a
tile into the adjacent blank. Instances are scrambled from the goal by a
random walk of `scramble` moves, which guarantees solvability. The PCF driver
then takes its own random walk from that state each episode.

  uv run python -m plan_compilation_framework.pddl.sliding_tile OUT_DIR --size 4 --scramble 80 --seed 0
"""
from __future__ import annotations

import argparse
import os
import random


def domain_pddl() -> str:
  return """(define (domain sliding-tile)
  (:requirements :strips :typing)
  (:types tile position)
  (:predicates
    (at ?t - tile ?p - position)
    (blank ?p - position)
    (adjacent ?p1 - position ?p2 - position))
  (:action move
    :parameters (?t - tile ?from - position ?to - position)
    :precondition (and (at ?t ?from) (blank ?to) (adjacent ?from ?to))
    :effect (and (at ?t ?to) (blank ?from) (not (at ?t ?from)) (not (blank ?to)))))
"""


def _pos(r, c):
  return f'p{r}{c}'


def _neighbours(r, c, n):
  for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
    rr, cc = r + dr, c + dc
    if 0 <= rr < n and 0 <= cc < n:
      yield rr, cc


def scrambled_board(n: int, scramble: int, rng: random.Random):
  """Goal board is tiles 1..n^2-1 in reading order, blank last; random-walk it."""
  board = [[r * n + c + 1 for c in range(n)] for r in range(n)]
  board[n - 1][n - 1] = 0
  br, bc = n - 1, n - 1
  last = None
  for _ in range(scramble):
    options = [p for p in _neighbours(br, bc, n) if p != last]
    rr, cc = rng.choice(options)
    board[br][bc], board[rr][cc] = board[rr][cc], 0
    last = (br, bc)
    br, bc = rr, cc
  return board


def problem_pddl(name: str, n: int, board) -> str:
  tiles = ' '.join(f't{i}' for i in range(1, n * n))
  positions = ' '.join(_pos(r, c) for r in range(n) for c in range(n))
  init = []
  for r in range(n):
    for c in range(n):
      v = board[r][c]
      init.append(f'(blank {_pos(r, c)})' if v == 0 else f'(at t{v} {_pos(r, c)})')
  for r in range(n):
    for c in range(n):
      for rr, cc in _neighbours(r, c, n):
        init.append(f'(adjacent {_pos(r, c)} {_pos(rr, cc)})')
  goal = [f'(at t{r * n + c + 1} {_pos(r, c)})' for r in range(n) for c in range(n)
          if not (r == n - 1 and c == n - 1)]
  return (f'(define (problem {name})\n  (:domain sliding-tile)\n'
          f'  (:objects {tiles} - tile {positions} - position)\n'
          f'  (:init\n    ' + '\n    '.join(init) + ')\n'
          f'  (:goal (and ' + ' '.join(goal) + ')))\n')


def write_instance(out_dir: str, size: int = 4, scramble: int = 80, seed: int = 0):
  os.makedirs(out_dir, exist_ok=True)
  rng = random.Random(seed)
  board = scrambled_board(size, scramble, rng)
  dom = os.path.join(out_dir, 'domain.pddl')
  prob = os.path.join(out_dir, f'p{size * size - 1}-s{scramble}-{seed}.pddl')
  with open(dom, 'w') as f:
    f.write(domain_pddl())
  with open(prob, 'w') as f:
    f.write(problem_pddl(f'puzzle{size * size - 1}-{seed}', size, board))
  return dom, prob


if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('out_dir')
  ap.add_argument('--size', type=int, default=4)
  ap.add_argument('--scramble', type=int, default=80)
  ap.add_argument('--seed', type=int, default=0)
  a = ap.parse_args()
  print(write_instance(a.out_dir, a.size, a.scramble, a.seed))
