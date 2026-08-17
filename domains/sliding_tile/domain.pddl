(define (domain sliding-tile)
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
