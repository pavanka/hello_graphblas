from graphblas import semiring, monoid, Matrix, Vector, unary, binary, op
from IPython import embed

# Create the graph and starting vector
"""
1 1 0 0

1 2 0 0
5 0 3 0
0 4 0 0
0 0 0 0
"""

G = Matrix.from_coo([0, 0, 1, 2, 1], [0, 1, 2, 1, 0], [1, 2, 3, 4, 5], nrows=4, ncols=4)
v = Vector.from_coo([0, 1], 1.0, size=4)

print(semiring.max_pair(v @ G[:, 0]))  # for sssp

# print(semiring.plus_times(v @ G))
print(semiring.plus_pair(v @ G))

# this has weird behavior due to pair behaving weirdly, always just returning right it seem
# v.ewise_union(G[:,1], op.pair, left_default=0, right_default=0).reduce(op.plus)

# this behaves differently and like what the semiring does; so just default to this always for learning definitions of behavior
# it does not really substitue to identity of monoid in the right for empty values!!
# persence itself has a meaning, not the value at (i, j)
# v.ewise_mult(G[:, 1], op.pair).reduce(op.plus)

v.reduce(op=binary.any)

func = unary.register_anonymous(lambda x: x + 5)
# print(G.apply(func))
# print(G.apply(lambda x: x+5))

from IPython import embed

# masking example
G.select(G.apply(lambda x: x > 1) == True)
