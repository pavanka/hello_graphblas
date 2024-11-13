# https://python-graphblas.readthedocs.io/en/stable/getting_started/primer.html#sssp-in-python-graphblas

from dataclasses import dataclass
from graphblas import op, semiring, Matrix, Vector
from typing import List


@dataclass
class Node:
    G: Matrix

    def sssp(self, v: Vector) -> Vector:
        return semiring.min_plus(v @ self.G)


@dataclass
class Aggregator:
    nodes: List[Node]

    def sssp(self, v: Vector) -> Vector:
        while True:
            w = v.dup()

            # map: distribute computation to nodes
            transforms = [node.sssp(v) for node in self.nodes]
            # reduce: aggregate sub results
            for transform in transforms:
                v(op.min) << transform

            if v.isequal(w):
                return v
        raise Exception("unreachable")


if __name__ == "__main__":
    v = Vector.from_coo([0], [0.0], size=4)  # start_node

    single_node = Aggregator(
        nodes=[
            Node(
                G=Matrix.from_coo(
                    [0, 0, 1, 1, 2],
                    [1, 2, 2, 3, 3],
                    [2.0, 5.0, 1.5, 4.25, 0.5],
                    nrows=4,
                    ncols=4,
                )
            )
        ]
    )
    print(single_node.sssp(v))

    # assume 2 nodes, data is vertex partitioned such that 0 is in node1 and 1, 2 are in node2
    multi_node = Aggregator(
        nodes=[
            Node(G=Matrix.from_coo([0, 0], [1, 2], [2.0, 5.0], nrows=4, ncols=4)),
            Node(
                G=Matrix.from_coo(
                    [1, 1, 2], [2, 3, 3], [1.5, 4.25, 0.5], nrows=4, ncols=4
                )
            ),
        ]
    )
    print(multi_node.sssp(v))
