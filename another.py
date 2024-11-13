from collections import defaultdict
from dataclasses import dataclass
from graphblas import Matrix, Vector, binary, monoid, op, semiring, unary
from typing import List, Tuple, Callable
from IPython import embed

users = [0, 1, 2, 3]
pages = [4, 5, 6]
rels = [7, 8, 9, 10, 11, 12, 13]
timestamps = [14, 15, 16, 17, 18, 19]
total_ids = 20

G1 = Matrix.from_coo(
    [0, 0, 0, 1, 1, 2, 3],
    [4, 5, 6, 5, 6, 5, 4],
    [7, 8, 9, 10, 11, 12, 13],
    nrows=total_ids,
    ncols=total_ids,
)

# real part represents row indices, imaginary part represents column indices
mapping = Vector.from_coo(
    [7, 8, 9, 10, 11, 12, 13],
    [0 + 4j, 0 + 5j, 0 + 6j, 1 + 5j, 1 + 6j, 2 + 5j, 3 + 4j],
    size=total_ids,
)

G2 = Matrix.from_coo(
    [7, 7, 8, 11, 13],
    [14, 15, 16, 17, 18],
    [1003, 1001, 1003, 1002, 1001],
    nrows=total_ids,
    ncols=total_ids,
)

## main functions used
# select, apply, reduce_rowwise, diag
## main operators used
# op.max, unary.one, op.plus
## more generally https://python-graphblas.readthedocs.io/en/stable/user_guide/types.html


def summarize_visits(users, pages, window, reduction):
    users_vec = Vector.from_coo(users, 1, size=total_ids).diag()
    pages_vec = Vector.from_coo(pages, 1, size=total_ids).diag()

    indexes = Vector.from_coo(
        (users_vec @ G1 @ pages_vec).to_coo()[-1], 1, size=total_ids
    )
    rows = indexes.diag() @ G2

    visit_activity_in_timewindow = reduction(
        rows.select(rows.apply(binary.ge, window[0]).apply(binary.le, window[1]))
    )

    def with_zeros():
        G1_coo = G1.to_coo()
        vec = Vector.from_coo(G1_coo[2], 0, size=total_ids)
        new_vals = vec.ewise_add(visit_activity_in_timewindow, op=op.max)
        result = Matrix.from_coo(
            G1_coo[0],
            G1_coo[1],
            new_vals.to_coo()[-1],
            nrows=total_ids,
            ncols=total_ids,
        )
        return result

    def without_zeros():
        indices = visit_activity_in_timewindow.ewise_mult(mapping, binary.second)
        rows = indices.to_coo()[1].real.astype(int)
        cols = indices.to_coo()[1].imag.astype(int)
        values = visit_activity_in_timewindow.to_coo()[-1]
        return Matrix.from_coo(
            rows,
            cols,
            values,
            nrows=total_ids,
            ncols=total_ids,
        )

    print("user -> page assocs\n", G1.to_dicts())
    print("user -> page assocs with visit activity\n", without_zeros().to_dicts())


def find_visit_counts(users, pages, window):
    print("summarize user visit counts:\n")
    summarize_visits(
        users, pages, window, lambda x: x.apply(unary.one).reduce_rowwise(op.plus)
    )


def find_latest_visit(users, pages, window):
    get_timestamps = lambda x: x.reduce_rowwise(op.max)
    print("summarize latest user visits:\n")
    summarize_visits(users, pages, window, get_timestamps)


find_visit_counts([0, 1], [4, 5, 6], (1001, 1003))
print("-" * 10)
find_latest_visit([0, 1], [4, 5, 6], (1001, 1003))
