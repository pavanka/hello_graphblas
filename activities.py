from collections import defaultdict
from dataclasses import dataclass
from graphblas import Matrix, Vector, binary, monoid, op, semiring, unary, indexunary
from typing import List, Tuple, Callable
from IPython import embed

users = [0, 1, 2, 3]
pages = [4, 5, 6]
rels = [7, 8, 9, 10, 11, 12, 13]
timestamps = [14, 15, 16, 17, 18, 19]
total_ids = 20

# map i, j -> (i, j) and also keep both reverse assocs
# (i, j) -> i and (i, j) -> j

# read this as view activity of user 0 and page 4 is stored in row 7 of G4
G1 = Matrix.from_coo(
    [0, 0, 0, 1, 1, 2, 3],
    [4, 5, 6, 5, 6, 5, 4],
    [7, 8, 9, 10, 11, 12, 13],
    nrows=total_ids,
    ncols=total_ids,
)
# read this as, inverted index mapping row 7 in G4 to row 0 in G1
G2 = Matrix.from_coo(
    [7, 8, 9, 10, 11, 12, 13],
    [0, 0, 0, 1, 1, 2, 3],
    1,
    nrows=total_ids,
    ncols=total_ids,
)
# read this as, inverted index mapping row 7 in G4 to col 4 in G1
G3 = Matrix.from_coo(
    [7, 8, 9, 10, 11, 12, 13],
    [4, 5, 6, 5, 6, 5, 4],
    1,
    nrows=total_ids,
    ncols=total_ids,
)

# activity graph
# read this as (user 0, page 4)'s interactions have id = 7
# and 7 has two records one at t=1 (with id 14) one at t=3 (with id 15)
G4 = Matrix.from_coo(
    [7, 7, 8, 11, 13],
    [14, 15, 16, 17, 18],
    # see how 15 occurs before 14, this is unavoidable in real-world and a case for storing full timestamp
    # optionally could subtract the epoch hour this G4 represents for less data storage
    [1003, 1001, 1003, 1002, 1001],
    nrows=total_ids,
    ncols=total_ids,
)

# TODO: figure out if there is a way to treat values as indices more natively rather than SerDe (unlikely imo)


def summarize_visits(users, pages, window, reduction):
    # for all pages j, user i is connected to this extracts
    # i’s visit activities (i, j) → [visit_ts1, visit_ts2,...]
    # [NOTE: G4’s indices are (i, j) and ts_k]
    users_vec = Vector.from_coo(users, 1, size=total_ids).diag()
    pages_vec = Vector.from_coo(pages, 1, size=total_ids).diag()

    # nnz-G1 SERDE-1
    indexes = Vector.from_coo(
        (users_vec @ G1 @ pages_vec).to_coo()[-1], 1, size=total_ids
    )
    rows = indexes.diag() @ G4

    # now we have a vector whose indices are (i, j) and value indicates the number of visits in time window
    visit_activity_in_timewindow = reduction(
        # rows.select(rows.apply(binary.ge, window[0]).apply(binary.le, window[1]))
        rows.select(indexunary.valuege, window[0]).select(indexunary.valuele, window[1])
    )

    # next we need to cast all (i, j) back to the desired shape
    # this result lists the total pages visits by each user in the given time window
    # print(
    #    "user activity in time window\n", (visit_activity_in_timewindow @ G2).to_dict()
    # )
    # this result lists number of visits (from users) per page in the given time window
    # print("page visits in time window\n", (visit_activity_in_timewindow @ G3).to_dict())

    # this is how we can create a new matrix that represents the full G1 but with page visit count as values
    def with_zeros():
        G1_coo = G1.to_coo()
        vec = Vector.from_coo(G1_coo[2], 0, size=total_ids)
        new_vals = vec.ewise_add(visit_activity_in_timewindow, op=op.max)
        # nnz-G1 SERDE-2
        result = Matrix.from_coo(
            G1_coo[0],
            G1_coo[1],
            new_vals.to_coo()[-1],
            nrows=total_ids,
            ncols=total_ids,
        )
        return result

    # this is how we can create a new matrix that represents the subset of G1 (sliced by users and pages input)
    # which have non zero page visits (so output size is smaller that full G1 connectivity)
    def without_zeros():
        # nnz-result SERDE-2
        return Matrix.from_coo(
            (visit_activity_in_timewindow.diag() @ G2).to_coo()[1],
            (visit_activity_in_timewindow.diag() @ G3).to_coo()[1],
            (visit_activity_in_timewindow).to_coo()[-1],
            nrows=total_ids,
            ncols=total_ids,
        )

    print("user -> page assocs\n", G1.to_dicts())
    # print("user -> page assocs with visit activity\n", with_zeros().to_dicts())
    print("user -> page assocs with visit activity\n", without_zeros().to_dicts())


def find_visit_counts(users, pages, window):
    print("summarize user visit counts:\n")
    summarize_visits(
        users, pages, window, lambda x: x.apply(unary.one).reduce_rowwise(op.plus)
    )


def find_latest_visit(users, pages, window):
    # if we want activity ids instead of timestamps
    get_indices = lambda x: x.apply(unary.ss.positionj).reduce_rowwise(op.max)
    # another way => x.ewise_mult(x.apply(unary.one), op.ss.secondj).reduce_rowwise(op.max)

    get_timestamps = lambda x: x.reduce_rowwise(op.max)
    print("summarize latest user visits:\n")
    summarize_visits(users, pages, window, get_timestamps)


find_visit_counts([0, 1], [4, 5, 6], (1001, 1003))
print("-" * 10)
find_latest_visit([0, 1], [4, 5, 6], (1001, 1003))
