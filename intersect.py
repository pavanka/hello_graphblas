from collections import defaultdict
from dataclasses import dataclass
from graphblas import Matrix, Vector, monoid, semiring, op
from typing import List, Tuple, Callable


@dataclass
class Node:
    U: Matrix
    P: Matrix

    def apply(self, func, target):
        return func(self.U if target == "U" else self.P)


@dataclass
class Orchestrator:
    total_ids: int
    nodes: List[Node]

    # find friends of start user who like target page
    def execute(self, start_user, target_page):
        # Step 1: Create a one-hot vectors for the start user and target_page
        one_hot_user = Vector.from_coo([start_user], values=[1], size=self.total_ids)
        one_hot_page = Vector.from_coo([target_page], values=[1], size=self.total_ids)

        # Step 2: Find friends of the user
        friends = Vector.from_dict({}, size=self.total_ids)
        friends_parts = [
            node.apply(lambda G: one_hot_user @ G, "U") for node in self.nodes
        ]
        for friends_part in friends_parts:
            friends(op.plus) << friends_part

        # Step 3: Intersect friends with followers of target page
        intersections = [
            node.apply(lambda G: friends.diag() @ G @ one_hot_page, "P")
            for node in self.nodes
        ]
        result = Vector.from_dict({}, size=self.total_ids)
        for intersection in intersections:
            result(op.plus) << intersection

        # Print the output
        print(f"Computed across {len(self.nodes)} nodes")
        print(
            f"{result.reduce(monoid.plus).value or 0} friends of user {start_user} like page {target_page}: {list(result.to_coo()[0])}"
        )


@dataclass
class Setup:
    num_users: int
    num_pages: int
    user_friend_edges: List[Tuple[int, int]]
    user_page_edges: List[Tuple[int, int]]
    sharding_func: Callable[[int], int]

    @property
    def total_ids(self):
        return self.num_users + self.num_pages

    def _make_matrix_from_coo(self, edges):
        return Matrix.from_coo(
            [l for l, _ in edges],
            [r for _, r in edges],
            1,
            nrows=self.total_ids,
            ncols=self.total_ids,
        )

    def _make_friends_graphs(self):
        res = defaultdict(list)
        for user, friend in user_friend_edges:
            res[self.sharding_func(user)].append((user, friend))
        return [self._make_matrix_from_coo(edges) for edges in res.values()]

    def _make_follows_graphs(self):
        res = defaultdict(list)
        for user, page in user_page_edges:
            res[self.sharding_func(user)].append((user, page))
        return [self._make_matrix_from_coo(edges) for edges in res.values()]

    def run(self, start_user, target_page):
        o = Orchestrator(
            total_ids,
            [
                Node(U, P)
                for U, P in zip(
                    self._make_friends_graphs(), self._make_follows_graphs()
                )
            ],
        )
        o.execute(start_user, target_page)


if __name__ == "__main__":
    # Define users and pages
    num_users = 5  # e.g., users 0 to 4
    num_pages = 3  # e.g., pages 5 to 7
    total_ids = num_users + num_pages  # Total number of nodes including pages

    # Define user-to-user (friendship) edges
    user_friend_edges = [
        (0, 1),
        (1, 0),
        (0, 2),
        (2, 0),
        (1, 3),
        (3, 1),
    ]

    # Define user-to-page (follow) edges
    user_page_edges = [(0, 5), (1, 5), (2, 5), (3, 6), (3, 7)]

    def sharding_func(user_id):
        if user_id in [0, 1]:
            return 0
        else:
            return 1

    Setup(num_users, num_pages, user_friend_edges, user_page_edges, sharding_func).run(
        start_user=0, target_page=5
    )
