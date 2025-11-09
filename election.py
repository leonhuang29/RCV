import numpy as np
from collections import Counter, defaultdict
import math

class Election:
    def __init__(self, left_positions, right_positions, voter_positions, seats, voter_densities=None, tol=1e-9):
        """
        left_positions, right_positions: lists of floats (positions on line)
        voter_positions: list of floats (voter positions)
        voter_densities: optional list of floats (same length as voter_positions)
                         specifying how much weight each voter contributes.
                         If None, all voters are equally weighted.
        seats: integer number of seats
        tol: numerical tolerance for tie comparisons
        """
        # represent candidates as tuple keys (party, idx, pos)
        self.candidates = []
        for i, pos in enumerate(left_positions):
            self.candidates.append(("L", i, pos))
        for i, pos in enumerate(right_positions):
            self.candidates.append(("R", i, pos))

        self.voter_positions = list(voter_positions)
        self.voter_densities = (
            np.array(voter_densities, dtype=float)
            if voter_densities is not None
            else np.ones(len(voter_positions))
        )
        assert len(self.voter_densities) == len(self.voter_positions), \
            "voter_densities must match voter_positions in length."

        self.seats = int(seats)
        self.tol = tol

        # number of candidates and voters
        self.n_cand = len(self.candidates)
        self.n_voters = len(self.voter_positions)

    def _build_ballots(self):
        """
        Build rank_groups: list (length n_voters) of tuples of groups, each group is tuple of candidate indices.
        Each voter ranks by increasing distance; ties (within tol) are placed in same group.
        """
        rank_groups = []
        for vpos in self.voter_positions:
            dists = np.array([abs(vpos - c[2]) for c in self.candidates])
            uniq = sorted(set([round(float(d), 9) for d in dists]))
            groups = []
            for d in uniq:
                group = tuple(i for i, dist in enumerate(dists) if abs(dist - d) <= self.tol)
                if group:
                    groups.append(group)
            rank_groups.append(tuple(groups))
        return rank_groups

    def run(self):
        """
        Recursive Ranked Choice (RRC) with voter densities.
        Each voter contributes weight voter_densities[i].
        
        Returns:
            {"left": expected_left_seats, "right": expected_right_seats, "per_candidate": dict(candidate->expected_seats)}
        """
        rank_groups = self._build_ballots()
        n_cand = self.n_cand
        n_voters = self.n_voters
        seats = self.seats
        weights = self.voter_densities

        # initial allocation matrix (weighted by densities)
        alloc = [[0.0]*n_cand for _ in range(n_voters)]
        for vi, groups in enumerate(rank_groups):
            top = groups[0]
            share = 1.0 / len(top)
            for j in top:
                alloc[vi][j] = share

        memo = {}

        def recurse_alloc(alloc_state, active_set, seats_left):
            """
            alloc_state: n_voters x n_cand matrix (each row sums to 1.0)
                alloc[vi][j] = fraction of voter vi’s 1 vote currently assigned to candidate j.
            active_set: indices of currently active candidates
            seats_left: how many seats remain to fill
            """
            active_idxs = tuple(sorted(active_set))
            if seats_left == 0 or not active_idxs:
                return defaultdict(float)
            if len(active_idxs) <= seats_left:
                out = defaultdict(float)
                for j in active_idxs:
                    out[j] += 1.0
                return out

            # compute weighted totals among active candidates
            totals = [0.0]*n_cand
            for vi in range(n_voters):
                row = alloc_state[vi]
                for j in active_idxs:
                    totals[j] += weights[vi] * row[j]

            key = (active_idxs, seats_left, tuple(round(totals[j], 9) for j in active_idxs))
            if key in memo:
                return memo[key].copy()

            max_vote = max(totals[j] for j in active_idxs)
            winners = [j for j in active_idxs if abs(totals[j] - max_vote) <= 1e-9]

            out_acc = defaultdict(float)

            for winner in winners:
                new_alloc = [list(row) for row in alloc_state]
                for vi in range(n_voters):
                    w = new_alloc[vi][winner]
                    if w <= 0:
                        continue
                    new_alloc[vi][winner] = 0.0
                    groups = rank_groups[vi]
                    found_idx = None
                    for gi, g in enumerate(groups):
                        if winner in g:
                            found_idx = gi
                            break
                    next_group = None
                    if found_idx is not None:
                        for g in groups[found_idx+1:]:
                            alive = [c for c in g if c in active_set and c != winner]
                            if alive:
                                next_group = alive
                                break
                    if next_group:
                        share = w / len(next_group)
                        for cid in next_group:
                            new_alloc[vi][cid] += share

                new_active = set(active_set)
                new_active.remove(winner)
                sub = recurse_alloc(new_alloc, new_active, seats_left - 1)
                for j, val in sub.items():
                    out_acc[j] += val / len(winners)
                out_acc[winner] += 1.0 / len(winners)

            memo[key] = out_acc.copy()
            return out_acc

        initial_active = set(range(n_cand))
        winners_expected = recurse_alloc(alloc, initial_active, seats)

        per_candidate = {self.candidates[j]: winners_expected.get(j, 0.0) for j in range(n_cand)}
        left_total = sum(val for cand, val in per_candidate.items() if cand[0] == "L")
        right_total = sum(val for cand, val in per_candidate.items() if cand[0] == "R")

        return {"left": left_total, "right": right_total, "per_candidate": per_candidate}
