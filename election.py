import numpy as np
from collections import defaultdict
from math import floor
import math

class Election:
    def __init__(self, left_positions, right_positions, voter_positions, seats, voter_densities=None, tol=1e-9):
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
        self.seats = int(seats)
        self.tol = tol
        self.n_cand = len(self.candidates)
        self.n_voters = len(self.voter_positions)

    def _build_ballots(self):
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
        rank_groups = self._build_ballots()
        n_cand = self.n_cand
        n_voters = self.n_voters
        seats = self.seats
        weights = self.voter_densities.copy()

        # initial allocation (fractions of each voter's single vote)
        alloc = [[0.0]*n_cand for _ in range(n_voters)]
        for vi, groups in enumerate(rank_groups):
            top = groups[0]
            share = 1.0 / len(top)
            for j in top:
                alloc[vi][j] = share

        total_votes = float(np.sum(weights))
        quota = math.floor(total_votes / (seats + 1)) + 1

        memo = {}

        def stv_recursive(alloc_state, active_set, seats_left):
            """
            alloc_state: n_voters x n_cand array-like where each row is fractions summing to 1 (for remaining prefs)
            active_set: set of active candidate indices
            seats_left: how many seats remain to allocate

            RETURNS: defaultdict(float) mapping candidate_index -> expected number of seats (for the seats_left)
            """
            active_idxs = tuple(sorted(active_set))
            # base cases
            if seats_left == 0 or not active_idxs:
                return defaultdict(float)

            # if fewer or equal candidates than seats_left, give each one seat
            if len(active_idxs) <= seats_left:
                out = defaultdict(float)
                for j in active_idxs:
                    out[j] += 1.0
                return out

            # compute totals
            totals = [0.0]*n_cand
            for vi in range(n_voters):
                row = alloc_state[vi]
                for j in active_idxs:
                    totals[j] += weights[vi] * row[j]

            # memo key: active set, seats left, rounded totals for numeric stability
            key = (active_idxs, seats_left, tuple(round(totals[j], 9) for j in active_idxs))
            if key in memo:
                return memo[key].copy()

            max_vote = max(totals[j] for j in active_idxs)

            out_acc = defaultdict(float)

            # If any candidate(s) meet or exceed quota, process those as winners (handle ties)
            over_quota = [j for j in active_idxs if totals[j] >= quota - self.tol]
            if over_quota:
                # resolve ties among all over-quota candidates (average branches)
                for winner in over_quota:
                    # make copies of alloc_state for branch
                    new_alloc = [list(row) for row in alloc_state]
                    new_active = set(active_set)
                    new_active.remove(winner)

                    total_w = totals[winner]
                    surplus = total_w - quota
                    if surplus > 0:
                        factor = surplus / total_w
                        # for each voter, subtract the fractional part corresponding to winner*factor
                        for vi in range(n_voters):
                            w_share = new_alloc[vi][winner]
                            if w_share <= 0:
                                continue
                            # reduce winner's fraction by w_share * factor
                            reduction = w_share * factor
                            new_alloc[vi][winner] -= reduction
                            # compute the absolute transferable mass (weights[vi] * reduction)
                            transferable_mass = weights[vi] * reduction
                            # find next alive preference group for this voter
                            groups = rank_groups[vi]
                            found_idx = None
                            for gi, g in enumerate(groups):
                                if winner in g:
                                    found_idx = gi
                                    break
                            if found_idx is not None:
                                for g in groups[found_idx+1:]:
                                    alive = [c for c in g if c in new_active]
                                    if alive:
                                        # split transferable mass equally among alive choices
                                        per_cand_mass = transferable_mass / len(alive)
                                        # convert to alloc fractions by dividing by the voter's total weight
                                        # (so alloc entries stay as fractions of the voter's vote)
                                        for cid in alive:
                                            new_alloc[vi][cid] += per_cand_mass / weights[vi]
                                        break
                    # recurse for remaining seats
                    sub = stv_recursive(new_alloc, new_active, seats_left - 1)
                    # aggregate: winner gets 1 seat in this branch plus whatever sub returns
                    for j, val in sub.items():
                        out_acc[j] += val / len(over_quota)
                    out_acc[winner] += 1.0 / len(over_quota)

                memo[key] = out_acc.copy()
                return out_acc

            # Otherwise no one meets quota -> eliminate lowest (handle ties by averaging)
            min_vote = min(totals[j] for j in active_idxs)
            losers_tied = [j for j in active_idxs if abs(totals[j] - min_vote) <= self.tol]

            for loser in losers_tied:
                new_alloc = [list(row) for row in alloc_state]
                new_active = set(active_set)
                new_active.remove(loser)

                # transfer all of loser's fractions
                for vi in range(n_voters):
                    w_share = new_alloc[vi][loser]
                    if w_share <= 0:
                        continue
                    new_alloc[vi][loser] = 0.0
                    transferable_mass = weights[vi] * w_share
                    groups = rank_groups[vi]
                    found_idx = None
                    for gi, g in enumerate(groups):
                        if loser in g:
                            found_idx = gi
                            break
                    if found_idx is not None:
                        for g in groups[found_idx+1:]:
                            alive = [c for c in g if c in new_active]
                            if alive:
                                per_cand_mass = transferable_mass / len(alive)
                                for cid in alive:
                                    new_alloc[vi][cid] += per_cand_mass / weights[vi]
                                break

                sub = stv_recursive(new_alloc, new_active, seats_left)
                for j, val in sub.items():
                    out_acc[j] += val / len(losers_tied)

            memo[key] = out_acc.copy()
            return out_acc

        winners_expected = stv_recursive(alloc, set(range(n_cand)), seats)
        per_candidate = {self.candidates[j]: winners_expected.get(j, 0.0) for j in range(n_cand)}
        left_total = sum(val for cand, val in per_candidate.items() if cand[0] == "L")
        right_total = sum(val for cand, val in per_candidate.items() if cand[0] == "R")

        # sanity check (should sum to seats)
        total_allocated = sum(per_candidate.values())
        if abs(total_allocated - seats) > 1e-6:
            # small numeric tolerance allowed; otherwise warn
            print(f"Warning: total expected seats = {total_allocated}, expected = {seats}")

        return {"left": left_total, "right": right_total, "per_candidate": per_candidate}