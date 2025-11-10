import numpy as np
from math import floor
from copy import deepcopy
from itertools import combinations_with_replacement as comb


class Election:
    def __init__(self, left_positions, right_positions, voter_positions, seats, voter_densities = None, tol = 1e-9):
        """
        left_positions, right_positions: lists of candidate coordinates
        voter_positions: list of positions (floats)
        voter_weights: list of weights (same length as voter_positions)
        seats: number of available seats
        """
        self.left_positions = list(left_positions)
        self.right_positions = list(right_positions)
        self.voter_positions = list(voter_positions)
        self.voter_weights = (
            [1.0] * len(voter_positions)
            if voter_densities is None
            else list(voter_densities)
        )
        assert len(self.voter_positions) == len(self.voter_weights)
        self.seats = seats

        # Build voter dictionary {position: weight}
        self.voters = {p: w for p, w in zip(self.voter_positions, self.voter_weights)}

        # Build candidate list: (party, idx, position)
        self.candidates = []
        i = 0
        for j, pos in enumerate(self.left_positions):
            self.candidates.append(("L", i, pos))
            i += 1
        for j, pos in enumerate(self.right_positions):
            self.candidates.append(("R", i, pos))
            i += 1

    # -------------------------------
    # Internal recursive STV method
    # -------------------------------
    def _vote(self, candidates, voters, quota, elected):
        candidates = deepcopy(candidates)
        voters = deepcopy(voters)
        elected = deepcopy(elected)

        num_elected = sum(elected.values())

        while num_elected < self.seats:
            # if remaining candidates == remaining seats, elect them all
            if len(candidates) == (self.seats - num_elected):
                for (party, _, _) in candidates:
                    elected[party] += 1
                return elected

            # calculate votes
            votes = [[0, {}] for _ in candidates]
            for voter_pos in voters:
                l = min(abs(voter_pos - c[2]) for c in candidates)
                nearest = [i for i, c in enumerate(candidates) if abs(c[2] - voter_pos) == l]
                for i in nearest:
                    votes[i][0] += voters[voter_pos] / len(nearest)
                    votes[i][1][voter_pos] = voters[voter_pos] / len(nearest)

            max_votes = max(v[0] for v in votes)
            min_votes = min(v[0] for v in votes)
            winners = [i for i, v in enumerate(votes) if v[0] == max_votes]
            losers = [i for i, v in enumerate(votes) if v[0] == min_votes]

            # Elect winners if they meet quota
            if max_votes >= quota:
                # Simple winner
                if len(winners) == 1:
                    [winner] = winners
                    elected[candidates[winner][0]] += 1
                    num_elected += 1
                    surplus = max_votes - quota
                    factor = surplus / max_votes if max_votes > 0 else 0.0
                    recv = votes[winner][1]
                    for voter_pos in recv:
                        voters[voter_pos] -= recv[voter_pos] * factor
                    candidates.pop(winner)
                else:
                    # Tied winners → average over branches
                    results = []
                    for winner in winners:
                        w_elected = deepcopy(elected)
                        w_candidates = deepcopy(candidates)
                        w_voters = deepcopy(voters)

                        w_elected[candidates[winner][0]] += 1
                        w_candidates.pop(winner)

                        surplus = max_votes - quota
                        factor = surplus / max_votes if max_votes > 0 else 0.0
                        recv = votes[winner][1]
                        for voter_pos in recv:
                            w_voters[voter_pos] -= recv[voter_pos] * factor

                        res = self._vote(w_candidates, w_voters, quota, w_elected)
                        results.append(res)
                    # average over outcomes
                    avg = {"L": np.mean([r["L"] for r in results]),
                           "R": np.mean([r["R"] for r in results])}
                    return avg
            else:
                # Eliminate lowest
                if len(losers) == 1:
                    [loser] = losers
                    candidates.pop(loser)
                else:
                    # Multiple tied losers → average over branches
                    results = []
                    for loser in losers:
                        w_candidates = deepcopy(candidates)
                        w_candidates.pop(loser)
                        res = self._vote(w_candidates, voters, quota, elected)
                        results.append(res)
                    avg = {"L": np.mean([r["L"] for r in results]),
                           "R": np.mean([r["R"] for r in results])}
                    return avg

        return elected

    # -------------------------------
    # Public method
    # -------------------------------
    def run(self):
        num_voters = sum(self.voters.values())
        quota = floor(num_voters / (self.seats + 1)) + 1

        elected = {"L": 0, "R": 0}
        res = self._vote(self.candidates, self.voters, quota, elected)

        # per-candidate seat distribution (we don’t track individual candidate wins in this simple model)
        per_candidate = {c: 0.0 for c in self.candidates}
        # all L candidates share L’s seats equally, same for R
        nL = len(self.left_positions)
        nR = len(self.right_positions)
        for c in per_candidate:
            if c[0] == "L":
                per_candidate[c] = res["L"] / nL
            else:
                per_candidate[c] = res["R"] / nR

        return {"left": res["L"], "right": res["R"], "per_candidate": per_candidate}