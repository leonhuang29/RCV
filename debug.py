import numpy as np
from math import floor
from copy import deepcopy
from itertools import combinations_with_replacement as comb
from scipy.optimize import linprog

# === VOTING + PAYOFF CODE (unchanged from your version) ===

def vote(candidates, voters, quota, elected, seats):
    candidates = deepcopy(candidates)
    voters = deepcopy(voters)
    elected = deepcopy(elected)
    num_elected = sum(elected)

    while num_elected < seats:

        if len(candidates) == (seats - num_elected):
            for (party, _, _) in candidates:
                elected[party] += 1
            return elected

        votes = [[0, {}] for _ in candidates]
        for voter in voters:
            l = min(abs(voter - c[2]) for c in candidates)
            nearest = [i for i, c in enumerate(candidates) if abs(c[2] - voter) == l]
            for i in nearest:
                votes[i][0] += voters[voter] / len(nearest)
                votes[i][1][voter] = voters[voter] / len(nearest)

        max_votes = max(v[0] for v in votes)
        winners = [i for i, v in enumerate(votes) if v[0] == max_votes]
        losers  = [i for i, v in enumerate(votes) if v[0] == min(v[0] for v in votes)]

        if max_votes >= quota:
            if len(winners) == 1:
                winner = winners[0]
                elected[candidates[winner][0]] += 1
                num_elected += 1
                candidates.pop(winner)
                surplus = max_votes - quota
                factor = surplus / max_votes
                recv = votes[winner][1]
                for voter in recv:
                    voters[voter] -= recv[voter] * (1 - factor)
            else:
                results = []
                for winner in winners:
                    w_elected = deepcopy(elected)
                    w_candidates = deepcopy(candidates)
                    w_voters = deepcopy(voters)

                    w_elected[candidates[winner][0]] += 1
                    w_candidates.pop(winner)

                    surplus = max_votes - quota
                    factor = surplus / max_votes
                    recv = votes[winner][1]
                    for voter in recv:
                        w_voters[voter] -= recv[voter] * (1 - factor)

                    results.append(vote(w_candidates, w_voters, quota, w_elected, seats))

                return [sum(x)/len(results) for x in zip(*results)]
        else:
            if len(losers) == 1:
                candidates.pop(losers[0])
            else:
                results = []
                for loser in losers:
                    w_candidates = deepcopy(candidates)
                    w_candidates.pop(loser)
                    results.append(vote(w_candidates, voters, quota, elected, seats))

                return [sum(x)/len(results) for x in zip(*results)]
    
    return elected


def elect(parties, voters, seats):
    candidates = []
    i = 0
    for j, party in enumerate(parties):
        for location in party:
            candidates.append((j, i, location))
            i += 1

    num_voters = sum(voters.values())
    quota = floor((num_voters) / (seats + 1)) + 1
    return vote(candidates, voters, quota, [0] * len(parties), seats)


def payoff(possible, size, voters, seats):
    strategies = [list(comb(p, s)) for p, s in zip(possible, size)]
    payoff = np.zeros([len(s) for s in strategies] + [len(size)])

    for index in np.ndindex(*[len(s) for s in strategies]):
        s = [strategies[i][index[i]] for i in range(len(size))]
        payoff[index] = elect(s, voters, seats)

    return payoff, strategies


# === DEBUG TEST HARNESS ===

def debug_profile(possible, size, voters, seats, profile):
    results, strategies = payoff(possible, size, voters, seats)
    idx = []

    for p in range(len(size)):
        sorted_profile = tuple(sorted(profile[p]))
        idx.append(strategies[p].index(sorted_profile))
    idx = tuple(idx)

    print("\n=== TESTING CANDIDATE PROFILE ===")
    for p in range(len(profile)):
        print(f"Player {p}: {profile[p]} → payoff {results[idx][p]:.6f}")

    print("\n=== UNILATERAL DEVIATION CHECK ===")

    for p in range(len(size)):
        print(f"\nPlayer {p} deviations:")
        for si, strat in enumerate(strategies[p]):
            if strat == tuple(sorted(profile[p])):
                continue

            dev_idx = list(idx)
            dev_idx[p] = si
            dev_idx = tuple(dev_idx)
            dev_payoff = results[dev_idx][p]
            current = results[idx][p]

            marker = "IMPROVES!" if dev_payoff > current + 1e-12 else ""
            print(f"  {profile[p]} → {strat}, payoff {dev_payoff:.6f} {marker}")


# === RUN DEBUG ===

if __name__ == "__main__":
    possible = [
        [0, 0.5, 1, 1.5, 2, 2.5, 3],     # Party 1
        [-3, -2.5, -2, -1.5, -1, -0.5, 0]  # Party 2
    ]
    size = [3, 3]
    voters = {-3:1, -2.5:1, -2:1, -1.5:1, -1:1, -0.5:1, 0:1,
              0.5:1, 1:1, 1.5:1, 2:1, 2.5:1, 3:1}
    seats = 3

    test_profile = [
        (0, 0, 1.5),       # Player 1 strategy
        (-3, 0, 0)      # Player 2 strategy
    ]

    debug_profile(possible, size, voters, seats, test_profile)
