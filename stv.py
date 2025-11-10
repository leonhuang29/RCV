import numpy as np
np.set_printoptions(linewidth=np.inf)
from math import floor
from copy import deepcopy
from itertools import combinations_with_replacement as comb
from scipy.optimize import linprog


# helper function for recursion
def vote(candidates, voters, quota, elected, seats):
  # copy structures
  candidates = deepcopy(candidates)
  voters     = deepcopy(voters)
  elected    = deepcopy(elected)

  num_elected = sum(elected)

  # elect candidates
  while (num_elected < seats):

    # if number candidates == number seats
    if len(candidates) == (seats - num_elected):
      for (party, _, _) in candidates:
        elected[party] += 1
      return elected
    
    # calculate votes for each candidate
    votes = [[0, {}] for _ in candidates]
    for voter in voters:
      l = min(abs(voter - c[2]) for c in candidates)
      nearest = [i for i, c in enumerate(candidates) if abs(c[2] - voter) == l]
      for i in nearest:
        # increment the number of votes for candidate
        votes[i][0] += voters[voter] / len(nearest)
        # indicate how many votes we contribute (to proportion)
        votes[i][1][voter] = voters[voter] / len(nearest)
    
    # elect if a candidate exceeds quota
    max_votes = max(v[0] for v in votes)
    winners = [i for i, v in enumerate(votes) if v[0] == max_votes]
    losers  = [i for i, v in enumerate(votes) if v[0] == min(v[0] for v in votes)]
    
    if (max_votes >= quota):
      # if there is only one winner, elect them
      if len(winners) == 1: 
        [winner] = winners
        elected[candidates[winner][0]] += 1
        num_elected += 1
        # remove winner from the running
        candidates.pop(winner)
        # transfer votes
        surplus = max_votes - quota
        factor  = surplus / max_votes
        recv    = votes[winner][1]
        for voter in recv:
          voters[voter] -= recv[voter] * factor
      # otherwise, recurse to find expected value
      else:
        results = []
        for winner in winners:
          # copy structures for recursion
          w_elected    = deepcopy(elected)
          w_candidates = deepcopy(candidates)
          w_voters     = deepcopy(voters)

          # elect candidate
          w_elected[candidates[winner][0]] += 1
          w_candidates.pop(winner)

          surplus = max_votes - quota
          factor  = surplus / max_votes
          recv    = votes[winner][1]
          for voter in recv:
            w_voters[voter] -= recv[voter] * factor
          
          # recurse
          res = vote(w_candidates, w_voters, quota, w_elected, seats)
          results.append(res)

        # average
        return [sum(x)/len(results) for x in zip(*results)]
    else:
      # if there is only one loser, eliminate them
      if len(losers) == 1:
        [loser] = losers
        candidates.pop(loser)
      # otherwise, recurse to find expected value
      else:
        results = []
        for loser in losers:
          # copy structure
          w_candidates = deepcopy(candidates)
          w_candidates.pop(loser)

          # recurse
          res = vote(w_candidates, voters, quota, elected, seats)
          results.append(res)
        # average
        return [sum(x)/len(results) for x in zip(*results)]

  return elected

# parties: list of lists for each parties positions
# voters: dictionary of voter positions with densities
# seats: number of seats up for grabs
def elect(parties, voters, seats):

  # construct candidate list
  candidates = []
  i = 0
  for j, party in enumerate(parties):
    for location in party:
      candidates.append((j, i, location))
      i += 1

  # calculate droop quota
  num_voters = sum(voters.values())
  quota = floor((num_voters) / (seats + 1)) + 1

  return vote(candidates, voters, quota, [0] * len(parties), seats)

# possible: list of possible positions for each party
# size: list of sizes for each party
# voters: dictionary of voter positions with densities
# seats: number of seats up for grabs
def payoff(possible, size, voters, seats):
  strategies = [list(comb(p, s)) for p, s in zip(possible, size)]
  payoff = np.zeros([len(s) for s in strategies] + [len(size)])

  for index in np.ndindex(*[len(s) for s in strategies]):
    # current strategy vector
    s = [strategies[i][index[i]] for i in range(len(size))]

    # compute and update payoffs for each party
    payoff[index] = elect(s, voters, seats)

  return payoff, strategies

def pure_nash(payoff):
  n = payoff.shape[-1]
  shape = payoff.shape[:-1]
  eq = []
  for idx in np.ndindex(*shape):
    for p in range(n):
      others = tuple(slice(None) if i == p else j for i, j in enumerate(idx))
      if payoff[idx + (p,)] < payoff[others + (p,)].max():
        break
    else:
      eq.append(idx)
  return eq

def mixed_nash(payoff):
  assert payoff.shape[-1] == 2, "2-player game required"
  
  A = payoff[..., 0]
  m, n = A.shape

  # LP for "row player"
  c = np.r_[np.zeros(m), -1] # max v
  # deviation constraints (no opponent column gives > v)
  A_dev = np.hstack([-A.T, np.ones((n, 1))])
  b_dev = np.zeros(n)
  # probability constraints (x sums to 1)
  A_prob = np.hstack([np.ones((1, m)), np.zeros((1, 1))])
  b_prob = [1]

  # solve LP
  res = linprog(
    c,
    A_ub=A_dev, b_ub=b_dev,
    A_eq=A_prob, b_eq=b_prob,
    bounds=[(0, 1)] * m + [(None, None)]
  )

  x, v = res.x[:m], res.x[-1]
  y = res.ineqlin.marginals # get dual variables
  y = y / y.sum()
  return x, y, v

def find_nash(possible, size, voters, seats):
  # get payoff matrix and strategies
  results, strategies = payoff(possible, size, voters, seats)

  # find pure nash
  eqs = pure_nash(results)

  print("Pure Nash Equilibria:")
  for eq in eqs:
    print(f" - {[strategies[p][eq[p]] for p in range(len(eq))]}")

  # if mixed nash solvable (2-player zero-sum), find:
  if (results.shape[-1] == 2):
    x, y, v = mixed_nash(results)

    print("\nMixed Nash Equilibrium:")
    print("Party 1 strategy:")
    for s, prob in zip(strategies[0], x):
      if prob > 0: print(f"  {s}: {prob:.3f}")
    print("Party 2 strategy:")
    for s, prob in zip(strategies[1], y):
      if prob > 0: print(f"  {s}: {prob:.3f}")
    print(f"Game value: {v:.3f}")

if __name__ == "__main__":
  possible = [
    [-1, -0.5, 0],   # Party 1 possible positions
    [0, 0.5, 1]      # Party 2 possible positions
  ]
  size = [3, 3]  # each chooses three positions
  voters = {-1: 1, -0.5: 1, 0: 1, 0.5: 1, 1: 1}
  seats = 3

  find_nash(possible, size, voters, seats)
