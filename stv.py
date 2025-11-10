import numpy as np
from math import floor
from copy import deepcopy

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

if __name__ == "__main__":
  parties = [[-1, -0.5, 0], [0, 0.5, 1]]
  voters  = {-1: 1, -0.5: 1, 0: 1, 0.5: 1, 1: 1}
  seats   = 3

  results = elect(parties, voters, seats)

  print("\nResults (number of seats per party):")
  for i, r in enumerate(results):
    print(f"  Party {i}: {r} seats")

