import numpy as np
import itertools
from election import Election

# TODO: i just ripped this out of nash.py; clean up later
def compute_payoff_table(left_possibilities, right_possibilities, k, voter_positions, seats, voter_densities=None):
    left_strats = list(itertools.combinations_with_replacement(left_possibilities, k))
    right_strats = list(itertools.combinations_with_replacement(right_possibilities, k))

    payoff = {}  # (L, R) -> (left_val, right_val)
    for L in left_strats:
        for R in right_strats:
            election = Election(list(L), list(R), voter_positions, seats, voter_densities)
            res = election.run()
            if isinstance(res, dict):
                left_val = float(res.get("left", res.get("left_seats", 0)))
                right_val = float(res.get("right", res.get("right_seats", seats - left_val)))
            else:
                left_val, right_val = float(res[0]), float(res[1])
            payoff[(L, R)] = (left_val, right_val)

    return left_strats, right_strats, payoff


def hedge_dynamics(left_strats, right_strats, payoff, seats, T=5000, eta=0.3, burn_in_frac=0.5):
    """
    Run multiplicative-weights (Hedge) for both players in this constant-sum game.

    left_strats, right_strats: lists of pure strategies (tuples)
    payoff: dict[(L, R)] -> (left_val, right_val)
    seats: total seats (used for normalization)
    T: number of iterations
    eta: learning rate
    burn_in_frac: fraction of iterations to discard when averaging

    Returns:
        (pL_avg, pR_avg)  # approximate mixed strategies
    """
    nL = len(left_strats)
    nR = len(right_strats)

    # initial weights = uniform
    wL = np.ones(nL)
    wR = np.ones(nR)

    hist_pL = []
    hist_pR = []

    for t in range(T):
        pL = wL / wL.sum()
        pR = wR / wR.sum()
        hist_pL.append(pL)
        hist_pR.append(pR)

        # expected payoff for each Left pure strategy against Right's mixed strategy
        uL_rows = np.zeros(nL)
        for i, L in enumerate(left_strats):
            for j, R in enumerate(right_strats):
                uL_rows[i] += pR[j] * payoff[(L, R)][0]

        # expected payoff for each Right pure strategy against Left's mixed strategy
        uR_cols = np.zeros(nR)
        for j, R in enumerate(right_strats):
            for i, L in enumerate(left_strats):
                uR_cols[j] += pL[i] * payoff[(L, R)][1]

        # Normalize to [0,1]
        uL_rows /= seats
        uR_cols /= seats

        # hedge update with numerical stability
        # (subtract max to prevent overflow)
        log_update_L = eta * uL_rows
        log_update_R = eta * uR_cols
        log_update_L -= log_update_L.max()
        log_update_R -= log_update_R.max()

        wL *= np.exp(log_update_L)
        wR *= np.exp(log_update_R)

        wL = wL / wL.sum()
        wR = wR / wR.sum()

    # time-average the strategies after "burn-in"
    burn = int(burn_in_frac * T)
    pL_avg = np.mean(hist_pL[burn:], axis=0)
    pR_avg = np.mean(hist_pR[burn:], axis=0)

    return pL_avg, pR_avg


def run_hedge_experiment():
    left_possibilities = [-1.0, -0.5, 0.0, 0.5, 1.0]
    right_possibilities = [-1.0, -0.5, 0.0, 0.5, 1.0]
    voter_positions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    seats = 3
    k = 3
    density1 = [1, 1, 1, 1, 1]

    left_strats, right_strats, payoff = compute_payoff_table(
        left_possibilities,
        right_possibilities,
        k,
        voter_positions,
        seats,
        voter_densities=density1,
    )

    pL, pR = hedge_dynamics(left_strats, right_strats, payoff, seats, T=5000, eta=0.3)

    idxL_sorted = np.argsort(-pL)
    idxR_sorted = np.argsort(-pR)

    print("Approx Left mixed strategy (top mass):")
    for idx in idxL_sorted[:10]:
        print(left_strats[idx], pL[idx])

    print("\nApprox Right mixed strategy (top mass):")
    for idx in idxR_sorted[:10]:
        print(right_strats[idx], pR[idx])

