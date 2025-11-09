import itertools
import math
from election import Election
import csv
import os
from datetime import datetime


def find_nash_equilibria(left_possibilities, right_possibilities, k, voter_positions, seats):
    """
    Enumerate all candidate placements and find pure-strategy Nash equilibria.
    """
    left_strats = list(itertools.combinations_with_replacement(left_possibilities, k))
    right_strats = list(itertools.combinations_with_replacement(right_possibilities, k))

    print(f"Enumerating {len(left_strats)} left × {len(right_strats)} right = {len(left_strats) * len(right_strats)} profiles")

    payoff = {}
    for L in left_strats:
        for R in right_strats:
            election = Election(list(L), list(R), voter_positions, seats)
            result = election.run()
            l_val = float(result.get("left", 0))
            r_val = float(result.get("right", 0))
            payoff[(L, R)] = (l_val, r_val)

    # --- Best responses ---
    best_L_for_R = {
        R: _best_responses(lambda L: payoff[(L, R)][0], left_strats)
        for R in right_strats
    }
    best_R_for_L = {
        L: _best_responses(lambda R: payoff[(L, R)][1], right_strats)
        for L in left_strats
    }

    # --- Nash equilibria ---
    nash = []
    for L in left_strats:
        for R in right_strats:
            if (L in best_L_for_R[R][1]) and (R in best_R_for_L[L][1]):
                nash.append((L, R, payoff[(L, R)]))

    # --- Output ---
    print("\n=== Nash candidate portfolios (pure strategies) ===")
    if not nash:
        print("No pure Nash equilibria found.")
    else:
        for (L, R, (lval, rval)) in nash:
            print(f"\nL = {L}    R = {R}")
            print(f"  Expected seats → Left: {lval:.3f}, Right: {rval:.3f}")

    # also return the list for later inspection
    return nash


def _best_responses(value_func, strategy_list):
    """Helper: compute best-response payoff and argmax set."""
    best_val = -math.inf
    best_strats = []
    for s in strategy_list:
        val = value_func(s)
        if val > best_val + 1e-9:
            best_val = val
            best_strats = [s]
        elif abs(val - best_val) <= 1e-9:
            best_strats.append(s)
    return best_val, best_strats

import itertools

def best_response(
    fixed_side: str,
    fixed_strategy: tuple,
    variable_possibilities: list,
    k: int,
    voter_positions,
    seats: int,
):
    """
    Generic best-response solver.

    Args:
        fixed_side: "L" or "R" — the side whose strategy is fixed.
        fixed_strategy: tuple of that side's candidate positions.
        variable_possibilities: list of positions the *other* side can choose from.
        k: number of candidates to run per side.
        voter_positions: list of voter positions on the line.
        seats: number of seats to fill.

    Returns:
        (best_value, best_strategies)
        where best_value = max expected seat count for optimizing side,
        and best_strategies = list of tuples of positions achieving that payoff.
    """
    best_val = float("-inf")
    best_strategies = []

    # Determine which side is optimizing
    optimizing_side = "R" if fixed_side == "L" else "L"

    for strategy in itertools.combinations(variable_possibilities, k):
        # Build election depending on which side is fixed
        if fixed_side == "L":
            election = Election(
                left_positions=list(fixed_strategy),
                right_positions=list(strategy),
                voter_positions=voter_positions,
                seats=seats
            )
        else:
            election = Election(
                left_positions=list(strategy),
                right_positions=list(fixed_strategy),
                voter_positions=voter_positions,
                seats=seats
            )

        result = election.run()
        payoff = result["right"] if optimizing_side == "R" else result["left"]

        if payoff > best_val + 1e-9:
            best_val = payoff
            best_strategies = [strategy]
        elif abs(payoff - best_val) <= 1e-9:
            best_strategies.append(strategy)

    print(f"Best {optimizing_side} payoff: {best_val}")
    print(f"Best {optimizing_side} strategies: {best_strategies}")

    return best_val, best_strategies

def best_response_right(left_possibilities, right_possibilities, k, voter_positions, seats):
    """
    Compute best response for Right given Left's fixed strategy.
    """
    # For demonstration, fix Left's strategy to the first k positions
    fixed_left_strategy = tuple(left_possibilities[:k])
    print(f"Computing best response for Right given Left's fixed strategy: {fixed_left_strategy}")
    return best_response(
        fixed_side="L",
        fixed_strategy=fixed_left_strategy,
        variable_possibilities=right_possibilities,
        k=k,
        voter_positions=voter_positions,
        seats=seats
    )

def best_response_left(left_possibilities, right_possibilities, k, voter_positions, seats): 
    """
    Compute best response for Left given Right's fixed strategy.
    """
    # For demonstration, fix Right's strategy to the first k positions
    fixed_right_strategy = tuple(right_possibilities[:k])
    print(f"Computing best response for Left given Right's fixed strategy: {fixed_right_strategy}")
    return best_response(
        fixed_side="R",
        fixed_strategy=fixed_right_strategy,
        variable_possibilities=left_possibilities,
        k=k,
        voter_positions=voter_positions,
        seats=seats
    )

def brute_force_nash(left_possibilities, right_possibilities, k, voter_positions, seats, voter_densities = None,
                     output_dir = "results", csv_name = None):
    """
    Enumerate all portfolios (combinations_with_replacement),
    compute payoffs using Election, print every profile, mark Nash profiles,
    and export all results to a CSV file in `output_dir`.

    Args:
        output_dir: Folder where CSV will be saved (default: "results").
        csv_name: Optional filename. If None, timestamped name is generated.
    """
    # ensure folder exists
    os.makedirs(output_dir, exist_ok=True)

    # build filename
    if csv_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_name = f"nash_results_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_name)

    left_strats = list(itertools.combinations_with_replacement(left_possibilities, k))
    right_strats = list(itertools.combinations_with_replacement(right_possibilities, k))

    print(f"Enumerating {len(left_strats)} left × {len(right_strats)} right = {len(left_strats)*len(right_strats)} profiles\n")

    payoff = {}
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

    # precompute best responses
    best_L_for_R = {}
    for R in right_strats:
        vals = [(L, payoff[(L, R)][0]) for L in left_strats]
        max_val = max(v for _, v in vals)
        bests = [L for L, v in vals if abs(v - max_val) <= 1e-9]
        best_L_for_R[R] = (max_val, bests)

    best_R_for_L = {}
    for L in left_strats:
        vals = [(R, payoff[(L, R)][1]) for R in right_strats]
        max_val = max(v for _, v in vals)
        bests = [R for R, v in vals if abs(v - max_val) <= 1e-9]
        best_R_for_L[L] = (max_val, bests)

    nash_list = []
    csv_rows = []

    # print every profile and collect for CSV
    for L in left_strats:
        for R in right_strats:
            lval, rval = payoff[(L, R)]
            is_left_best = L in best_L_for_R[R][1]
            is_right_best = R in best_R_for_L[L][1]
            is_nash = is_left_best and is_right_best
            #tag = "  <-- NASH" if is_nash else ""
            #print(f"L={L}\tR={R}\t-> (L={lval:.6f}, R={rval:.6f}){tag}")
            if is_nash:
                nash_list.append((L, R, (lval, rval)))

            csv_rows.append({
                "left_strategy": str(L),
                "right_strategy": str(R),
                "left_payoff": lval,
                "right_payoff": rval,
                "is_left_best": int(is_left_best),
                "is_right_best": int(is_right_best),
                "is_nash": int(is_nash),
            })

    # write to CSV inside output_dir
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "left_strategy",
            "right_strategy",
            "left_payoff",
            "right_payoff",
            "is_left_best",
            "is_right_best",
            "is_nash",
        ])
        writer.writeheader()
        writer.writerows(csv_rows)

    # summary
    print("\n=== Summary ===")
    print(f"Total profiles: {len(left_strats)*len(right_strats)}")
    print(f"Total Nash profiles found: {len(nash_list)}")
    print(f"Results written to: {csv_path}")
    if nash_list:
        print("List of Nash profiles (L, R, (left_val, right_val)):")
        for item in nash_list:
            print(item)

    return nash_list


