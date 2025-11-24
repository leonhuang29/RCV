from nash import find_nash_equilibria, best_response_right, brute_force_nash
from mixed_nash import MixedNashLP

def main():
    
    # left_possibilities = [-1.0, -0.5, 0.0]
    # right_possibilities = [0.0, 0.5, 1.0]
    # voter_positions = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # seats = 3
    # k = 3
    # density1 = [5, 2, 5, 2, 5]
    # output_dir = "results/52525"

    # #find_nash_equilibria(left_possibilities, right_possibilities, k, voter_positions, seats)
    # #best_response_right(left_possibilities, right_possibilities, k, voter_positions, seats)
    # brute_force_nash(left_possibilities, right_possibilities, k, voter_positions, seats, density1, output_dir, csv_name="nash_for_zero_one_restriction.csv")

    # left_possibilities = [-1.0, -0.5, 0.0, 0.5]
    # right_possibilities = [-0.5, 0.0, 0.5, 1.0]

    # #find_nash_equilibria(left_possibilities, right_possibilities, k, voter_positions, seats)
    # #best_response_right(left_possibilities, right_possibilities, k, voter_positions, seats)
    # brute_force_nash(left_possibilities, right_possibilities, k, voter_positions, seats, density1, output_dir, csv_name="nash_for_half_one_restriction.csv")

    # left_possibilities = [-1.0, -0.5, 0.0, 0.5, 1.0]
    # right_possibilities = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # #find_nash_equilibria(left_possibilities, right_possibilities, k, voter_positions, seats)
    # #best_response_right(left_possibilities, right_possibilities, k, voter_positions, seats)
    # brute_force_nash(left_possibilities, right_possibilities, k, voter_positions, seats, density1, output_dir, csv_name="nash_for_no_restriction.csv")
    
    dir = "results/52525/nash_for_no_restriction.csv"
    MixedNashLP_instance = MixedNashLP(csv_path=dir, k=3)
    MixedNashLP_instance.solve()
    MixedNashLP_instance.pretty_print_solution()
if __name__ == "__main__":
    main()