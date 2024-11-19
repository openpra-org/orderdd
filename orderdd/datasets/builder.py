from tqdm import tqdm
from multiprocessing import Pool
import csv
from typing import List

from bdd_sizer import var_count_for_ordering
from generator import generate_expressions

def process_expression(process_args):
    """Wrapper function to process a single expression and ordering."""
    x_expression, orderings = process_args
    initial_size, reordered_size, reordering = var_count_for_ordering(x_expression, orderings, reordering=True)
    ordering_colon = ':'.join(orderings)
    reordering_colon = ':'.join(reordering)
    return x_expression, ordering_colon, initial_size, reordering_colon, reordered_size


if __name__ == "__main__":
    my_operands: List[str] = list({'a', 'b', 'c', 'd', 'e', 'f', 'g'})
    my_operators: List[str] = list({'|', '&'})
    X_expressions, Y_expressions = generate_expressions(operands=my_operands, operators=my_operators,
                                                        num_operands=len(my_operands))
    print(X_expressions[0:10])
    print(Y_expressions[0:10])

    # Create a list to store the variable counts
    X_bdd_size = []
    X_bdd_size_reordered = []

    # Prepare the arguments for the pool
    args = zip(X_expressions, Y_expressions)

    # Use a multiprocessing pool to process the expressions in parallel
    with Pool() as pool:
        # Use tqdm to display a progress bar
        results = list(tqdm(pool.imap(process_expression, args), total=len(X_expressions), desc="Processing Expressions"))

    # Write the results to a CSV file
    with open('inputs.csv', 'w', newline='') as input_csv:
        csv_writer = csv.writer(input_csv)
        csv_writer.writerow(["expression", "ordering", "bdd_size", "reordering", "bdd_size_reordered"])

        for expression, ordering_str, bdd_size, reordering_str, bdd_size_reordered in results:
            # Append the variable count to the list
            X_bdd_size.append(bdd_size)
            X_bdd_size_reordered.append(bdd_size_reordered)

            # Write the row to the CSV file
            csv_writer.writerow([expression, ordering_str, bdd_size, reordering_str, bdd_size_reordered])

            # Print the row
            #print(f"{expression},{ordering_str},{bdd_size},{reordering_str},{bdd_size_reordered}")