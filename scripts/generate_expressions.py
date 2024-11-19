from itertools import product, permutations
import random
import csv
from typing import List, Tuple, Set

OPERANDS: List[str] = ['a', 'b', 'c', 'd', 'e']
"""List[str]: The default operands to use in boolean expressions."""

OPERATORS: List[str] = ['|', '&']
"""List[str]: The default operators to use in boolean expressions."""

def generate_boolean_expressions(operands: List[str], operators: List[str], N: int = 4) -> List[str]:
    """
    Generate all possible boolean expressions of length N using the given operands and operators.

    This function generates all possible combinations of operands and operators to form boolean expressions
    of a specified length N. Operands and operators can be customized or default to predefined lists.

    Parameters:
        operands (List[str]): A list of operand strings to use in expressions. Defaults to OPERANDS.
        operators (List[str]): A list of operator strings to use in expressions. Defaults to OPERATORS.
        N (int): The number of operands in each expression. Must be between 1 and len(operands).

    Returns:
        List[str]: A list of all possible boolean expressions as strings.

    Raises:
        ValueError: If N is less than 1 or greater than the number of operands.
    """
    if N < 1 or N > len(operands):
        raise ValueError("N must be between 1 and len(operands)")
    operands = operands[:N]
    # Generate all possible combinations of operands of length N with repetition
    operand_combinations = product(operands, repeat=N)
    # Generate all possible expressions
    expressions: List[str] = []
    for operand_combination in operand_combinations:
        # Generate all possible combinations of operators of length N-1
        operator_combinations = product(operators, repeat=N-1)
        for operator_combination in operator_combinations:
            # Construct the expression by interleaving operands with operators
            expression = ''.join(
                f"{operand}{operator}"
                for operand, operator in zip(operand_combination, operator_combination)
            ) + operand_combination[-1]  # Add the last operand without an operator
            expressions.append(expression)
    return expressions

def generate_boolean_expressions_upto(operands: List[str] = OPERANDS, operators: List[str] = OPERATORS, N: int = 4) -> List[str]:
    """
    Generate all possible boolean expressions of lengths from 1 up to N using given operands and operators.

    This function generates all unique boolean expressions by combining operands and operators for lengths
    ranging from 1 to N. It starts by including single operands and then builds expressions of increasing length.

    Parameters:
        operands (List[str]): A list of operand strings to use in expressions. Defaults to OPERANDS.
        operators (List[str]): A list of operator strings to use in expressions. Defaults to OPERATORS.
        N (int): Maximum number of operands in expressions. Must be between 1 and len(operands).

    Returns:
        List[str]: A list of all unique boolean expressions as strings.
    """
    all_unique_expressions: List[str] = []
    # Explicitly add single operands as unique expressions
    all_unique_expressions.extend(operands)
    for n in range(2, N + 1):
        # Start from 2 since single operands are already added
        expressions = generate_boolean_expressions(operands, operators, n)
        all_unique_expressions.extend(expressions)
    return all_unique_expressions

def generate_random_boolean_expressions(operands: List[str] = OPERANDS, operators: List[str] = OPERATORS, N: int = 4, num_expressions: int = 10) -> List[str]:
    """
    Generate a specified number of random boolean expressions of length N.

    This function creates random boolean expressions by randomly selecting operands and operators
    and interleaving them to form expressions of length N.

    Parameters:
        operands (List[str]): A list of operand strings to use in expressions. Defaults to OPERANDS.
        operators (List[str]): A list of operator strings to use in expressions. Defaults to OPERATORS.
        N (int): The number of operands in each expression. Must be at least 1.
        num_expressions (int): The number of random expressions to generate.

    Returns:
        List[str]: A list of randomly generated boolean expressions as strings.

    Raises:
        ValueError: If N is less than 1 or greater than the number of operands.
    """
    if N < 1:
        raise ValueError("N must be at least 1")
    if N > len(operands):
        raise ValueError("N must be at at most N")
    operands = operands[:N]
    expressions: List[str] = []
    # Handle the case when N is 1 (single operand, no operators)
    if N == 1:
        return operands.copy()
    for _ in range(num_expressions):
        # Randomly choose operands and operators
        chosen_operands = [random.choice(operands) for _ in range(N)]
        chosen_operators = [random.choice(operators) for _ in range(N - 1)]
        # Construct the expression by interleaving operands with operators
        expression = ''.join(
            f"{operand}{operator}"
            for operand, operator in zip(chosen_operands, chosen_operators)
        ) + chosen_operands[-1]  # Add the last operand without an operator
        expressions.append(expression)
    return expressions

def generate_random_boolean_expressions_upto(operands: List[str] = OPERANDS, operators: List[str] = OPERATORS, N: int = 4, num_expressions_per_length: int = 10) -> List[str]:
    """
    Generate random boolean expressions of lengths from 1 up to N.

    This function generates random boolean expressions for each length from 1 to N.
    For each length, a specified number of random expressions are generated.

    Parameters:
        operands (List[str]): A list of operand strings to use in expressions. Defaults to OPERANDS.
        operators (List[str]): A list of operator strings to use in expressions. Defaults to OPERATORS.
        N (int): Maximum number of operands in expressions. Must be at least 1.
        num_expressions_per_length (int): Number of random expressions to generate per length.

    Returns:
        List[str]: A list of randomly generated boolean expressions as strings.
    """
    all_random_expressions: List[str] = []
    for n in range(1, N + 1):
        expressions = generate_random_boolean_expressions(operands, operators, n, num_expressions_per_length)
        all_random_expressions.extend(expressions)
    return all_random_expressions

def find_unique_operands(expression: str) -> List[str]:
    """
    Find all unique operands in a boolean expression.

    This function extracts all unique operands from a boolean expression.
    Operands are assumed to be single lowercase letters.

    Parameters:
        expression (str): The boolean expression as a string.

    Returns:
        List[str]: A sorted list of unique operands found in the expression.
    """
    operands: Set[str] = set()
    for char in expression:
        if char.isalpha() and char.islower():
            operands.add(char)
    # Convert the set to a list to have a consistent order
    available_operands: List[str] = list(operands)
    return available_operands

def generate_expressions() -> Tuple[List[str], List[Tuple[str, ...]]]:
    """
    Generate boolean expressions and corresponding operand orderings.

    This function generates all unique boolean expressions up to a certain length
    and for each expression, it determines the ordering of operands.

    It returns two lists:
    - A list of expressions.
    - A list of corresponding operand orderings (as tuples).

    Returns:
        Tuple[List[str], List[Tuple[str, ...]]]: A tuple containing:
            - X_expressions: List of boolean expressions as strings.
            - Y_expressions: List of tuples, each containing an ordering of operands.
    """
    num_classes: int = len(OPERANDS) + 1
    unique_expressions: List[str] = list(set(generate_boolean_expressions_upto(N=num_classes - 1)))
    print('unique expressions:', len(unique_expressions))
    X_expressions: List[str] = []
    Y_expressions: List[Tuple[str, ...]] = []
    # For each expression in unique_expressions
    for expression in unique_expressions:
        available_operands = find_unique_operands(expression)
        if len(available_operands) == 1:
            # If there is only one operand, add the expression and the operand as a tuple
            X_expressions.append(expression)
            Y_expressions.append((available_operands[0],))
        else:
            # Get all unique permutations of the variable orderings
            variable_orderings = list(permutations(available_operands))
            # For each variable ordering, add the expression to X_expressions
            # and the corresponding ordering to Y_expressions
            for ordering in variable_orderings:
                X_expressions.append(expression)  # Add the expression once for each ordering
                Y_expressions.append(ordering)
    return X_expressions, Y_expressions

from typing import List
from multiprocessing import Pool

def var_count_for_ordering(
        expression: str = 'a & (b & (c | d))',
        variable_ordering: List[str] = ['d', 'a', 'b', 'c'],
        reordering: bool = False
) -> Tuple[int, int]:
    """
    Compute the size of the Binary Decision Diagram (BDD) for a given boolean expression.

    This function builds a BDD for the provided boolean expression using a specified
    variable ordering. Optionally, it can perform dynamic variable reordering to minimize
    the BDD size. The function returns the number of nodes in the resultant BDD.

    Parameters:
        expression (str): The boolean expression to be converted into a BDD.
        variable_ordering (List[str]): The list of variable names specifying the variable ordering.
        reordering (bool): If True, enables dynamic variable reordering to minimize BDD size.

    Returns:
        int: The number of nodes in the BDD representing the given expression.

    Example:
        >>> size = var_count_for_ordering('a & (b & (c | d))', ['d', 'a', 'b', 'c'])
        >>> print(size)
        7

    Raises:
        ValueError: If variables in the expression are not declared in `variable_ordering`.
    """

    if len(variable_ordering) == 1:
        return 1, 1

    # Map variables to their levels based on the given variable_ordering
    levels = {var: idx for idx, var in enumerate(variable_ordering)}
    import dd.autoref as _bdd
    # Initialize the BDD manager with specified variable levels
    bdd = _bdd.BDD(levels)
    bdd.configure(reordering=reordering)

    # Declare variables with the specified ordering
    bdd.declare(*variable_ordering)
    size = -1
    reordered_size = -1
    # Add the boolean expression to the BDD
    try:
        u = bdd.add_expr(expression)
        size = len(bdd)
        bdd.reorder()
        reordered_size = len(bdd)
    except KeyError as e:
        raise ValueError(f"Variable {e} in expression not declared in variable_ordering.") from e
    except:
        print(f"bdd error for : {expression}, for ordering: {variable_ordering}, size: {size}")


    # Return the size of the BDD (number of nodes)
    return size, reordered_size


from tqdm import tqdm
import csv

def process_expression(args):
    """Wrapper function to process a single expression and ordering."""
    expression, ordering = args
    bdd_size, bdd_size_reordered = var_count_for_ordering(expression, ordering, reordering=True)
    ordering_str = ':'.join(ordering)
    return expression, ordering_str, bdd_size, bdd_size_reordered


if __name__ == "__main__":
    X_expressions, Y_expressions = generate_expressions()
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
        csv_writer.writerow(["expression", "ordering", "bdd_size", "bdd_size_reordered"])

        for expression, ordering_str, bdd_size, bdd_size_reordered in results:
            # Append the variable count to the list
            X_bdd_size.append(bdd_size)
            X_bdd_size_reordered.append(bdd_size_reordered)

            # Write the row to the CSV file
            csv_writer.writerow([expression, ordering_str, bdd_size, bdd_size_reordered])

            # Print the row
            print(f"{expression},{ordering_str},{bdd_size},{bdd_size_reordered}")