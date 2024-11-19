from itertools import product, permutations
from typing import List, Tuple
from multiprocessing import Pool
from itertools import chain
import re


def generate_boolean_expressions(operands: List[str], operators: List[str], num_operands=None) -> List[str]:
    if num_operands is None:
        num_operands = len(operands)
    if num_operands < 1 or num_operands > len(operands):
        raise ValueError("N must be between 1 and len(operands)")
    # Generate all possible combinations of operands of length N with repetition
    operand_combinations = product(operands, repeat=num_operands)
    # Generate all possible expressions
    expressions: List[str] = []
    for operand_combination in operand_combinations:
        # Generate all possible combinations of operators of length N-1
        operator_combinations = product(operators, repeat=num_operands-1)
        for operator_combination in operator_combinations:
            # Construct the expression by interleaving operands with operators
            expression = ''.join(
                f"{operand}{operator}" for operand, operator in zip(operand_combination, operator_combination)
            ) + operand_combination[-1]  # Add the last operand without an operator
            expressions.append(expression)
    return expressions

def generate_boolean_expressions_upto(operands: List[str], operators: List[str], num_operands=None) -> List[str]:
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
    for n in range(2, num_operands + 1):
        # Start from 2 since single operands are already added
        expressions = generate_boolean_expressions(operands, operators, n)
        all_unique_expressions.extend(expressions)
    return all_unique_expressions


def find_unique_operands(expression: str, exclusion_list: List[str], operators: List[str]) -> List[str]:
    """
    Find all unique operands in a boolean expression.

    This function extracts all unique operands from a boolean expression.
    Operands can be multi-character strings and are assumed not to be any of the specified
    operators or exclusion characters.

    Parameters:
        expression (str): The boolean expression as a string.
        exclusion_list (List[str]): A list of characters to exclude (e.g., ['(', ')']).
        operators (List[str]): A list of operator strings to exclude.

    Returns:
        List[str]: A sorted list of unique operands found in the expression.
    """
    # Combine operators and exclusion characters
    tokens_to_replace = operators + exclusion_list
    # Escape tokens for use in regular expressions
    escaped_tokens = [re.escape(token) for token in tokens_to_replace]
    # Create a pattern that matches any operator or exclusion character
    pattern = '|'.join(escaped_tokens)
    # Replace all operators and exclusions in the expression with a space
    expression_no_ops = re.sub(pattern, ' ', expression)
    # Remove extra whitespace
    expression_no_ops = re.sub(r'\s+', ' ', expression_no_ops.strip())
    # Split the expression by spaces to get operands
    operands = expression_no_ops.split(' ')
    # Remove duplicates and sort
    unique_operands = sorted(set(operands))
    return unique_operands

def process_single_expression(expression: str, operators: List[str]):
    available_operands = find_unique_operands(expression=expression, exclusion_list=['(', ')'], operators=operators)
    results = []
    if len(available_operands) == 1:
        results.append((expression, (available_operands[0],)))
    else:
        variable_orderings = list(permutations(available_operands))
        for ordering in variable_orderings:
            results.append((expression, ordering))
    return results

def generate_expressions(operands: List[str], operators: List[str], num_operands: int) -> Tuple[List[str], List[Tuple[str, ...]]]:

    if num_operands < 1 or num_operands > len(operands):
        raise ValueError("N must be between 1 and len(operands)")
    if num_operands is None:
        num_operands = len(operands)

    unique_expressions: List[str] = generate_boolean_expressions_upto(operands, operators, num_operands)
    print('Number of unique expressions:', len(unique_expressions))

    # Prepare the arguments for pool.starmap
    pool_args = [(expression, operators) for expression in unique_expressions]

    # Use multiprocessing Pool to process expressions in parallel
    with Pool() as pool:
        # Map the processing function to each expression and operators
        results = pool.starmap(process_single_expression, pool_args)

    # Flatten the list of results
    all_results = list(chain.from_iterable(results))

    # Unzip the results into X_expressions and Y_expressions
    x_expressions, y_expressions = zip(*all_results)

    return list(x_expressions), list(y_expressions)
