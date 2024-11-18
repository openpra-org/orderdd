from itertools import product, permutations
import random

# Define the operands and operators
OPERANDS = ['a', 'b', 'c', 'd', 'e']
OPERATORS = ['|', '&']


def generate_boolean_expressions(operands=OPERANDS, operators=OPERATORS, N=4):
    if N < 1 or N > len(operands):
        raise ValueError("N must be between 1 and len(operands)")

    operands = operands[:N]

    # Generate all possible combinations of operands of length N with repetition
    operand_combinations = product(operands, repeat=N)

    # Generate all possible expressions
    expressions = []
    for operand_combination in operand_combinations:
        # Generate all possible combinations of operators of length N-1
        operator_combinations = product(operators, repeat=N-1)
        for operator_combination in operator_combinations:
            # Construct the expression by interleaving operands with operators
            expression = ''.join(
                f"{operand}{operator}" for operand, operator in zip(operand_combination, operator_combination)
            ) + operand_combination[-1]  # Add the last operand without an operator
            expressions.append(expression)

    return expressions


def generate_boolean_expressions_upto(operands=OPERANDS, operators=OPERATORS, N=4):
    all_unique_expressions = []
    # Explicitly add single operands as unique expressions
    all_unique_expressions.extend(operands)
    for n in range(2, N+1):  # Start from 2 since single operands are already added
        expressions = generate_boolean_expressions(operands, operators, n)
        all_unique_expressions.extend(expressions)
    return all_unique_expressions


def generate_random_boolean_expressions(operands=OPERANDS, operators=OPERATORS, N=4, num_expressions=10):
    if N < 1:
      raise ValueError("N must be at least 1")

    # Handle the case when N is 1 (single operand, no operators)
    if N == 1:
      return operands

    if N > len(operands):
      raise ValueError("N must be at at most N")

    operands = operands[:N]
    expressions = []

    for _ in range(num_expressions):
        # Randomly choose operands and operators
        chosen_operands = [random.choice(operands) for _ in range(N)]
        chosen_operators = [random.choice(operators) for _ in range(N - 1)]

        # Construct the expression by interleaving operands with operators
        expression = ''.join(
            f"{operand}{operator}" for operand, operator in zip(chosen_operands, chosen_operators)
        ) + chosen_operands[-1]  # Add the last operand without an operator
        expressions.append(expression)

    return expressions

def generate_random_boolean_expressions_upto(operands=OPERANDS, operators=OPERATORS, N=4, num_expressions_per_length=10):
    all_random_expressions = []
    for n in range(1, N+1):
        expressions = generate_random_boolean_expressions(operands, operators, n, num_expressions_per_length)
        all_random_expressions.extend(expressions)
    return all_random_expressions


def find_unique_operands(expression):
    # Assuming operands are single lowercase letters
    operands = set()
    for char in expression:
        if char.isalpha() and char.islower():
            operands.add(char)
    # Convert the set to a sorted list to have a consistent order
    available_operands = list(operands)
    return available_operands


def generate_expressions():
    num_classes = len(OPERANDS) + 1
    unique_expressions = list(set(generate_boolean_expressions_upto(N=num_classes - 1)))
    print('unique expressions:', len(unique_expressions))

    # Assuming unique_expressions is already defined and contains all unique expressions
    X_expressions = []
    Y_expressions = []

    # For each expression in unique_expressions
    for expression in unique_expressions:
        available_operands = find_unique_operands(expression)
        #print(expression, available_operands)
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

if __name__ == "__main__":
    X_expressions, Y_expressions = generate_expressions()
    print(X_expressions[0:10])
    print(Y_expressions[0:10])