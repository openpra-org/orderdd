from itertools import product


def generate_boolean_expressions(operands: set[str], operators: set[str], num_operands=None):
    if num_operands is None:
        num_operands = len(operands)

    print(num_operands)
    # Generate all possible combinations of operands of len(operands) with repetition
    operand_combinations = product(operands, repeat=num_operands)
    operator_combinations = product(operators, repeat=num_operands - 1)

    # Generate all possible expressions
    expressions = []
    for operand_combination in operand_combinations:
        operator_combinations = product(operators, repeat=num_operands - 1)
        for operator_combination in operator_combinations:
            # Construct the expression by interleaving operands with operators
            expression = ''.join(
                f"{operand}{operator}" for operand, operator in zip(operand_combination, operator_combination)
            ) + operand_combination[-1]  # Add the last operand without an operator
            expressions.append(expression)

    return expressions
