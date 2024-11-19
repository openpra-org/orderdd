from typing import Tuple, List


def var_count_for_ordering(
        expression: str,              # 'a & (b & (c | d))'
        variable_ordering: List[str], # ['d', 'a', 'b', 'c']
        reordering: bool = False
) -> Tuple[int, int, List[str]]:
    if len(variable_ordering) == 1:
        return 2, 2, variable_ordering

    # Map variables to their levels based on the given variable_ordering
    levels = {var: idx for idx, var in enumerate(variable_ordering)}
    import dd.autoref as _bdd
    # Initialize the BDD manager with specified variable levels
    bdd = _bdd.BDD(levels)
    bdd.configure(reordering=reordering)

    # Declare variables with the specified ordering
    bdd.declare(*variable_ordering)
    size = float('nan')
    reordered_size = float('nan')
    new_order = []
    # Add the boolean expression to the BDD
    try:
        u = bdd.add_expr(expression)
        size = len(bdd)
        bdd.reorder()
        reordered_size = len(bdd)
        # Get the new, reordered variable list
        # Retrieve the mapping of variable names to their levels after reordering
        var_levels = bdd.vars  # dict: variable names -> levels
        # Sort variable names based on their levels to get the new order
        new_order = sorted(var_levels, key=lambda var: var_levels[var])
    except KeyError as e:
        raise ValueError(f"Variable {e} in expression not declared in variable_ordering.") from e
    except:
        print(f"bdd error for : {expression}, for ordering: {variable_ordering}, size: {size}")

    # Return the size of the BDD (number of nodes)
    return size, reordered_size, new_order
