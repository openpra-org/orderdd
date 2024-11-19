import unittest
from orderdd.bdd_ops import BDDDB


class TestBddOps(unittest.TestCase):
    def test_no_redundant_nodes(self):
        expr = "(x1 & x2) | (x1 & ~x2)"
        var_order = ["x1", "x2"]
        bdd_db = BDDDB(force_build=True)
        size = bdd_db.get_size(expr, var_order, reorder=False)
        expected_size = 2  # One for x1 and one terminal node
        self.assertEqual(size, expected_size, "ROBDD should not have redundant nodes")

    def test_var_order(self):
        expr = "(x3 & x1 | ~x4) & x2"
        var_order = ["x1", "x2", "x3", "x4"]
        bdd_db = BDDDB(force_build=True)
        size_without_reorder = bdd_db.get_size(expr, var_order, reorder=False)
        size_with_reorder = bdd_db.get_size(expr, var_order, reorder=True)

        print(f"Size without reordering: {size_without_reorder}")
        print(f"Size with reordering: {size_with_reorder}")

        self.assertEqual(True, False)  # add assertion here

    def test_ROBDD_properties(self):
        expressions = [
            "(x1 & x2) | (x1 & x2)",
            "x1 & ~x1",
            "(x1 & (x2 | x3)) | (~x1 & (x2 | x3))"
        ]
        var_order = ["x1", "x2", "x3"]

        bdd_db = BDDDB(force_build=True)

        for expr in expressions:
            size_without_reorder = bdd_db.get_size(expr, var_order, reorder=False)
            size_with_reorder = bdd_db.get_size(expr, var_order, reorder=True)

            print(f"Expression: {expr}")
            print(f"Size without reordering (BDD): {size_without_reorder}")
            print(f"Size with reordering (ROBDD): {size_with_reorder}")

            # Check if ROBDD size is less than or equal to BDD size
            assert size_with_reorder <= size_without_reorder, "ROBDD size should be <= BDD size"

    def test_ROBDD_properties_with_expected_sizes(self):
        bdd_db = BDDDB(force_build=True)
        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4)", ["x4", "x3", "x2", "x1"]), 5)
        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4)", ["x4", "x1", "x3", "x2"]), 7)
        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4)", ["x1", "x4", "x2", "x3"]), 7)

        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4) | (x5 & x6)", ["x1", "x2", "x3", "x4", "x5", "x6"]), 7)
        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4) | (x5 & x6)", ["x1", "x3", "x2", "x5", "x4", "x6"]), 11)

        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4) | (x5 & x6) | (x7 & x8)", ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]), 9)
        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4) | (x5 & x6) | (x7 & x8)", ["x1", "x8", "x2", "x7", "x3", "x6", "x4", "x5"]), 13)
        self.assertEqual(bdd_db.get_size("(x1 & x2) | (x3 & x4) | (x5 & x6) | (x7 & x8)", ["x1", "x3", "x5", "x7", "x2", "x4", "x6", "x8"]), 31)

    def test_ROBDD_properties_with_expected_sizes_including_negations(self):
        bdd_db = BDDDB(force_build=True)
        self.assertEqual(bdd_db.get_size("(x1 | (x3 & ~x5)) & (x2 | (x4 & ~x6))", ["x1", "x2", "x3", "x4", "x5", "x6"]), 11)
        self.assertEqual(bdd_db.get_size("(x1 | (x3 & ~x5)) & (x2 | (x4 & ~x6))", ["x3", "x5", "x1", "x2", "x4", "x6"]), 7)

    def test_var_order_repeated(self):
        expr = "(x3 & x1&x1 | ~x4) & x2"
        var_order = ["x1", "x2", "x3", "x4"]
        bdd_db = BDDDB()
        size_without_reorder = bdd_db.get_size(expr, var_order, reorder=False)
        size_with_reorder = bdd_db.get_size(expr, var_order, reorder=True)

        print(f"Size without reordering: {size_without_reorder}")
        print(f"Size with reordering: {size_with_reorder}")

        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
