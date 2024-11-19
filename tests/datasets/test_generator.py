import unittest
import numpy as np

from orderdd.datasets import generator


class TestDatasetGenerator(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_generate_boolean_expressions(self):
        xpr = generator.generate_boolean_expressions(operands={'x1', 'x2', 'x3'}, operators={'•', '|'})
        print(xpr)


if __name__ == '__main__':
    unittest.main()
