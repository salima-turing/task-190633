import unittest
import random
from typing import List
import numpy as np

# Function to calculate the Pearson correlation coefficient as a data similarity metric
def pearson_correlation(x: List[float], y: List[float]) -> float:
	if len(x) != len(y):
		raise ValueError("Lists must have the same length.")

	n = len(x)
	sum_x = sum(x)
	sum_y = sum(y)
	sum_xx = sum(xi**2 for xi in x)
	sum_yy = sum(yi**2 for yi in y)
	sum_xy = sum(xi*yi for xi, yi in zip(x, y))

	denominator = np.sqrt((n*sum_xx - sum_x**2) * (n*sum_yy - sum_y**2))
	if denominator == 0:
		return 0.0

	return (n*sum_xy - sum_x*sum_y) / denominator

class TestPearsonCorrelation(unittest.TestCase):

	def test_pearson_correlation_with_property_based_testing(self):
		"""
		Use property-based testing to generate random inputs and verify that the Pearson correlation
		metric satisfies properties like symmetry, transitivity, and non-negativity.
		"""
		for _ in range(1000):  # Run the test for 1000 iterations
			# Generate random lists of length between 2 and 10
			length = random.randint(2, 10)
			list_x = [random.uniform(-100, 100) for _ in range(length)]
			list_y = [random.uniform(-100, 100) for _ in range(length)]
			list_z = [random.uniform(-100, 100) for _ in range(length)]

			corr_xy = pearson_correlation(list_x, list_y)
			corr_yx = pearson_correlation(list_y, list_x)
			corr_xz = pearson_correlation(list_x, list_z)
			corr_zx = pearson_correlation(list_z, list_x)
			corr_yz = pearson_correlation(list_y, list_z)
			corr_zy = pearson_correlation(list_z, list_y)

			# Test symmetry
			self.assertAlmostEqual(corr_xy, corr_yx, places=6, msg="Pearson correlation should be symmetric")

			# Test transitivity
			if corr_xy != 0 and corr_yz != 0:
				self.assertAlmostEqual(corr_xz, corr_zy, places=6, msg="Pearson correlation should be transitive")

			# Test non-negativity
			self.assertGreaterEqual(corr_xy, -1.0, msg="Pearson correlation should be non-negative")
			self.assertLessEqual(corr_xy, 1.0, msg="Pearson correlation should be non-negative")

if __name__ == '__main__':
	unittest.main()
