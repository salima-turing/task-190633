import unittest
import numpy as np
from similarity_metrics import cosine_similarity

class TestCosineSimilarity(unittest.TestCase):

    def test_cosine_similarity_accuracy(self):
        """
        Test the cosine similarity metric using an exhaustive data-driven approach.
        We will generate random vectors and verify the similarity results against known correct values.
        """
        np.random.seed(42)  # Set seed for reproducibility
        num_tests = 1000
        max_vector_size = 100
        tolerance = 0.0001  # Adjust tolerance as needed

        for _ in range(num_tests):
            vector_size = np.random.randint(1, max_vector_size + 1)
            vector1 = np.random.randn(vector_size)
            vector2 = np.random.randn(vector_size)

            # Calculate the cosine similarity using the implemented function
            cosine_sim_result = cosine_similarity(vector1, vector2)

            # Calculate the cosine similarity using the numpy function
            cosine_sim_numpy = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

            # Check if the absolute difference between the two results is less than the tolerance
            self.assertAlmostEqual(cosine_sim_result, cosine_sim_numpy, delta=tolerance,
                                   msg=f"Cosine similarity failed for vectors of size {vector_size}")

if __name__ == '__main__':
    unittest.main()
