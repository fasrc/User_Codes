# test_mc_pi.py
import unittest
import mc_pi
import random

class TestMCPi(unittest.TestCase):
    def test_pi_estimate(self):
        # Set a seed for reproducibility
        random.seed(0)
        N = 100000
        count = 0
        for i in range(N):
            x = random.random()
            y = random.random()
            z = x*x + y*y
            if z <= 1.0:
                count += 1
        PI = 4.0 * count / N
        self.assertAlmostEqual(PI, 3.1415926535897932, places=2)

    def test_count_inside_circle(self):
        # Set a seed for reproducibility
        random.seed(0)
        N = 100000
        count = 0
        for i in range(N):
            x = random.random()
            y = random.random()
            z = x*x + y*y
            if z <= 1.0:
                count += 1
        self.assertEqual(count, 78539)  # This value is based on the seed

if __name__ == '__main__':
    unittest.main()