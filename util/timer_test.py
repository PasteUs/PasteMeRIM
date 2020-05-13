import unittest
import time
from util import timer


class MyTimerTestCase(unittest.TestCase):
    def test_timer(self):

        with timer():
            time.sleep(1)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
