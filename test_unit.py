import unittest
from loadPic import hand_recognition
from model import KeyPointClassifier

class TestStringMethods(unittest.TestCase):
    classifier = KeyPointClassifier()

    def test_enter(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('enter',i,self.classifier)
            tests.append((sign_id,0))

        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)

    def test_right(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('right',i,self.classifier)
            tests.append((sign_id,1))

        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)

    def test_left(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('left',i,self.classifier)
            tests.append((sign_id,2))
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)

    def test_zoomin(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('zoomin',i,self.classifier)
            tests.append((sign_id,3))
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)


    def test_zoomout(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('zoomout',i,self.classifier)
            tests.append((sign_id,4))
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)

    def test_up(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('up',i,self.classifier)
            tests.append((sign_id,5))
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)


    def test_down(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('down',i,self.classifier)
            tests.append((sign_id,6))
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)

    def test_other(self):

        tests = []
        for i in range(1,1000):
            sign_id = hand_recognition('other',i,self.classifier)
            tests.append((sign_id,i))
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(value, expected)

if __name__ == '__main__':
    unittest.main()