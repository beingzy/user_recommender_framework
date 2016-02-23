import unittest
from distance_metrics.distance import CategoryWrapper


class TestCategoryWrapper(unittest.TestCase):
    def setUp(self):
        x = [1, 2, 'a', 5.5, 'b']
        y = [4, 0, 'a', 2.1, 'c']
        self._data = {"x": x, "y": y}
        self._catwrapper = CategoryWrapper(category_index=[2, 4])

    def test_num_elements(self):
        num_elements, _ = self._catwrapper.wrapper(self._data["x"])
        true_num_elements = [1, 2, 5.5]
        self.assertEqual(num_elements, true_num_elements)

    def test_cat_elements(self):
        _, cat_elements = self._catwrapper.wrapper(self._data["x"])
        true_cat_elements = ['a', 'b']
        self.assertEqual(cat_elements, true_cat_elements)

    def test_num_component_differnce(self):
        num_diff, cat_diff = self._catwrapper.get_component_difference(self._data["x"], self._data["y"])
        true_num_diff = [-3, 2, 3.4]
        self.assertEqual(num_diff, true_num_diff)

    def test_cat_component_differnce(self):
        num_diff, cat_diff = self._catwrapper.get_component_difference(self._data["x"], self._data["y"])
        true_cat_diff = [0, 1]
        self.assertEqual(cat_diff, true_cat_diff)

    def test_cat_differnce(self):
        res = self._catwrapper.get_difference(self._data["x"], self._data["y"])
        true = [-3, 2, 0, 3.4, 1]
        self.assertEqual(res, true)


if __name__ == '__main__':
    unittest.main()
