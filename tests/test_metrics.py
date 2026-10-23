import unittest

from tinyocr.common import normalize, reading_order
from tinyocr.metrics import aggregate, counts


class MetricTests(unittest.TestCase):
    def test_known_edit_distance(self):
        self.assertEqual(counts("kitten", "sitting")["char_errors"], 3)
        self.assertEqual(counts("a b c", "a x c d")["word_errors"], 2)

    def test_empty_prediction_is_not_dropped(self):
        result = aggregate([counts("hello world", "")])
        self.assertEqual(result["cer"], 1)
        self.assertEqual(result["wer"], 1)

    def test_micro_weighting(self):
        result = aggregate([counts("a", ""), counts("123456789", "123456789")])
        self.assertEqual(result["cer"], 0.1)
        self.assertEqual(result["exact_match"], 0.5)

    def test_extra_text_can_exceed_100_percent(self):
        self.assertEqual(aggregate([counts("a", "abcd")])["cer"], 3)

    def test_blank_control(self):
        result = aggregate([counts("", "oops")])
        self.assertIsNone(result["cer"])
        self.assertEqual(result["blank_false_positive_chars"], 4)
        self.assertEqual(result["exact_match"], 0)
        self.assertEqual(aggregate([counts("", "")])["exact_match"], 1)

    def test_no_destructive_normalization(self):
        self.assertEqual(normalize("  A\nB  "), "a b")
        self.assertEqual(counts("Total: $12.50", "total $1250")["char_errors"], 2)
        self.assertEqual(counts("ABC", "abc")["char_errors"], 0)
        self.assertEqual(counts("ABC", "abc", casefold=False)["char_errors"], 3)
        self.assertEqual(normalize("e\u0301"), "é")

    def test_empty_set_rejected(self):
        with self.assertRaises(ValueError):
            aggregate([])

    def test_receipt_row_order(self):
        regions = [{"text": "12.50", "box": [200, 20, 250, 31]},
                   {"text": "Total", "box": [10, 21, 80, 32]},
                   {"text": "Thank you", "box": [10, 40, 100, 52]}]
        self.assertEqual([r["text"] for r in reading_order(regions)], ["Total", "12.50", "Thank you"])


if __name__ == "__main__":
    unittest.main()
