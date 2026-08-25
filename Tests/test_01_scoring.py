import unittest

from Dart_Scoring.DartScore import Score, undo_last_throw


class Test01Scoring(unittest.TestCase):
    def test_standard_turn_reduces_a_501_score(self):
        score = Score(501, doubleOut=True)
        points, ends_on_double = score.calculatePoints(20, 3, 20, 3, 20, 3)

        self.assertEqual((points, ends_on_double), (180, False))
        self.assertTrue(score.pointsScored(points, ends_on_double))
        self.assertEqual(score.currentScore, 321)

    def test_outer_and_inner_bulls_are_valid(self):
        score = Score(501, doubleOut=True)

        self.assertEqual(score.calculatePoints(25, 1, 20, 3, 0, 1), (85, False))
        self.assertEqual(score.calculatePoints(20, 1, 0, 1, 50, 1), (70, True))

    def test_inner_bull_can_finish_a_double_out_game(self):
        score = Score(50, doubleOut=True)
        points, ends_on_double = score.calculatePoints(0, 1, 0, 1, 50, 1)

        self.assertTrue(score.pointsScored(points, ends_on_double))
        self.assertEqual(score.currentScore, 0)
        self.assertTrue(score.finished)

    def test_outer_bull_cannot_finish_a_double_out_game(self):
        score = Score(25, doubleOut=True)
        points, ends_on_double = score.calculatePoints(0, 1, 0, 1, 25, 1)

        self.assertFalse(score.pointsScored(points, ends_on_double))
        self.assertEqual(score.currentScore, 25)
        self.assertFalse(score.finished)

    def test_busts_leave_the_score_unchanged(self):
        score = Score(40, doubleOut=True)

        self.assertFalse(score.pointsScored(41, hitDouble=True))
        self.assertEqual(score.currentScore, 40)
        self.assertFalse(score.pointsScored(39, hitDouble=False))
        self.assertEqual(score.currentScore, 40)

    def test_undo_removes_only_the_last_uncommitted_throw(self):
        values = [20, 5]
        multipliers = [3, 2]

        self.assertEqual(undo_last_throw(values, multipliers), (5, 2))
        self.assertEqual(values, [20])
        self.assertEqual(multipliers, [3])
        self.assertIsNone(undo_last_throw([], []))

    def test_invalid_bull_multiplier_is_rejected(self):
        with self.assertRaises(ValueError):
            Score(501).calculatePoints(25, 2, 0, 1, 0, 1)


if __name__ == "__main__":
    unittest.main()
