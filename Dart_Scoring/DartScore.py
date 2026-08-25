class Score:
    """State and rules for an ``01`` darts game.

    A bull is represented as either 25 (outer bull) or 50 (inner bull), both
    with multiplier 1.  The inner bull counts as a double for double-out;
    the outer bull does not.
    """

    def __init__(self, nominalScore, doubleOut=True):
        if nominalScore < 1:
            raise ValueError("The starting score must be positive")
        self.nominalScore = nominalScore
        self.currentScore = nominalScore
        self.doubleOut = doubleOut
        self.finished = False

    def setNominalScore(self, points):
        if points < 1:
            raise ValueError("The starting score must be positive")
        self.nominalScore = points
        self.currentScore = points
        self.finished = False

    def getCurrentScore(self):
        return self.currentScore

    def setCurrentScore(self, points):
        self.currentScore = points

    def pointsScored(self, points, hitDouble):
        """
        Subtract the scored points from the current score and check, if the game is finished.
        """
        if points < 0:
            raise ValueError("Points scored cannot be negative")

        remaining = self.currentScore - points
        is_bust = remaining < 0 or (self.doubleOut and remaining == 1)
        invalid_checkout = self.doubleOut and remaining == 0 and not hitDouble

        if is_bust:
            print("Overthrown! You have only " + str(self.currentScore) + " points left!")
            return False
        if invalid_checkout:
            print("You cannot finish without hitting a Double!")
            return False

        self.currentScore = remaining
        print(f"Current Score: {self.currentScore}")

        # Check, if the player has won the game
        if self.currentScore == 0:
            self.finished = True
            print("Congratulations!!! You won the game!")
        return True

    def calculatePoints(self, firstThrow, firstMultiplier, secondThrow, secondMultiplier, thirdThrow, thirdMultiplier):
        """
        Calculate the thrown points and check, if a double field was hit.
        The inputs are directly the fields, which were hit.
        """
        throws = (
            (firstThrow, firstMultiplier),
            (secondThrow, secondMultiplier),
            (thirdThrow, thirdMultiplier),
        )
        return self.calculate_points(throws)

    @staticmethod
    def calculate_points(throws):
        """Return ``(points, ends_on_double)`` for exactly three throws.

        ``0`` is permitted for a missed board.  Values 25 and 50 are valid
        bulls only with multiplier 1.
        """
        throws = tuple(throws)
        if len(throws) != 3:
            raise ValueError("A turn must contain exactly three throws")

        total = 0
        ends_on_double = False
        for index, (value, multiplier) in enumerate(throws):
            if not isinstance(value, int) or not isinstance(multiplier, int):
                raise ValueError("Throw values and multipliers must be integers")
            if value in (25, 50):
                if multiplier != 1:
                    raise ValueError("Bull values must use multiplier 1")
                total += value
            elif 0 <= value <= 20 and 1 <= multiplier <= 3:
                total += value * multiplier
            else:
                raise ValueError("Invalid dart throw")

            if index == len(throws) - 1:
                ends_on_double = multiplier == 2 or value == 50

        return total, ends_on_double


def undo_last_throw(values_of_round, mults_of_round):
    """Remove and return the most recent uncommitted throw, if there is one."""
    if len(values_of_round) != len(mults_of_round):
        raise ValueError("Round values and multipliers are out of sync")
    if not values_of_round:
        return None
    return values_of_round.pop(), mults_of_round.pop()
