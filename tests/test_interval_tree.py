import random
from typing import Generic, List, Optional, Tuple, TypeVar

import pytest

from kernels.layer._interval_tree import IntervalTree, _Node

T = TypeVar("T")


class SimpleIntervalStore(Generic[T]):
    """A simple O(n) implementation that stores intervals in a list."""

    def __init__(self):
        self.intervals: List[Tuple[int, int, T]] = []

    def insert(self, start: int, end: int, data: T) -> None:
        """Insert an interval into the store."""
        # Replace data if the interval already exists.
        for i, (existing_start, existing_end, existing_data) in enumerate(
            self.intervals
        ):
            if existing_start == start and existing_end == end:
                self.intervals[i] = (start, end, data)
                return

        self.intervals.append((start, end, data))

    def find_smallest_interval(self, point: int) -> Optional[T]:
        """Find the best match using linear search."""
        matches = []
        for start, end, data in self.intervals:
            if start <= point <= end:
                matches.append((start, end, data))

        if not matches:
            return None

        # Return the smallest interval, sort by memory location when
        # there are multiple matches with the same interval size. This
        # mirrors the ordering in the intervan tree.
        best_match = min(matches, key=lambda x: (x[1] - x[0], id(x[2])))
        return best_match[2]


def is_balanced(tree: IntervalTree[T]) -> bool:
    """Check if the AVL tree is properly balanced."""

    def check_balance(node: Optional[_Node[T]]) -> Tuple[bool, int]:
        if node is None:
            return True, 0

        # Left and right subtrees should be balanced.
        left_balanced, left_height = check_balance(node.left)
        if not left_balanced:
            return False, -1

        right_balanced, right_height = check_balance(node.right)
        if not right_balanced:
            return False, -1

        # The difference in height should not exceed 1.
        if abs(left_height - right_height) > 1:
            return False, -1

        # Check if the height is correct.
        expected_height = 1 + max(left_height, right_height)
        if node.height != expected_height:
            return False, -1

        return True, expected_height

    balanced, _ = check_balance(tree.root)
    return balanced


@pytest.fixture
def populated_tree() -> IntervalTree[str]:
    """Provides a pre-populated IntervalTree for testing."""
    tree = IntervalTree[str]()
    kernels = [
        (80, 89, "Kernel_A_General_80_89"),
        (86, 89, "Kernel_B_Ampere_86_89"),
        (80, 86, "Kernel_C_Older_Ampere_80_86"),
        (70, 75, "Kernel_D_Volta_70_75"),
        (86, 87, "Kernel_E_Specific_86_87"),
    ]
    for start, end, name in kernels:
        tree.insert(start, end, name)
    return tree


def test_find_smallest_interval_match_with_multiple_overlaps(populated_tree):
    # Check that the smallest inteval is selected when there are
    # multiple matching intervals.
    assert populated_tree.find_smallest_interval(86) == "Kernel_E_Specific_86_87"


def test_find_single_match(populated_tree):
    assert populated_tree.find_smallest_interval(72) == "Kernel_D_Volta_70_75"
    assert populated_tree.find_smallest_interval(75) == "Kernel_D_Volta_70_75"


def test_no_match_outside_all_ranges(populated_tree):
    # Check that no interval is found when the value is out of range
    # (too small/too large).
    assert populated_tree.find_smallest_interval(65) is None
    assert populated_tree.find_smallest_interval(95) is None


def test_no_match_in_gap_between_ranges(populated_tree):
    # Check that no interval is found when the value is between two
    # intervals.
    assert populated_tree.find_smallest_interval(78) is None


def test_boundary_conditions_start_and_end(populated_tree):
    # Test exact upper/lower bounds of intervals.
    assert populated_tree.find_smallest_interval(80) == "Kernel_C_Older_Ampere_80_86"
    assert populated_tree.find_smallest_interval(89) == "Kernel_B_Ampere_86_89"


def test_empty_tree():
    # Searching in an empty tree should return None.
    empty_tree = IntervalTree[str]()
    assert empty_tree.find_smallest_interval(100) is None


def test_multiple_equally_specific_matches():
    # Check that we pick the match in a stable way when there is are
    # multiple matching intervals with the same size.
    tree = IntervalTree[str]()
    str1 = "First_Narrow_Kernel"
    str2 = "Second_Narrow_Kernel"
    tree.insert(10, 20, "Wide_Kernel")
    tree.insert(12, 17, str1)
    tree.insert(14, 19, str2)

    if id(str1) < id(str2):
        assert tree.find_smallest_interval(15) == str1
    else:
        assert tree.find_smallest_interval(15) == str2


def test_property_based_interval_tree():
    # Quick-check property-based testing:
    #
    # - Verify that the tree is balanced after each insertion.
    # - Verify the query against a simple list-based implementation.

    random.seed(42)  # For reproducible tests

    test_points = list(range(0, 101))

    for _ in range(5):
        tree = IntervalTree[str]()
        simple = SimpleIntervalStore[str]()

        intervals = []
        for i in range(100):
            start = random.randint(0, 90)
            end = random.randint(start, 100)
            data = f"interval_{i}_s{start}_e{end}"
            intervals.append((start, end, data))

        for i, (start, end, data) in enumerate(intervals):
            tree.insert(start, end, data)
            simple.insert(start, end, data)

            # Check that tree is still balanced
            assert is_balanced(
                tree
            ), f"Tree became unbalanced after inserting interval {i}: ({start}, {end})"

            for point in test_points:
                tree_result = tree.find_smallest_interval(point)
                simple_result = simple.find_smallest_interval(point)

                assert tree_result == simple_result, (
                    f"Mismatch for point {point} after inserting {i + 1} intervals. "
                    f"Tree: {tree_result}, Simple: {simple_result}. "
                    f"Last inserted: ({start}, {end})"
                )


def test_property_based_edge_cases():
    random.seed(123)

    tree = IntervalTree[str]()
    simple = SimpleIntervalStore[str]()

    # Single-point intervals.
    for i in range(10):
        point = random.randint(0, 100)
        data = f"single_point_{i}_{point}"
        tree.insert(point, point, data)
        simple.insert(point, point, data)

        assert is_balanced(
            tree
        ), f"Tree unbalanced after inserting single point {point}"

        # Test the exact point and neighbors
        for test_point in [point - 1, point, point + 1]:
            if 0 <= test_point <= 100:
                tree_result = tree.find_smallest_interval(test_point)
                simple_result = simple.find_smallest_interval(test_point)
                assert tree_result == simple_result


def test_unique_intervals_override():
    """Test that inserting an interval with the same start/end overrides the previous value."""
    tree = IntervalTree[str]()

    tree.insert(10, 20, "original_value")
    assert tree.find_smallest_interval(15) == "original_value"

    tree.insert(10, 20, "new_value")
    assert tree.find_smallest_interval(15) == "new_value"

    tree.insert(10, 25, "different_interval")
    results = tree.search(15)
    assert "new_value" in results
    assert "different_interval" in results
    assert len(results) == 2

    tree.insert(10, 20, "final_value")
    assert tree.find_smallest_interval(15) == "final_value"

    assert is_balanced(tree)
