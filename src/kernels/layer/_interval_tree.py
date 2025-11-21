# AVL-balanced interval trees. We could use the intervaltree
# packages, but it seems unmaintained and does not have type
# annotations.

from typing import Generic, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class _Node(Generic[T]):
    """A node in the interval tree."""

    def __init__(self, start: int, end: int, data: T):
        self.start: int = start
        self.end: int = end
        self.data: T = data
        self.max_end: int = end
        self.left: Optional["_Node[T]"] = None
        self.right: Optional["_Node[T]"] = None
        self.height: int = 1

    def __repr__(self) -> str:
        return f"Node({self.start}, {self.end})"


class IntervalTree(Generic[T]):
    """A data structure to hold and query (unique) intervals."""

    root: Optional[_Node[T]]

    def __init__(self):
        self.root = None

    def insert(self, start: int, end: int, data: T) -> None:
        """
        Inserts a new interval into the tree.

        Args:
            start: The starting point of the interval.
            end: The ending point of the interval.
            data: The data associated with this interval.
        """
        self.root = self._insert(self.root, start, end, data)

    def _get_height(self, node: Optional[_Node[T]]) -> int:
        if not node:
            return 0
        return node.height

    def _get_balance(self, node: Optional[_Node[T]]) -> int:
        if not node:
            return 0
        return self._get_height(node.left) - self._get_height(node.right)

    def _update_node_attributes(self, node: _Node[T]) -> None:
        node.height = 1 + max(self._get_height(node.left), self._get_height(node.right))
        node.max_end = node.end
        if node.left:
            node.max_end = max(node.max_end, node.left.max_end)
        if node.right:
            node.max_end = max(node.max_end, node.right.max_end)

    def _right_rotate(self, y: _Node[T]) -> _Node[T]:
        """Performs a right rotation."""
        x = y.left
        assert x is not None
        T2 = x.right

        x.right = y
        y.left = T2

        self._update_node_attributes(y)
        self._update_node_attributes(x)

        return x

    def _left_rotate(self, x: _Node[T]) -> _Node[T]:
        """Performs a left rotation."""
        y = x.right
        assert y is not None
        T2 = y.left

        y.left = x
        x.right = T2

        self._update_node_attributes(x)
        self._update_node_attributes(y)

        return y

    def _insert(
        self, node: Optional[_Node[T]], start: int, end: int, data: T
    ) -> _Node[T]:
        """Recursive helper to insert a new node and balance the tree."""
        if not node:
            return _Node(start, end, data)

        # Replace the data if the interval already exists.
        if start == node.start and end == node.end:
            node.data = data
            return node

        if start < node.start:
            node.left = self._insert(node.left, start, end, data)
        else:
            node.right = self._insert(node.right, start, end, data)

        self._update_node_attributes(node)

        balance = self._get_balance(node)

        # Left Left Case
        if balance > 1 and node.left and start < node.left.start:
            return self._right_rotate(node)

        # Right Right Case
        if balance < -1 and node.right and start >= node.right.start:
            return self._left_rotate(node)

        # Left Right Case
        if balance > 1 and node.left and start >= node.left.start:
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # Right Left Case
        if balance < -1 and node.right and start < node.right.start:
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    def search(self, point: int) -> List[T]:
        """
        Searches for all intervals that contain the given point.

        Args:
            point: The point to search for.

        Returns:
            A list of data items from all matching intervals.
        """
        results: List[T] = []
        self._search(self.root, point, results)
        return results

    def _search(self, node: Optional[_Node[T]], point: int, results: List[T]) -> None:
        """Recursive helper to find all overlapping intervals."""
        if node is None or point > node.max_end:
            return

        if node.left:
            self._search(node.left, point, results)

        if node.start <= point <= node.end:
            results.append(node.data)

        if point >= node.start and node.right:
            self._search(node.right, point, results)

    def find_smallest_interval(self, point: int) -> Optional[T]:
        """
        Finds the item with the most specific (smallest) range for a given point.

        Args:
            point: The capability to look up.

        Returns:
            The data of the best-matching item, or None if no match is found.
        """
        matches: List[Tuple[int, int, T]] = []
        self._find_with_intervals(self.root, point, matches)

        if not matches:
            return None

        # Return the smallest interval, sort by memory location when
        # there are multiple matches with the same interval size. This
        # is just to ensure that we can compare against a trivial
        # implementation in tests.
        best_match = min(matches, key=lambda x: (x[1] - x[0], id(x[2])))
        return best_match[2]

    def _find_with_intervals(
        self,
        node: Optional[_Node[T]],
        point: int,
        results: List[Tuple[int, int, T]],
    ) -> None:
        """A modified search that collects interval ranges along with data."""
        if node is None or point > node.max_end:
            return

        if node.left:
            self._find_with_intervals(node.left, point, results)

        if node.start <= point <= node.end:
            results.append((node.start, node.end, node.data))

        if point >= node.start and node.right:
            self._find_with_intervals(node.right, point, results)
