"""Prefix Trie (Retrieval Tree) data structure for entity matching and prefix search.

Provides O(K) lookup, insertion, and autocomplete where K is key length.
Essential for exact financial ticker lookup, entity recognition, and dictionary filtering.
"""

from typing import Any, Generic, Optional, TypeVar

V = TypeVar("V")


class TrieNode(Generic[V]):
    """Individual node within the Prefix Trie."""

    __slots__ = ("children", "is_end_of_word", "value", "frequency")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode[V]] = {}
        self.is_end_of_word: bool = False
        self.value: Optional[V] = None
        self.frequency: int = 0


class PrefixTrie(Generic[V]):
    """High-performance Prefix Trie for dictionary indexing and entity extraction.

    Supports exact word matching, prefix autocomplete, and text scanning.

    Complexity:
        - Insert: O(K)
        - Search: O(K)
        - Prefix Query: O(P + S) where P is prefix length, S is total characters in matching subtrees.
    """

    def __init__(self, case_sensitive: bool = False) -> None:
        self.root: TrieNode[V] = TrieNode()
        self.case_sensitive: bool = case_sensitive
        self._total_words: int = 0

    @property
    def total_words(self) -> int:
        """Return the count of unique words stored in the Trie."""
        return self._total_words

    def _normalize(self, key: str) -> str:
        """Normalize token case if case-insensitive mode is enabled."""
        return key if self.case_sensitive else key.lower()

    def insert(self, word: str, value: Optional[V] = None) -> None:
        """Insert a word and its optional associated metadata payload into the Trie.

        Time Complexity: O(K) where K is the length of the word.
        """
        if not word:
            return

        normalized = self._normalize(word)
        current = self.root

        for char in normalized:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]

        if not current.is_end_of_word:
            self._total_words += 1

        current.is_end_of_word = True
        current.value = value
        current.frequency += 1

    def search(self, word: str) -> Optional[V]:
        """Search for an exact word match.

        Returns:
            Optional[V]: The payload value if word exists, otherwise None.

        Time Complexity: O(K)
        """
        if not word:
            return None

        current = self._traverse_to_node(word)
        if current and current.is_end_of_word:
            return current.value
        return None

    def starts_with(self, prefix: str) -> bool:
        """Check if any word in the Trie begins with the given prefix.

        Time Complexity: O(P) where P is prefix length.
        """
        if not prefix:
            return True
        return self._traverse_to_node(prefix) is not None

    def autocomplete(self, prefix: str, limit: int = 10) -> list[tuple[str, Optional[V]]]:
        """Find all words sharing the given prefix, ranked by frequency.

        Args:
            prefix: The starting string to complete.
            limit: Maximum number of completions to return.

        Returns:
            list of (matched_word, value) tuples sorted by frequency descending.
        """
        node = self._traverse_to_node(prefix) if prefix else self.root
        if not node:
            return []

        results: list[tuple[str, Optional[V], int]] = []
        normalized_prefix = self._normalize(prefix)

        def _dfs(current: TrieNode[V], path: list[str]) -> None:
            if current.is_end_of_word:
                results.append((normalized_prefix + "".join(path), current.value, current.frequency))

            for char, next_node in current.children.items():
                path.append(char)
                _dfs(next_node, path)
                path.pop()

        _dfs(node, [])
        # Rank suggestions by highest frequency first
        results.sort(key=lambda item: item[2], reverse=True)
        return [(word, val) for word, val, _ in results[:limit]]

    def extract_entities(self, text: str) -> list[tuple[str, int, int, Optional[V]]]:
        """Scan raw unstructured text and extract all matching registered dictionary entities.

        Returns:
            list of tuples: (matched_entity, start_idx, end_idx, payload_value)
        """
        if not text:
            return []

        matched_entities: list[tuple[str, int, int, Optional[V]]] = []
        n = len(text)
        text_to_scan = text if self.case_sensitive else text.lower()

        # Sliding window prefix match starting from each character index
        for i in range(n):
            current = self.root
            longest_match: Optional[tuple[str, int, int, Optional[V]]] = None

            for j in range(i, n):
                char = text_to_scan[j]
                if char not in current.children:
                    break
                current = current.children[char]
                if current.is_end_of_word:
                    longest_match = (text[i : j + 1], i, j + 1, current.value)

            if longest_match:
                matched_entities.append(longest_match)

        return matched_entities

    def delete(self, word: str) -> bool:
        """Remove a word from the Trie and prune redundant branch nodes.

        Returns:
            bool: True if the word was found and deleted, False otherwise.

        Time Complexity: O(K)
        """
        if not word:
            return False

        normalized = self._normalize(word)
        deleted = False

        def _delete_helper(current: TrieNode[V], index: int) -> bool:
            nonlocal deleted
            if index == len(normalized):
                if not current.is_end_of_word:
                    return False
                current.is_end_of_word = False
                current.value = None
                self._total_words -= 1
                deleted = True
                # Prune if this node has no children
                return len(current.children) == 0

            char = normalized[index]
            if char not in current.children:
                return False

            should_prune = _delete_helper(current.children[char], index + 1)
            if should_prune:
                del current.children[char]
                return not current.is_end_of_word and len(current.children) == 0
            return False

        _delete_helper(self.root, 0)
        return deleted

    def _traverse_to_node(self, prefix: str) -> Optional[TrieNode[V]]:
        """Helper to walk the Trie down to the final character node of the prefix."""
        normalized = self._normalize(prefix)
        current = self.root
        for char in normalized:
            if char not in current.children:
                return None
            current = current.children[char]
        return current

    def __contains__(self, word: str) -> bool:
        return self.search(word) is not None

    def __len__(self) -> int:
        return self._total_words

    def __repr__(self) -> str:
        return f"PrefixTrie(total_words={self._total_words}, case_sensitive={self.case_sensitive})"
