"""Unit tests for Core DSA modules: LRUCache and PrefixTrie.

Compatible with both standard library `unittest` and `pytest`.
"""

import unittest
from src.core.dsa.lru_cache import LRUCache
from src.core.dsa.trie import PrefixTrie


class TestLRUCache(unittest.TestCase):
    """Test suite verifying correctness, edge cases, and telemetry of LRUCache."""

    def test_initialization_invalid_capacity(self) -> None:
        """Cache should reject capacity <= 0."""
        with self.assertRaises(ValueError):
            LRUCache(capacity=0)
        with self.assertRaises(ValueError):
            LRUCache(capacity=-5)

    def test_basic_put_and_get(self) -> None:
        """Verify insertion, retrieval, and hit/miss counters."""
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        self.assertEqual(len(cache), 0)

        cache.put("k1", 100)
        cache.put("k2", 200)
        self.assertEqual(len(cache), 2)

        self.assertEqual(cache.get("k1"), 100)
        self.assertEqual(cache.get("k2"), 200)
        self.assertIsNone(cache.get("k3"))  # Miss

        self.assertEqual(cache.hits, 2)
        self.assertEqual(cache.misses, 1)

    def test_eviction_policy(self) -> None:
        """When capacity is exceeded, the least recently used item must be evicted."""
        cache: LRUCache[str, str] = LRUCache(capacity=2)
        cache.put("a", "alpha")
        cache.put("b", "beta")
        
        # Insert 3rd item; 'a' is LRU and should be evicted
        cache.put("c", "gamma")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "beta")
        self.assertEqual(cache.get("c"), "gamma")
        self.assertEqual(cache.evictions, 1)

    def test_recency_update_on_get(self) -> None:
        """Accessing an item with get() must make it most recently used."""
        cache: LRUCache[str, str] = LRUCache(capacity=2)
        cache.put("a", "alpha")
        cache.put("b", "beta")

        # Access 'a', making 'b' the LRU item
        self.assertEqual(cache.get("a"), "alpha")

        # Now put 'c'; 'b' should be evicted instead of 'a'
        cache.put("c", "gamma")
        self.assertEqual(cache.get("a"), "alpha")
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), "gamma")

    def test_peek_does_not_update_recency(self) -> None:
        """peek() should return value without altering LRU order."""
        cache: LRUCache[str, str] = LRUCache(capacity=2)
        cache.put("a", "alpha")
        cache.put("b", "beta")

        # Peek 'a'; 'a' remains the LRU candidate
        self.assertEqual(cache.peek("a"), "alpha")

        # Put 'c'; 'a' should be evicted because peek didn't update recency
        cache.put("c", "gamma")
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), "beta")

    def test_overwrite_existing_key(self) -> None:
        """Updating an existing key must update value without increasing size."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("x", 1)
        cache.put("x", 99)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get("x"), 99)

    def test_remove_operation(self) -> None:
        """remove() should delete the key in O(1) and update size."""
        cache: LRUCache[str, int] = LRUCache(capacity=3)
        cache.put("item1", 10)
        cache.put("item2", 20)

        self.assertTrue(cache.remove("item1"))
        self.assertFalse(cache.remove("non_existent"))
        self.assertIsNone(cache.get("item1"))
        self.assertEqual(len(cache), 1)

    def test_cache_telemetry_stats(self) -> None:
        """Verify telemetry snapshot and hit_rate calculation."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.get("a")  # hit
        cache.get("b")  # miss

        stats = cache.stats
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(float(stats["hit_rate"]), 0.5)

    def test_clear(self) -> None:
        """clear() must empty the cache and reset counters."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("a", 1)
        cache.get("a")
        cache.clear()

        self.assertEqual(len(cache), 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
        self.assertEqual(cache.evictions, 0)

    def test_dunder_methods(self) -> None:
        """Verify `in` operator, len(), and repr()."""
        cache: LRUCache[str, int] = LRUCache(capacity=2)
        cache.put("hello", 42)
        self.assertIn("hello", cache)
        self.assertNotIn("world", cache)
        self.assertIn("LRUCache", repr(cache))


class TestPrefixTrie(unittest.TestCase):
    """Test suite verifying PrefixTrie search, autocompletion, and entity extraction."""

    def test_insert_and_exact_search(self) -> None:
        """Verify inserting words and retrieving payloads."""
        trie: PrefixTrie[dict[str, str]] = PrefixTrie()
        trie.insert("NVDA", {"name": "Nvidia Corporation", "sector": "Semiconductors"})

        result = trie.search("NVDA")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "Nvidia Corporation")
        self.assertIsNone(trie.search("INTC"))

    def test_case_insensitivity_by_default(self) -> None:
        """Default mode should treat lookups case-insensitively."""
        trie: PrefixTrie[str] = PrefixTrie(case_sensitive=False)
        trie.insert("OpenAI", "AI Research Lab")

        self.assertEqual(trie.search("openai"), "AI Research Lab")
        self.assertEqual(trie.search("OPENAI"), "AI Research Lab")
        self.assertEqual(trie.search("OpenAI"), "AI Research Lab")

    def test_case_sensitive_mode(self) -> None:
        """Case sensitive mode must distinguish uppercase from lowercase."""
        trie: PrefixTrie[str] = PrefixTrie(case_sensitive=True)
        trie.insert("Apple", "Fruit/Tech")

        self.assertIsNotNone(trie.search("Apple"))
        self.assertIsNone(trie.search("apple"))
        self.assertIsNone(trie.search("APPLE"))

    def test_starts_with(self) -> None:
        """Verify prefix presence checking."""
        trie: PrefixTrie[int] = PrefixTrie()
        trie.insert("finance", 1)
        trie.insert("financial", 2)

        self.assertTrue(trie.starts_with("fin"))
        self.assertTrue(trie.starts_with("finance"))
        self.assertFalse(trie.starts_with("tech"))
        self.assertTrue(trie.starts_with(""))  # Empty prefix always True

    def test_autocomplete_and_frequency_ranking(self) -> None:
        """Autocomplete must return matching words ranked by insertion frequency."""
        trie: PrefixTrie[int] = PrefixTrie()
        trie.insert("tensor", 1)
        trie.insert("tensorflow", 2)
        trie.insert("tensorflow", 2)  # Inserted twice to increase frequency
        trie.insert("tenacity", 3)

        results = trie.autocomplete("ten")
        words = [w for w, _ in results]

        self.assertIn("tensorflow", words)
        self.assertIn("tensor", words)
        self.assertIn("tenacity", words)
        # 'tensorflow' has frequency=2, so it must be ranked first
        self.assertEqual(words[0], "tensorflow")

    def test_entity_extraction_from_text(self) -> None:
        """Scan text and locate all registered entities with start/end indices."""
        trie: PrefixTrie[str] = PrefixTrie()
        trie.insert("MSFT", "Microsoft")
        trie.insert("GOOGL", "Google")

        text = "Analysts upgraded MSFT and GOOGL today."
        entities = trie.extract_entities(text)

        self.assertEqual(len(entities), 2)
        # Check first extracted entity
        token, start, end, payload = entities[0]
        self.assertEqual(token, "MSFT")
        self.assertEqual(text[start:end], "MSFT")
        self.assertEqual(payload, "Microsoft")

    def test_delete_and_branch_preservation(self) -> None:
        """Deleting a word must not break child words or sibling branches."""
        trie: PrefixTrie[int] = PrefixTrie()
        trie.insert("app", 1)
        trie.insert("apple", 2)
        self.assertEqual(len(trie), 2)

        # Delete "app"; "apple" should remain untouched
        self.assertTrue(trie.delete("app"))
        self.assertEqual(len(trie), 1)
        self.assertIsNone(trie.search("app"))
        self.assertEqual(trie.search("apple"), 2)

        # Deleting non-existent word returns False
        self.assertFalse(trie.delete("app"))
        self.assertFalse(trie.delete("banana"))

    def test_empty_string_handling(self) -> None:
        """Trie should handle empty strings without crashing."""
        trie: PrefixTrie[str] = PrefixTrie()
        trie.insert("", "empty")
        self.assertIsNone(trie.search(""))
        self.assertFalse(trie.delete(""))


if __name__ == "__main__":
    unittest.main()
