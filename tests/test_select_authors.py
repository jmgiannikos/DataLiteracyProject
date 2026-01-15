import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from select_authors import select_authors_single, select_authors_graph, is_first_author

class MockAuthor:
    def __init__(self, name):
        self.name = name

class MockResult:
    def __init__(self, authors, entry_id="1"):
        self.authors = [MockAuthor(a) for a in authors]
        self.entry_id = entry_id

class TestSelectAuthors(unittest.TestCase):
    
    def test_is_first_author(self):
        # Basic cases
        paper = MockResult(["Alice", "Bob"])
        self.assertTrue(is_first_author(paper, "Alice"))
        self.assertFalse(is_first_author(paper, "Bob"))
        
        # Case insensitivity and whitespace
        self.assertTrue(is_first_author(paper, "ALICE")) 
        self.assertTrue(is_first_author(paper, "Alice ")) 
        
        # Initial handling (simple containment as per implementation)
        paper2 = MockResult(["Alice B. Name", "Bob"])
        self.assertTrue(is_first_author(paper2, "Alice B Name"))
        self.assertTrue(is_first_author(paper2, "Alice")) # "Alice" in "Alice B Name"
        
        # Empty authors
        paper3 = MockResult([])
        self.assertFalse(is_first_author(paper3, "Alice"))

    @patch('select_authors.get_papers_by_author')
    def test_select_authors_single(self, mock_get_papers):
        # Scenario: 
        # Entrypoint: "Alice"
        # Papers:
        # 1. [Alice, Bob] (Alice 1st) -> Bob co-author count: 1
        # 2. [Alice, Bob, Charlie] (Alice 1st) -> Bob: 2, Charlie: 1
        # 3. [Bob, Alice] (Bob 1st) -> Ignored (Alice not 1st)
        # 4. [Alice, Charlie] (Alice 1st) -> Charlie: 2
        # 5. [Alice] (Single author) -> Ignored for co-author count
        # 6. [Alice, Dave] (Alice 1st) -> Dave: 1
        
        # Expected Counts: Bob: 2, Charlie: 2, Dave: 1
        
        mock_get_papers.return_value = [
            MockResult(["Alice", "Bob"], "1"),
            MockResult(["Alice", "Bob", "Charlie"], "2"),
            MockResult(["Bob", "Alice"], "3"),
            MockResult(["Alice", "Charlie"], "4"),
            MockResult(["Alice"], "5"),
            MockResult(["Alice", "Dave"], "6")
        ]
        
        # Case 1: n=3, thresh=2
        # Candidates >= 2: Bob (2), Charlie (2).
        # Sorted: Bob, Charlie (order might vary strictly by count, but stable sort or key needed? code uses count order)
        # n-1 = 2 results.
        results = select_authors_single("Alice", n=3, thresh=2)
        self.assertEqual(len(results), 2)
        self.assertCountEqual(results, ["Bob", "Charlie"])
        
        # Case 2: n=2, thresh=2
        # n-1 = 1 result. Should be one of Bob or Charlie.
        results_n2 = select_authors_single("Alice", n=2, thresh=2)
        self.assertEqual(len(results_n2), 1)
        self.assertTrue(results_n2[0] in ["Bob", "Charlie"])
        
        # Case 3: thresh=3
        # No candidates >= 3
        results_t3 = select_authors_single("Alice", n=3, thresh=3)
        self.assertEqual(results_t3, [])
        
        # Case 4: thresh=1
        # Candidates: Bob(2), Charlie(2), Dave(1).
        # Sorted: Bob, Charlie, Dave.
        # n=4 -> n-1=3 results.
        results_t1 = select_authors_single("Alice", n=4, thresh=1)
        self.assertEqual(len(results_t1), 3)
        self.assertCountEqual(results_t1, ["Bob", "Charlie", "Dave"])

    @patch('select_authors.get_papers_by_author')
    def test_select_authors_graph(self, mock_get_papers):
        # Existing graph test is fine, but let's just make sure it runs.
        # We can simulate the scenario:
        # Clique {A, B, C}. 
        # A connected to B, C
        # B connected to A, C
        # C connected to A, B
        # D connected to A only.
        
        def side_effect(author_name, max_results=100):
            if author_name == "Alice":
                return [MockResult(["Alice", "Bob"], "1"), MockResult(["Alice", "Charlie"], "2"), MockResult(["Alice", "Dave"], "3")]
            elif author_name == "Bob":
                return [MockResult(["Alice", "Bob"], "1"), MockResult(["Bob", "Charlie"], "4")]
            elif author_name == "Charlie":
                return [MockResult(["Alice", "Charlie"], "2"), MockResult(["Bob", "Charlie"], "4")]
            return []
            
        mock_get_papers.side_effect = side_effect
        
        clique = select_authors_graph("Alice", n=3, thresh=1)
        self.assertEqual(len(clique), 3)
        self.assertCountEqual(clique, ["Alice", "Bob", "Charlie"])

if __name__ == '__main__':
    unittest.main()
