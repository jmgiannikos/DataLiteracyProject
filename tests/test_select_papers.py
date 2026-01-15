import unittest
from unittest.mock import MagicMock, patch
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from select_papers import (
    select_papers_by_author,
    select_papers_at_least_k_authors,
    get_co_authors,
    get_author_papers,
    generate_research_summary,
    is_first_author
)

class MockAuthor:
    def __init__(self, name):
        self.name = name

class MockResult:
    def __init__(self, authors, entry_id, doi=None, published=None):
        self.authors = [MockAuthor(a) for a in authors]
        self.entry_id = entry_id
        self.doi = doi
        self.published = published or datetime.now()
    
    def get_short_id(self):
        return self.entry_id

class TestSelectPapers(unittest.TestCase):
    
    @patch('select_papers.arxiv.Client')
    def test_select_papers_by_author(self, mock_client_cls):
        # Setup mock behavior
        mock_client = mock_client_cls.return_value
        
        # Test Data
        # Author A: 2 papers
        # Author B: 1 paper
        
        def results_side_effect(search):
            if 'au:"AuthorA"' in str(search.query):
                return [
                    MockResult(["AuthorA"], "id1", "doi1"), # AuthorA is first
                    MockResult(["AuthorA", "Others"], "id2", "doi2") # AuthorA is first
                ]
            elif 'au:"AuthorB"' in str(search.query):
                return [
                    MockResult(["AuthorB"], "id3") # AuthorB is first
                ]
            return []
            
        mock_client.results.side_effect = results_side_effect
        
        authors = ["AuthorA", "AuthorB"]
        results = select_papers_by_author(authors, n=5)
        
        self.assertEqual(len(results["AuthorA"]), 2)
        self.assertIn("doi1", results["AuthorA"])
        
        self.assertEqual(len(results["AuthorB"]), 0) # No DOI -> Excluded

    @patch('select_papers.arxiv.Client')
    def test_select_papers_at_least_k_authors(self, mock_client_cls):
        mock_client = mock_client_cls.return_value
        
        # Scenario:
        # P1: [Alice, Bob] (Overlap 2, Alice is first (in list)) -> KEEP
        # P2: [Alice] (Overlap 1) -> Exclude by K=2 check
        # P3: [Bob, Charlie] (Overlap 2, Bob is first (in list)) -> KEEP
        # P4: [Dave, Alice, Bob] (Overlap 2, Dave is first (NOT in list)) -> EXCLUDE
        
        # Mock Objects
        p1 = MockResult(["Alice", "Bob"], "id1", "doi1")
        p2 = MockResult(["Alice"], "id2", "doi2")
        p3 = MockResult(["Bob", "Charlie"], "id3", "doi3")
        p4 = MockResult(["Dave", "Alice", "Bob"], "id4", "doi4")
        
        def results_side_effect(search):
            q = str(search.query)
            if 'au:"Alice"' in q: return [p1, p2, p4]
            if 'au:"Bob"' in q: return [p1, p3, p4]
            if 'au:"Charlie"' in q: return [p3]
            return []
            
        mock_client.results.side_effect = results_side_effect
        
        # Test k=2
        # Input authors: Alice, Bob, Charlie.
        # P1: First=Alice (in list). Overlap=Alice,Bob (2). -> OK
        # P3: First=Bob (in list). Overlap=Bob,Charlie (2). -> OK
        # P4: First=Dave (NOT in list). Overlap=Alice,Bob (2). -> FAIL First Author Check

        results = select_papers_at_least_k_authors(["Alice", "Bob", "Charlie"], n=5, k=2)
        
        self.assertIn("doi1", results)
        self.assertIn("doi3", results)
        self.assertNotIn("doi2", results) # K failure
        self.assertNotIn("doi4", results) # First Author failure
        self.assertEqual(len(results), 2)
        
        # Test k=1
        # P2: First=Alice (in list). Overlap=1 -> OK.
        # P4: First=Dave (NOT in list). Overlap=2 -> FAIL First Author Check.
        results_k1 = select_papers_common(["Alice", "Bob", "Charlie"], n=5, k=1)
        self.assertIn("doi2", results_k1)
        self.assertNotIn("doi4", results_k1)

if __name__ == '__main__':
    unittest.main()
