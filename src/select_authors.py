import arxiv
import logging
from collections import Counter
import itertools
from typing import List, Set, Tuple, Optional

# max results out of arxiv query
MAX_RESULTS = 200
MAX_CLIQUE = 30

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_papers_by_author(author_name: str, max_results: int =MAX_RESULTS  ) -> List[arxiv.Result]:
    """
    Fetches papers for a given author using the arXiv API.
    """
    search = arxiv.Search(
        query=f'au:"{author_name}"',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    client = arxiv.Client()
    results = []
    try:
        results = list(client.results(search))
    except Exception as e:
        logger.error(f"Error fetching papers for {author_name}: {e}")
    return results

def is_first_author(paper: arxiv.Result, author_name: str) -> bool:
    """
    Checks if the given author is the first author of the paper.
    Checks index 0 of the authors list. 
    Approximate match for name comparison.
    """
    if not paper.authors:
        return False
    first_author = paper.authors[0].name
    
    def normalize(name):
        return name.lower().replace(".", "").replace(" ", "")
    
    return normalize(author_name) in normalize(first_author)

def select_authors_single(entrypoint: str, n: int, thresh: int) -> List[str]:
    """
    Find `n`-1 authors that appear as co-authors in at least `thresh` papers where `entrypoint` is first author.
    """
    logger.info(f"Starting Single Author Selection for {entrypoint}")

    all_papers = get_papers_by_author(entrypoint, max_results=MAX_RESULTS)
    
    # First author papers
    first_author_papers = [p for p in all_papers if is_first_author(p, entrypoint)]
    logger.info(f"Found {len(first_author_papers)} papers where {entrypoint} is first author.")
    
    # Co-authorship count
    co_author_counts = Counter()
    for paper in first_author_papers:
        for author in paper.authors:
            name = author.name
            if (name == entrypoint) or (entrypoint.lower() in name.lower()):
                continue
            co_author_counts[name] += 1
            
    # Filter by count >= thresh
    candidates = [name for name, count in co_author_counts.items() if count >= thresh]
    
    # Take top n-1 most common
    candidates_sorted = sorted(candidates, key=lambda x: co_author_counts[x], reverse=True)
    result = candidates_sorted[:n-1]
    
    logger.info(f"Selected {len(result)} authors.")
    return result

def select_authors_graph(entrypoint: str, n: int, thresh: int, visualize: bool = False) -> List[str]:
    """
    Finds n authors (including entrypoint) such that each shares at least 
    `thresh` coauthorships with each other. i.e. find a clique.
    If visualize is True, saves a graph of the clique.
    """
    logger.info(f"Starting Graph Author Selection for {entrypoint}")
    
    # STEP 1: Start with all co-authors of `entrypoint`    
    entry_papers = get_papers_by_author(entrypoint, max_results=MAX_RESULTS)
    
    # Count collaborations between `entrypoint` and other authors to filter out
    pool_counts = Counter()
    for p in entry_papers:
        authors = [a.name for a in p.authors]
        if entrypoint not in str(authors): # Safety check if name parsing fails?
             pass
        
        for a in authors:
             pool_counts[a] += 1
             
    # everyone must have coauthorships count >= thresh with `entrypoint` and everyone else
    potential_members = [name for name, count in pool_counts.items() if count >= thresh]
    if len(potential_members) < n:
        logger.warning(f"Not enough candidates connected to {entrypoint} with threshold {thresh}. Found {len(potential_members)}.")
        return potential_members 
        
    top_candidates = sorted(potential_members, key=lambda x: pool_counts[x], reverse=True)[:MAX_CLIQUE]
    
    # STEP 2: Build full graph
    # Avoid duplicate paper processing
    paper_ids = set()
    all_fetched_papers = []
    
    # Gather all papers first
    authors_to_fetch = [entrypoint] + [c for c in top_candidates if c != entrypoint]

    logging.info(f"Fetching papers for {len(authors_to_fetch)} authors to build graph...")
    
    for auth in authors_to_fetch:
        ps = get_papers_by_author(auth, max_results=MAX_RESULTS)
        for p in ps:
            pid = p.entry_id # Stable ID
            if pid not in paper_ids:
                paper_ids.add(pid)
                all_fetched_papers.append(p)
                
    # Build Graph
    graph_edges = Counter()
    for p in all_fetched_papers:
        p_auth_names = [a.name for a in p.authors]
        
        present_candidates = []
        for cand in top_candidates:
            for pa in p_auth_names:
                if cand == pa:
                    present_candidates.append(cand)
                    break
        
        # Add edges for clique
        for u, v in itertools.combinations(present_candidates, 2):
            # Sort to ensure undirected key
            k = tuple(sorted((u, v)))
            graph_edges[k] += 1
            
    logger.info(f"Graph construction complete. Nodes: {len(top_candidates)}, Edges: {len(graph_edges)}")
            
    # Find clique: a set S, |S| = n, entrypoint in S, for all u,v in S: graph_edges[(u,v)] >= threshold
    
    valid_cliques = []
    
    def find_clique(current_set, candidates_left):
        if len(current_set) == n:
            valid_cliques.append(list(current_set))
            return True
        
        if not candidates_left:
            return False
            
        # Try adding next
        cand = candidates_left[0]
        remaining = candidates_left[1:]
        
        can_add = True
        for member in current_set:
            k = tuple(sorted((cand, member)))
            if graph_edges[k] < thresh:
                can_add = False
                break
        
        if can_add:
            # Branch 1: Add cand
            if find_clique(current_set + [cand], remaining):
                return True
                
        # Branch 2: Skip cand
        if find_clique(current_set, remaining):
            return True
            
        return False

    # Filter candidates_left to strictly those filtered previously
    # We need to remove entrypoint from candidates_left to avoid dup
    candidates_for_search = [c for c in top_candidates if c != entrypoint]
    
    find_clique([entrypoint], candidates_for_search)
    
    if valid_cliques:
        result = valid_cliques[0]
        
        if visualize:
            try:
                import networkx as nx
                import matplotlib.pyplot as plt
                
                G = nx.Graph()
                for member in result:
                    G.add_node(member)
                    
                for u, v in itertools.combinations(result, 2):
                    weight = graph_edges.get(tuple(sorted((u, v))), 0)
                    G.add_edge(u, v, weight=weight)
                
                plt.figure(figsize=(10, 8))
                pos = nx.spring_layout(G, k=0.5)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue')
                nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
                
                # Draw edges with weight labels
                edges = G.edges(data=True)
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=2)
                edge_labels = {(u, v): d['weight'] for u, v, d in edges}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                
                plt.title(f"Author Clique (Threshold: {thresh})", fontsize=15)
                plt.axis('off')
                
                output_file = "author_clique_graph.png"
                plt.savefig(output_file)
                logger.info(f"Graph visualization saved to {output_file}")
                plt.close()
                
            except ImportError:
                logger.warning("networkx or matplotlib not installed, skipping visualization.")
            except Exception as e:
                logger.error(f"Error during visualization: {e}")
                
        return result
    else:
        logger.warning(f"Could not find a clique of size {n} with threshold {thresh}.")
        return []

if __name__ == "__main__":
    # Smoke test with a recent enough author:
    # There are 4 first-author papers for Alec Radford
    # Out of those, the most common co-author is Ilya Sutskever (3 papers)
    # The next most common are Jong Wook Kim, Tao Xu, and Greg Brockman (1 paper each)
    authors = select_authors_single("Alec Radford", 5, 1)
    print("Single:", authors)

    # Vibe-coded visualization to be sure it works
    authors = select_authors_graph("Alec Radford", 3, 7, visualize=True)
    print("Graph:", authors)
