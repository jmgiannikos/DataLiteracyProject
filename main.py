# Pipeline initiator file
# select authors -> select papers -> scrape metadata -> scrape text -> clean text -> extract features
# visualization will be done in a separate jupyter notebook

from src.scrape_paper_ids import get_papers
from src.scrape_metadata import scrape_metadata_arxivIDs

arxiv_ids = get_papers(
    entry="Florentin Millour",
    n=5,  # Get co-authors
    j=5,  # first-author papers per author
    k=5  # non-first-author papers per author
)

print(f"Testing with {len(arxiv_ids)} arXiv papers:")
for arxiv_id in arxiv_ids:
    print(f"  - {arxiv_id}")
print()

metadata_dataframe = scrape_metadata_arxivIDs(arxiv_ids)

print(f"Processed {len(metadata_dataframe)} papers")
print(f"\nDOIs found: {metadata_dataframe['doi'].notna().sum()}/{len(metadata_dataframe)}")
print(f"\nMetadata sources:")
print(metadata_dataframe['metadata_source'].value_counts())
print(f"\nAverage completeness: {metadata_dataframe['metadata_completeness'].mean():.2%}")
print(f"\nOutput saved to: data/metadata.csv")
