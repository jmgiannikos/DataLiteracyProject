# DataLiteracyProject
## main.py

Will be the pipeline orchestrator. 

select authors -> select papers -> scrape metadata -> scrape text -> clean text -> extract features

Visualization will be done in a separate jupyter notebook.

## select_papers.py

Handles arXiv queries to scrape a list of papers in the form of arXiv IDs.

All methods are helpers except for `get_papers()`.

```python
"""
Args:
        entry: Entry point author name
        n: Number of co-authors to include
        j: Number of first-author papers per author
        k: Number of non-first-author papers per author
        strict: If True, require exactly n co-authors, j first-author papers, and k non-first-author papers.
                If False (default), allow up to n, j, and k respectively.

    Returns:
        Set of arXiv IDs for all deduplicated papers
"""
```

TODO:
- [ ] recursive call to build a larger set of papers
- [ ] include "at least `m` co-authors among the `n` selected should appear in `i` papers among the `j+k` selected for each author"

## scrape_metadata.py

Main method: `run()`. First, it extracts basic metadata from arXiv. If DOI present, searches via OpenAlex (highest hitrate possible). If not, tries CrossRef.
Right now, 50% of papers have full metadata representation.

```python
"""
    Takes a list of arXiv IDs, fetches DOIs from arXiv API, then enriches with
    metadata from OpenAlex/Crossref.

    Args:
        arxiv_ids: List of arXiv identifiers
        contact_email: Contact email for polite API access (or set CONTACT_EMAIL env var)
        cache_dir: Base directory for caching API responses
        output_path: Path for output metadata.csv

    Returns:
        DataFrame with enriched metadata
    """
```

## scrape_text.py

Sim

## analysis.py
currently the two major functions in this file are 'feature_analysis_pipe' and 'prediction_pipe'. Both can be called without parameters, as they are pre-initialized with reasonable values. They do the following:
### 'feature_analysis_pipe': 
1) Extract a set of features for each document in the collection: common word frequencies, mean and stdev of sentence lenght, frequency of "easy" words, syllable count frequency
2) Group the documents into groups according to the value(s) passed in group_by parameter. Remove all groups that do not have a minimum number of samples
3) compute feature wise distribution for each group via histogram. Fixed number of bins, spread evenly between global min and max value of the binned feature. IMPROVEMENT POSSIBILITY: use some other form of density estimation (e.g. Kernel density estimation)
4) compute pairwise jensen-shannon divergence for each group (typically for each author, but other groupings are possible) with each other group. Jensen shannon divergence is computed between the respective distributions calculated in the previous step.
5) use mixed integer programming to find a minimal selection of features for which the average divergence between each pair of groups is above a set value (if this is infeasible there is an option to iteratively reduce the target until it becomes feasible)
6) returns dict of data frames showing the divergences for each pair for the selected features for each column name provided in group_by (standard). There is also a crossvalidation mode, which returns nested dictionaries with the following hierarchy:
-- outermost: keys are the groupings like in standard operation
-- middle: keys are the crossval split
-- innermost: keys are "divergence_df" (result of the divergence calc), "test" (test df), "train" (train df) and "group names" (names of the retained groups after dropping groups with too few members)

### 'prediction_pipe'
1) run crossval version of feature analysis pipe to select features
2) setup predictor with fit():
-- use Kernel Density Estimation to estimate p(features|group) for each group
-- estimate p(group) to be num_samples(group)/num_samples(all)
-- use both to get joint probability distribution 
3) iterate over holdout sets and predict group with predict():
-- use previously established distributions and bayes rule to compute p(group|feat) for each group
4) collect all predictions (across all splits) in one df
5) average prediction for each group (i.e. compute average p(author|sample) across all samples that belong to the same author) and collect results in df
6) return df calculated in 5) as a measure of performance