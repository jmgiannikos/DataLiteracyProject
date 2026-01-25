import pandas as pd
import requests
import logging
import json
import os
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Set, Union
from urllib.parse import quote
from src.utils import sanitize_article_id as sanitize_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API endpoints
OPENALEX_API = "https://api.openalex.org/works/doi:{doi}"
CROSSREF_API = "https://api.crossref.org/works/{doi}"

# Rate limiting (polite pool recommendations)
OPENALEX_DELAY = 0.1  # 10 req/s
CROSSREF_DELAY = 1.0  # 1 req/s

# Empty metadata template for error cases
EMPTY_ENRICHMENT_TEMPLATE = {
    'first_author_institution': None,
    'first_author_country': None,
    'coauthor_countries': None,
    'venue': None,
    'venue_issn': None,
    'publication_year': None,
    'cited_by_count': None,
    'biblio_volume': None,
    'biblio_issue': None,
    'biblio_pages': None,
    'metadata_source': 'ArXiv-only',
    'metadata_completeness': 0.0,
    'scrape_timestamp': None
}

def ensure_cache_dirs(base_dir: str = "data/cache"):
    """Create cache directories if they don't exist."""
    Path(f"{base_dir}/arxiv").mkdir(parents=True, exist_ok=True)
    Path(f"{base_dir}/openalex").mkdir(parents=True, exist_ok=True)
    Path(f"{base_dir}/crossref").mkdir(parents=True, exist_ok=True)


def load_cached_response(doi: str, source: str, cache_dir: str = "data/cache") -> Optional[Dict]:
    """Load cached API response if it exists."""
    filename = sanitize_filename(doi)
    cache_path = Path(cache_dir) / source / f"{filename}.json"

    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache for {doi} from {source}: {e}")
    return None


def save_cached_response(doi: str, source: str, data: Dict, cache_dir: str = "data/cache"):
    """Save API response to cache."""
    filename = sanitize_filename(doi)
    cache_path = Path(cache_dir) / source / f"{filename}.json"

    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save cache for {doi} to {source}: {e}")


def query_openalex(doi: str, contact_email: Optional[str] = None) -> Optional[Dict]:
    """Query OpenAlex API for paper metadata."""
    headers = {}
    if contact_email:
        headers['User-Agent'] = f'mailto:{contact_email}'

    url = OPENALEX_API.format(doi=quote(doi, safe=''))

    try:
        time.sleep(OPENALEX_DELAY)
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            logger.debug(f"OpenAlex: DOI not found: {doi}")
            return None
        else:
            logger.warning(f"OpenAlex API error for {doi}: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"OpenAlex query failed for {doi}: {e}")
        return None


def query_crossref(doi: str, contact_email: Optional[str] = None) -> Optional[Dict]:
    """Query Crossref API for paper metadata."""
    headers = {}
    if contact_email:
        headers['User-Agent'] = f'mailto:{contact_email}'

    url = CROSSREF_API.format(doi=doi)

    try:
        time.sleep(CROSSREF_DELAY)
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data.get('message', {})
        elif response.status_code == 404:
            logger.debug(f"Crossref: DOI not found: {doi}")
            return None
        else:
            logger.warning(f"Crossref API error for {doi}: {response.status_code}")
            return None
    except Exception as e:
        logger.warning(f"Crossref query failed for {doi}: {e}")
        return None


def extract_openalex_metadata(data: Dict) -> Dict[str, Any]:
    """Extract affiliation and publication metadata from OpenAlex response."""
    result = {
        'first_author_institution': None,
        'first_author_country': None,
        'coauthor_countries': [],
        'venue': None,
        'venue_issn': None,
        'publication_year': None,
        'cited_by_count': None,
        'biblio': {},
        'metadata_source': 'OpenAlex',
        'metadata_completeness': 0.0
    }

    # Extract affiliation data
    authorships = data.get('authorships', [])
    if authorships:
        # Extract first author info
        first_author = authorships[0]
        institutions = first_author.get('institutions', [])

        if institutions:
            first_inst = institutions[0]
            result['first_author_institution'] = first_inst.get('display_name')
            result['first_author_country'] = first_inst.get('country_code')

        # Extract all co-author countries
        countries = set()
        for authorship in authorships:
            for inst in authorship.get('institutions', []):
                country = inst.get('country_code')
                if country:
                    countries.add(country)

        result['coauthor_countries'] = sorted(list(countries))

    # Extract publication venue information
    primary_location = data.get('primary_location', {})
    if primary_location:
        source = primary_location.get('source')
        if source:
            result['venue'] = source.get('display_name')
            result['venue_issn'] = source.get('issn_l')

    # Extract publication year
    result['publication_year'] = data.get('publication_year')

    # Extract citation count
    result['cited_by_count'] = data.get('cited_by_count')

    # Extract bibliographic info (volume, issue, pages)
    biblio = data.get('biblio', {})
    if biblio:
        result['biblio'] = {
            'volume': biblio.get('volume'),
            'issue': biblio.get('issue'),
            'first_page': biblio.get('first_page'),
            'last_page': biblio.get('last_page')
        }

    # Calculate completeness score
    completeness = 0.0
    if result['first_author_institution']:
        completeness += 0.25
    if result['first_author_country']:
        completeness += 0.15
    if result['coauthor_countries']:
        completeness += 0.15
    if result['venue']:
        completeness += 0.25
    if result['cited_by_count'] is not None:
        completeness += 0.1
    if result['biblio'].get('volume') or result['biblio'].get('first_page'):
        completeness += 0.1

    result['metadata_completeness'] = completeness

    return result


def extract_crossref_metadata(data: Dict) -> Dict[str, Any]:
    """Extract affiliation and publication metadata from Crossref response."""
    result = {
        'first_author_institution': None,
        'first_author_country': None,
        'coauthor_countries': [],
        'venue': None,
        'venue_issn': None,
        'publication_year': None,
        'cited_by_count': None,
        'biblio': {},
        'metadata_source': 'Crossref',
        'metadata_completeness': 0.0
    }

    # Extract affiliation data
    authors = data.get('author', [])
    if authors:
        # Extract first author info (Crossref has less structured affiliation data)
        first_author = authors[0]
        affiliations = first_author.get('affiliation', [])

        if affiliations:
            # Crossref affiliation is usually just a name string
            result['first_author_institution'] = affiliations[0].get('name')

    # Extract publication venue
    container_title = data.get('container-title', [])
    if container_title:
        result['venue'] = container_title[0]

    # Extract ISSN
    issn = data.get('ISSN', [])
    if issn:
        result['venue_issn'] = issn[0]

    # Extract publication year
    published = data.get('published', {}) or data.get('published-print', {}) or data.get('published-online', {})
    if published:
        date_parts = published.get('date-parts', [[]])
        if date_parts and date_parts[0]:
            result['publication_year'] = date_parts[0][0]

    # Extract citation count (if available)
    result['cited_by_count'] = data.get('is-referenced-by-count')

    # Extract bibliographic info
    result['biblio'] = {
        'volume': data.get('volume'),
        'issue': data.get('issue'),
        'first_page': data.get('page', '').split('-')[0] if data.get('page') else None,
        'last_page': data.get('page', '').split('-')[-1] if data.get('page') and '-' in data.get('page', '') else None
    }

    # Calculate completeness score
    completeness = 0.0
    if result['first_author_institution']:
        completeness += 0.2
    if result['venue']:
        completeness += 0.3
    if result['cited_by_count'] is not None:
        completeness += 0.2
    if result['biblio'].get('volume') or result['biblio'].get('first_page'):
        completeness += 0.2
    if result['publication_year']:
        completeness += 0.1

    result['metadata_completeness'] = completeness

    return result


def search_crossref_by_journal_ref(journal_ref: str, title: str,
                                   contact_email: Optional[str] = None) -> Optional[Dict]:
    """
    Fallback: search Crossref by bibliographic fields when DOI is missing.

    Args:
        journal_ref: Journal reference string from arXiv
        title: Paper title for verification
        contact_email: Contact email for polite API access

    Returns:
        Crossref metadata if found, None otherwise
    """
    headers = {}
    if contact_email:
        headers['User-Agent'] = f'mailto:{contact_email}'

    # Try searching Crossref by title
    try:
        time.sleep(CROSSREF_DELAY)
        # Use Crossref works query endpoint
        search_url = "https://api.crossref.org/works"
        params = {
            'query.bibliographic': f"{journal_ref} {title}",
            'rows': 1
        }

        response = requests.get(search_url, params=params, headers=headers, timeout=30)

        if response.status_code == 200:
            data = response.json()
            items = data.get('message', {}).get('items', [])
            if items:
                logger.info(f"Found Crossref match via journal_ref search")
                return items[0]

        logger.debug(f"No Crossref match found for journal_ref: {journal_ref}")
        return None

    except Exception as e:
        logger.warning(f"Crossref journal_ref search failed: {e}")
        return None


def enrich_paper_metadata(paper_data: Dict[str, Any],
                         contact_email: Optional[str] = None,
                         cache_dir: str = "data/cache") -> Dict[str, Any]:
    """
    Args:
        paper_data: Dictionary containing paper info (must have 'arxiv_id', 'title', 'doi', 'journal_ref')
        contact_email: Contact email for polite API access
        cache_dir: Base directory for caching

    Returns:
        Dictionary with additional enriched metadata fields
    """
    arxiv_id = paper_data.get('arxiv_id')
    doi = paper_data.get('doi')
    journal_ref = paper_data.get('journal_ref')
    title = paper_data.get('title')

    # Initialize result with existing data + empty enrichment fields
    result = paper_data.copy()
    
    # Add default empty enrichment fields if they don't exist
    for key, value in EMPTY_ENRICHMENT_TEMPLATE.items():
        if key not in result:
            result[key] = value

    result['scrape_timestamp'] = datetime.utcnow().isoformat()
    
    # If DOI exists: query OpenAlex/Crossref
    enrichment_metadata = None

    if doi and pd.notna(doi):  # Handle NaN/None
        # logger.info(f"{arxiv_id}: DOI present ({doi}), querying OpenAlex/Crossref")

        # Try OpenAlex first
        openalex_data = load_cached_response(doi, 'openalex', cache_dir)
        if not openalex_data:
            openalex_data = query_openalex(doi, contact_email)
            if openalex_data:
                save_cached_response(doi, 'openalex', openalex_data, cache_dir)

        if openalex_data:
            enrichment_metadata = extract_openalex_metadata(openalex_data)
            # logger.info(f"{arxiv_id}: OpenAlex metadata retrieved")

        else:
            # Fallback to Crossref
            crossref_data = load_cached_response(doi, 'crossref', cache_dir)
            if not crossref_data:
                crossref_data = query_crossref(doi, contact_email)
                if crossref_data:
                    save_cached_response(doi, 'crossref', crossref_data, cache_dir)

            if crossref_data:
                enrichment_metadata = extract_crossref_metadata(crossref_data)
                # logger.info(f"{arxiv_id}: Crossref metadata retrieved")

    # If no DOI but journal_ref exists, try searching Crossref
    elif journal_ref and pd.notna(journal_ref) and title:
        logger.info(f"{arxiv_id}: No DOI, attempting journal_ref search")
        crossref_data = search_crossref_by_journal_ref(journal_ref, title, contact_email)

        if crossref_data:
            enrichment_metadata = extract_crossref_metadata(crossref_data)
            result['metadata_source'] = 'Crossref-via-journal_ref'
            logger.info(f"{arxiv_id}: Found via journal_ref search")
        else:
            logger.debug(f"{arxiv_id}: No match via journal_ref search")

    # Apply enrichment metadata if found
    if enrichment_metadata:
        result['first_author_institution'] = enrichment_metadata.get('first_author_institution')
        result['first_author_country'] = enrichment_metadata.get('first_author_country')

        coauthor_countries = enrichment_metadata.get('coauthor_countries', [])
        result['coauthor_countries'] = json.dumps(coauthor_countries) if coauthor_countries else None

        result['venue'] = enrichment_metadata.get('venue')
        result['venue_issn'] = enrichment_metadata.get('venue_issn')
        
        # Prefer enriched publication year if available, otherwise keep original
        if enrichment_metadata.get('publication_year'):
            result['publication_year'] = enrichment_metadata.get('publication_year')
            
        result['cited_by_count'] = enrichment_metadata.get('cited_by_count')

        # Extract biblio fields
        biblio = enrichment_metadata.get('biblio', {})
        result['biblio_volume'] = biblio.get('volume')
        result['biblio_issue'] = biblio.get('issue')

        # Combine pages
        first_page = biblio.get('first_page')
        last_page = biblio.get('last_page')
        if first_page and last_page and first_page != last_page:
            result['biblio_pages'] = f"{first_page}-{last_page}"
        elif first_page:
            result['biblio_pages'] = first_page

        result['metadata_source'] = enrichment_metadata.get('metadata_source', 'Unknown')
        result['metadata_completeness'] = enrichment_metadata.get('metadata_completeness', 0.0)

        # logger.info(f"{arxiv_id}: Enriched (completeness={result['metadata_completeness']:.2f})")
    else:
        # logger.debug(f"{arxiv_id}: No additional metadata found")
        pass

    return result


def enrich_metadata_dataframe(input_df: pd.DataFrame, contact_email: Optional[str] = None,
                             cache_dir: str = "data/cache", output_path: str = "data/metadata.csv") -> pd.DataFrame:
    """
    Args:
        input_df: DataFrame containing arXiv papers (must have 'arxiv_id', 'doi', 'journal_ref', 'title')
        contact_email: Contact email for polite API access (or set CONTACT_EMAIL env var)
        cache_dir: Base directory for caching API responses
        output_path: Path for output metadata.csv

    Returns:
        DataFrame with enriched metadata
    """
    logger.info(f"Starting metadata enrichment for {len(input_df)} papers")

    # Get contact email from env if not provided
    if not contact_email:
        contact_email = os.environ.get('CONTACT_EMAIL')
        if contact_email:
            logger.info(f"Using contact email from environment: {contact_email}")

    # Ensure cache directories exist
    ensure_cache_dirs(cache_dir)

    # Convert DataFrame to list of dicts for processing
    papers_data = input_df.to_dict('records')
    enriched_records = []

    for i, paper in enumerate(papers_data, 1):
        arxiv_id = paper.get('arxiv_id', 'Unknown')
        if i % 10 == 0:
            logger.info(f"Processing {i}/{len(papers_data)}: {arxiv_id}")

        try:
            enriched = enrich_paper_metadata(
                paper_data=paper,
                contact_email=contact_email,
                cache_dir=cache_dir
            )
            enriched_records.append(enriched)
        except Exception as e:
            logger.error(f"Failed to enrich metadata for {arxiv_id}: {e}")
            # Add original record with error note
            error_record = paper.copy()
            for key, val in EMPTY_ENRICHMENT_TEMPLATE.items():
                if key not in error_record:
                    error_record[key] = val
            error_record['metadata_source'] = 'Error'
            enriched_records.append(error_record)

    # Create metadata dataframe
    metadata_df = pd.DataFrame(enriched_records)

    # Save to CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    metadata_df.to_csv(output_path, index=False)
    logger.info(f"Saved enriched metadata to {output_path}")

    # Log statistics
    if 'doi' in metadata_df.columns:
        dois_found = metadata_df['doi'].notna().sum()
        logger.info(f"DOIs present: {dois_found}/{len(metadata_df)}")

    source_counts = metadata_df['metadata_source'].value_counts()
    logger.info(f"Metadata sources: {source_counts.to_dict()}")

    avg_completeness = metadata_df['metadata_completeness'].mean()
    logger.info(f"Average metadata completeness: {avg_completeness:.2%}")

    return metadata_df


if __name__ == "__main__":
    pass
