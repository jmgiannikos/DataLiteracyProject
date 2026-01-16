import os
import time
import requests
import tarfile
import shutil
import logging
import pandas as pd
from pylatexenc.latex2text import LatexNodes2Text
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("src/data/cache")
METADATA_FILE = Path("src/data/metadata.csv")

def ensure_cache_dir():
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_source(arxiv_id, output_path):
    """Downloads the source of an arXiv paper."""
    url = f"https://arxiv.org/e-print/{arxiv_id}"
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download source for {arxiv_id}: {e}")
        return False

def extract_text_from_source(source_path):
    """Extracts text from a tar.gz source file containing LaTeX."""
    text_content = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Try to open as a tar file
            if tarfile.is_tarfile(source_path):
                with tarfile.open(source_path) as tar:
                    tar.extractall(path=temp_dir)
            else:
                 # Sometimes it's a single .tex file or pdf (if no source)
                 # If it's not a tar, we might check if it's a gzipped file or just a pdf
                 # For now, let's assume if it's not tar, we might just try to read it if it's tex,
                 # or if it's PDF we can't do much with latex learner.
                 # Let's check headers or extension? Arxiv usually sends a tar.gz or a pdf.
                 # If it is a PDF we skip for now as we want latex.
                 pass

            # Find all .tex files
            tex_files = list(Path(temp_dir).rglob("*.tex"))
            
            # Simple heuristic: concatenate all tex files, main file usually includes others.
            # But order matters. Often the largest file is the main one.
            # Or we can just convert all of them and append.
            
            full_latex = ""
            for tex_file in tex_files:
                try:
                    with open(tex_file, 'r', encoding='utf-8', errors='replace') as f:
                        full_latex += f.read() + "\n"
                except Exception as e:
                    logger.warning(f"Could not read {tex_file}: {e}")

            if full_latex:
                converter = LatexNodes2Text()
                text_content = converter.latex_to_text(full_latex)

        except tarfile.ReadError:
            logger.error(f"File {source_path} is not a valid tar file.")
        except Exception as e:
            logger.error(f"Error extracting text from {source_path}: {e}")
            
    return text_content

def main():
    ensure_cache_dir()
    
    if not METADATA_FILE.exists():
        logger.error(f"Metadata file not found at {METADATA_FILE}")
        return

    df = pd.read_csv(METADATA_FILE)
    
    if 'arxiv_id' not in df.columns:
        logger.error("Column 'arxiv_id' not found in metadata.csv")
        return

    arxiv_ids = df['arxiv_id'].astype(str).tolist()
    
    logger.info(f"Found {len(arxiv_ids)} papers to process.")

    for i, arxiv_id in enumerate(arxiv_ids):
        # Clean ID just in case (e.g. version numbers)
        # Usually we want the base ID for current version, or specific version. 
        # Metadata csv usually has version ed IDs like 1234.5678v1
        # Arxiv e-print works with versions too.
        
        target_file = CACHE_DIR / f"{arxiv_id}.txt"
        
        if target_file.exists():
            logger.info(f"[{i+1}/{len(arxiv_ids)}] Skipping {arxiv_id}, already exists.")
            continue

        logger.info(f"[{i+1}/{len(arxiv_ids)}] Processing {arxiv_id}...")
        
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=True) as temp_source:
            if download_source(arxiv_id, temp_source.name):
                # Check if it looks like a PDF (magic number)
                with open(temp_source.name, 'rb') as f:
                    header = f.read(4)
                
                if header == b'%PDF':
                    logger.warning(f"Source for {arxiv_id} appears to be a PDF. Skipping text extraction.")
                    # Create empty file or marker to avoid redownloading?
                    # Maybe write "PDF_ONLY"
                    with open(target_file, 'w', encoding='utf-8') as f:
                        f.write("PDF_ONLY")
                else:
                    text = extract_text_from_source(temp_source.name)
                    if text:
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                        logger.info(f"Successfully saved text for {arxiv_id}")
                    else:
                        logger.warning(f"No text extracted for {arxiv_id}")
            
        # Rate limiting
        time.sleep(3) # Wait 3 seconds between requests

if __name__ == "__main__":
    main()
