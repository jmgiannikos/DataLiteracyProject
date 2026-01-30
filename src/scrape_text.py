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

# Rate limiting
ARXIV_DELAY_LIMIT = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/raw_text")
METADATA_FILE = Path("data/metadata.csv")

def ensure_cache_dir():
    if not CACHE_DIR.exists():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

def download_source(arxiv_id, output_path):
    """
    Downloads the source of an arXiv paper.
    Using export.arxiv.org for automated harvesting (guidelines)
    """
    url = f"https://export.arxiv.org/e-print/{arxiv_id}"
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

try:
    from src.clean_text import TEX_PARSING_RULES_LIST
except ImportError:
    try:
        from clean_text import TEX_PARSING_RULES_LIST
    except ImportError:
        # If running from root and src is not a package, but we need strictly local
        import sys
        sys.path.append(str(Path(__file__).parent))
        from clean_text import TEX_PARSING_RULES_LIST
import re

def preprocess_tex_string(tex_string):
    for rule in TEX_PARSING_RULES_LIST:
        tex_string = rule(tex_string)
    return tex_string

def postprocess_tex_string(tex_string):
    tex_string = re.sub(r"< g r a p h i c s >", "", tex_string)
    tex_string = re.sub("\n{2,}\s*", "\n", tex_string) # collapse long chains of newlines
    tex_string = re.sub(r"\\n{2,}\s*", "\n", tex_string)
    tex_string = tex_string.lower()
    return tex_string

def extract_text_from_source(source_path):
    """Extracts text from a tar.gz source file containing LaTeX using advanced parsing rules."""
    text_content = ""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Try to open as a tar file
            if tarfile.is_tarfile(source_path):
                with tarfile.open(source_path) as tar:
                    tar.extractall(path=temp_dir)
            else:
                 # If not tar, assume it might be a single file or handle widely 
                 # For now, if it's not a tar, we skip complex extraction or treat as single tex
                 pass

            # Find all .tex files
            tex_files = list(Path(temp_dir).rglob("*.tex"))
            
            full_latex = ""
            for tex_file in tex_files:
                try:
                    with open(tex_file, 'r', encoding='utf-8', errors='replace') as f:
                        # Append with a dot to ensure sentence boundaries are respected (legacy logic)
                        full_latex += "." + f.read() 
                except Exception as e:
                    logger.warning(f"Could not read {tex_file}: {e}")

            if full_latex:
                # Preprocessing
                full_latex = preprocess_tex_string(full_latex)
                
                converter = LatexNodes2Text()
                doc_string = converter.latex_to_text(full_latex)
                
                # Postprocessing
                text_content = postprocess_tex_string(doc_string)

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
        # Sanitize arxiv_id for filename (replace slashes with underscores for old-style IDs like "astro-ph/9906233")
        safe_arxiv_id = arxiv_id.replace("/", "_")
        target_file = CACHE_DIR / f"{safe_arxiv_id}.txt"
        # Clean ID just in case (e.g. version numbers)
        # Usually we want the base ID for current version, or specific version. 
        # Metadata csv usually has version ed IDs like 1234.5678v1
        # Arxiv e-print works with versions too.
        
        # NOTE: replace call is a quick fix. Need to investigate if this causes identification issues down the line
        target_file = CACHE_DIR / f"{arxiv_id.replace('/', '_')}.txt"
        
        if target_file.exists():
            logger.info(f"[{i+1}/{len(arxiv_ids)}] Skipping {arxiv_id}, already attempted (exists).")
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
                        logger.warning(f"No text extracted for {arxiv_id}. Marking as empty.")
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write("EMPTY_TEXT")
            else:
                logger.error(f"Download failed for {arxiv_id}. Marking as failed.")
                with open(target_file, 'w', encoding='utf-8') as f:
                    f.write("FAILED_DOWNLOAD")
            
        # Rate limiting
        time.sleep(ARXIV_DELAY_LIMIT)

if __name__ == "__main__":
    main()
