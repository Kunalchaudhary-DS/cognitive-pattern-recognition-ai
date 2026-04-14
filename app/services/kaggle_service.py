"""
Kaggle Service — searches and downloads datasets from Kaggle
using the official Kaggle API.

Two-stage dataset ranking:
  Stage 1 — Broad Retrieval: Fetch ≤50 results, apply lightweight
             priority scoring, discard score<0, keep top 15.
  Stage 2 — Deep Ranking: Compute 0-100 relevance score per candidate
             (title overlap, popularity, usability, recency, size),
             return top 6. Tie-broken by vote_count then usability.
"""

import os
import zipfile
import shutil
import kaggle
from app.core.config import DATASET_FOLDER
from app.services.dataset_finder_service import register_dataset


def get_kaggle_api():
    """Initialize and authenticate Kaggle API."""
    kaggle.api.authenticate()
    return kaggle.api


#Robust attribute reader (handles different Kaggle SDK versions) 
def _read(obj, *names, cast=None, default=0):
    """
    Try to read an attribute from a Kaggle API object, trying multiple
    possible names. Falls back to inspecting __dict__ if getattr fails.
    Handles Kaggle SDK v1.5.x and v1.6.x naming differences.
    """
    # First pass: try getattr with common names
    for name in names:
        val = getattr(obj, name, None)
        if val is not None and val != '' and val != 0:
            try:
                return cast(val) if cast else val
            except Exception:
                continue

    # Second pass: inspect __dict__ (works for dataclass-style objects)
    try:
        d = vars(obj)
        for name in names:
            val = d.get(name)
            if val is not None and val != '' and val != 0:
                try:
                    return cast(val) if cast else val
                except Exception:
                    continue
    except TypeError:
        pass

    # Third pass: try snake_case variants of camelCase names
    def to_snake(s):
        import re
        return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()

    for name in names:
        snake = to_snake(name)
        if snake != name:
            val = getattr(obj, snake, None)
            if val is not None and val != '' and val != 0:
                try:
                    return cast(val) if cast else val
                except Exception:
                    continue

    return default


def _read_size_mb(obj) -> float:
    """Extract dataset size in MB, trying all known attribute names and formats."""
    for attr in ['totalBytes', 'total_bytes', 'datasetTotalBytes', 'dataset_total_bytes']:
        val = getattr(obj, attr, None)
        if val:
            try:
                return round(float(val) / (1024 * 1024), 2)
            except Exception:
                continue

    # Some SDK versions return a human-readable 'size' string like "15 KB" or "2.3 MB"
    for attr in ['size', 'fileSize', 'file_size']:
        val = getattr(obj, attr, None)
        if not val:
            continue
        val_str = str(val).strip().upper()
        try:
            if 'GB' in val_str:
                return round(float(val_str.replace('GB', '').strip()) * 1024, 2)
            elif 'MB' in val_str:
                return round(float(val_str.replace('MB', '').strip()), 2)
            elif 'KB' in val_str:
                return round(float(val_str.replace('KB', '').strip()) / 1024, 2)
            else:
                # Plain number in bytes
                return round(float(val_str) / (1024 * 1024), 2)
        except Exception:
            continue

    return 0.0


def _debug_attrs(obj):
    """Print all non-None attributes of the Kaggle dataset object for debugging."""
    try:
        d = vars(obj)
        relevant = {k: v for k, v in d.items() if v is not None and v != 0 and v != ''}
        print(f"[Kaggle DEBUG] Dataset object attrs: {list(relevant.keys())}")
        print(f"[Kaggle DEBUG] Sample values: { {k: relevant[k] for k in list(relevant.keys())[:10]} }")
    except Exception as e:
        print(f"[Kaggle DEBUG] Could not inspect object: {e}")
        print(f"[Kaggle DEBUG] dir(): {[x for x in dir(obj) if not x.startswith('_')]}")


# Stage 1: Lightweight priority filter 
def _stage1_priority_score(ds_raw: object, problem_keywords: set) -> float:
    """
    Computes a rough priority score for Stage 1 filtering.
    Datasets with score < 0 are discarded entirely.
    """
    score = 0.0

    # Bonus: title contains any of the key problem words
    title_lower = str(_read(ds_raw, 'title', cast=str, default='')).lower()
    for kw in problem_keywords:
        if kw in title_lower:
            score += 10
            break

    # Bonus: high usability rating
    usability = _read(ds_raw, 'usabilityRating', 'usability_rating', 'usability', cast=float, default=0.0)
    if usability >= 0.8:
        score += 5

    # Penalty: too large to download quickly (>500 MB)
    size_mb = _read_size_mb(ds_raw)
    if size_mb > 500:
        score -= 10

    # Penalty: too few downloads
    downloads = _read(ds_raw, 'downloadCount', 'download_count', 'totalDownloads',
                      'total_downloads', cast=int, default=0)
    if downloads < 100:
        score -= 5

    return score


# Stage 2: Deep relevance scoring

def _stage2_relevance_score(problem_words: set, ds: dict) -> int:
    """
    Computes a 0-100 relevance score for a dataset dict.

    Breakdown:
      Title overlap   → max 40 pts
      Download count  → max 20 pts
      Usability       → max 20 pts
      Recency         → max 10 pts
      Size penalty    → up to -10 pts
    """
    from datetime import datetime, timezone
    score = 0

    # Title relevance (max 40 pts)
    stop_chars = set('.,!?;:()')
    title_words = set(
        ''.join(c for c in w if c not in stop_chars)
        for w in ds['title'].lower().split()
        if len(w) > 2
    )
    overlap = len(problem_words & title_words)
    score  += min(overlap * 15, 40)

    #Download popularity (max 20 pts)
    dl = ds.get('download_count', 0) or 0
    if   dl > 100_000: score += 20
    elif dl >  10_000: score += 15
    elif dl >   1_000: score += 10
    elif dl >     100: score +=  5
    # Give a small bonus even for very low downloads so not everything stays flat at 0
    elif dl >       0: score +=  2

    # Usability rating (max 20 pts)
    usability = ds.get('usability', 0) or 0
    score += int(usability * 20)

    #Vote count tiebreaker bonus (max 5 pts)
    votes = ds.get('vote_count', 0) or 0
    if   votes > 500: score += 5
    elif votes > 100: score += 3
    elif votes >  10: score += 1

    # Recency bonus (max 10 pts)
    try:
        updated_str = ds.get('last_updated', '')
        if updated_str and updated_str not in ('None', '', 'unknown'):
            updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            months_old = (datetime.now(timezone.utc) - updated).days / 30
            if   months_old <=  6: score += 10
            elif months_old <= 12: score +=  5
            elif months_old <= 24: score +=  2
    except Exception:
        pass

    # Size penalty
    size = ds.get('size_mb', 0) or 0
    if   size > 100: score -= 10
    elif size >  50: score -=  5

    return max(0, min(score, 100))


# AI-style overview generator (human-readable, no jargon)

def _generate_overview(title: str, description: str, size_mb: float,
                       downloads: int, usability: float) -> str:
    """
    Build exactly 3 human-readable sentences about a dataset so a user
    can decide whether it fits their problem — no jargon, no raw data.
    """
    parts = []

    # Sentence 1: What is this dataset about
    if description and description not in ('No description available.', 'None', ''):
        raw_sents = [s.strip() for s in description.replace('\n', ' ').split('.') if len(s.strip()) > 10]
        intro = '. '.join(raw_sents[:2])
        if intro and not intro.endswith('.'):
            intro += '.'
        parts.append(intro)
    else:
        parts.append(f"This dataset is focused on {title.lower().rstrip('.')}.")

    # Sentence 2: Popularity / reliability
    if downloads > 10_000:
        parts.append(f"It is very popular with over {downloads:,} downloads, indicating it is well-tested and reliable.")
    elif downloads > 1_000:
        parts.append(f"It has {downloads:,} downloads, making it a reasonably trusted community resource.")
    elif downloads > 0:
        parts.append(f"It is a newer or niche dataset with {downloads} downloads.")
    else:
        if usability > 0:
            u_pct = round(usability * 100)
            parts.append(f"Kaggle rates its usability at {u_pct}%, meaning it is reasonably well-documented and structured.")
        else:
            parts.append("This is a community-contributed dataset — review the column names after downloading.")

    # Sentence 3: Size warning (honest advice for the user)
    if size_mb <= 0:
        parts.append("File size is unknown — it should still load fine inside the AutoML pipeline.")
    elif size_mb < 1:
        parts.append(f"The file is tiny ({size_mb} MB), so it will download and load almost instantly.")
    elif size_mb <= 2:
        parts.append(f"At {size_mb} MB it is compact and will process quickly inside the AutoML pipeline.")
    elif size_mb <= 10:
        parts.append(f"At {size_mb} MB it is a medium-sized file — training will take a couple of minutes.")
    elif size_mb <= 50:
        parts.append(f"This dataset is {size_mb} MB which is quite large; the AutoML pipeline will consume more time and may use its sampling strategy.")
    else:
        parts.append(f"Warning: this dataset is {size_mb} MB — it is very large and the AutoML pipeline will take significant time to train.")

    return ' '.join(parts)


# Main search function
def search_kaggle_datasets(problem_statement: str, max_results: int = 6) -> list:
    """
    Two-stage Kaggle dataset search and ranking engine.
    """
    try:
        api = get_kaggle_api()

        stop_words = {
            'i', 'want', 'to', 'a', 'an', 'the', 'and', 'or', 'for',
            'with', 'using', 'based', 'on', 'from', 'that', 'this',
            'will', 'can', 'how', 'what', 'which', 'is', 'are', 'my',
            'we', 'our', 'build', 'create', 'make', 'develop', 'predict',
            'analysis', 'analyze', 'find', 'get', 'data', 'dataset',
            'about', 'its', 'their', 'there', 'use', 'used', 'have',
            'has', 'been', 'by', 'of', 'in', 'at', 'so', 'if', 'it',
        }

        raw_words     = problem_statement.lower().split()
        problem_words = set(
            w.strip('.,!?;:') for w in raw_words
            if w not in stop_words and len(w) > 2
        )
        keyword_list  = list(problem_words)[:5]
        search_query  = ' '.join(keyword_list) if keyword_list else problem_statement[:50]

        print(f"[Kaggle Stage 1] Searching: '{search_query}'")

        # Stage 1: Broad retrieval
        raw_datasets = list(api.dataset_list(search=search_query, max_size=None, file_type='csv'))
        print(f"[Kaggle Stage 1] Retrieved {len(raw_datasets)} raw datasets")

        # Debug the first one to see actual field names
        if raw_datasets:
            _debug_attrs(raw_datasets[0])

        stage1_candidates = []
        for ds_raw in raw_datasets[:50]:
            s1_score = _stage1_priority_score(ds_raw, problem_words)
            if s1_score < 0:
                continue

            # Extract all fields robustly
            size_mb   = _read_size_mb(ds_raw)
            downloads = _read(ds_raw, 'downloadCount', 'download_count',
                              'totalDownloads', 'total_downloads', cast=int, default=0)
            votes     = _read(ds_raw, 'voteCount', 'vote_count', cast=int, default=0)
            usability = _read(ds_raw, 'usabilityRating', 'usability_rating',
                              'usability', cast=float, default=0.0)
            last_upd  = str(_read(ds_raw, 'lastUpdated', 'last_updated',
                                  cast=str, default='') or '')[:10]

            # Description
            description = ''
            for desc_attr in ['description', 'subtitle', 'overview']:
                raw_desc = getattr(ds_raw, desc_attr, '') or ''
                d_str = str(raw_desc).strip()
                if d_str and d_str not in ('None', ''):
                    description = d_str
                    break

            # Tags — safely extract just the 'name' field from each tag dict/object
            raw_tags = getattr(ds_raw, 'tags', []) or []
            tags = []
            for t in raw_tags:
                if isinstance(t, dict):
                    tags.append(t.get('name') or t.get('ref') or str(t))
                elif hasattr(t, 'name'):
                    tags.append(str(t.name))
                else:
                    tags.append(str(t))

            title = str(_read(ds_raw, 'title', cast=str, default='Unknown'))
            ref   = str(_read(ds_raw, 'ref',   cast=str, default=''))

            # Generate human-readable overview (no tags sentence)
            overview = _generate_overview(title, description, size_mb, downloads, usability)

            stage1_candidates.append({
                "_s1_score":      s1_score,
                "ref":            ref,
                "title":          title,
                "description":    overview,   # <-- now the AI-generated overview
                "size_mb":        size_mb,
                "download_count": downloads,
                "vote_count":     votes,
                "usability":      usability,
                "last_updated":   last_upd,
                "tags":           tags,
            })

        # Sort by Stage 1 score, keep top 15
        stage1_candidates.sort(key=lambda d: d["_s1_score"], reverse=True)
        stage1_candidates = stage1_candidates[:15]
        print(f"[Kaggle Stage 1] {len(stage1_candidates)} candidates survive filter")

        #Stage 2: Deep relevance scoring
        for ds in stage1_candidates:
            ds["relevance_score"] = _stage2_relevance_score(problem_words, ds)

        # Sort by score, then tiebreak by votes then usability
        stage1_candidates.sort(
            key=lambda d: (d["relevance_score"], d.get("vote_count", 0), d.get("usability", 0)),
            reverse=True
        )
        top_results = stage1_candidates[:max_results]

        for ds in top_results:
            ds.pop("_s1_score", None)

        print(f"[Kaggle Stage 2] Returning top {len(top_results)} datasets "
              f"(scores: {[d['relevance_score'] for d in top_results]})")
        return top_results

    except Exception as e:
        print(f"Kaggle search error: {e}")
        return [{"error": str(e)}]


# Download function
def download_kaggle_dataset(dataset_ref: str, dataset_title: str) -> dict:
    """
    Downloads a Kaggle dataset and extracts CSV files to datasets/ folder.
    """
    try:
        api = get_kaggle_api()

        temp_dir = os.path.join(DATASET_FOLDER, "_kaggle_temp")
        os.makedirs(temp_dir, exist_ok=True)

        api.dataset_download_files(
            dataset_ref,
            path=temp_dir,
            unzip=False,
            quiet=False
        )

        zip_files = [f for f in os.listdir(temp_dir) if f.endswith('.zip')]
        if not zip_files:
            return {"error": "No zip file found after download"}

        zip_path    = os.path.join(temp_dir, zip_files[0])
        extract_dir = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)

        csv_files = []
        for root, dirs, files in os.walk(extract_dir):
            for f in files:
                if f.endswith('.csv'):
                    csv_files.append(os.path.join(root, f))

        if not csv_files:
            return {"error": "No CSV files found in dataset"}

        downloaded = []
        for csv_path in csv_files:
            filename    = os.path.basename(csv_path)
            destination = os.path.join(DATASET_FOLDER, filename)
            shutil.copy2(csv_path, destination)
            downloaded.append(filename)

        shutil.rmtree(temp_dir, ignore_errors=True)

        register_dataset(
            filenames=downloaded,
            kaggle_ref=dataset_ref,
            kaggle_title=dataset_title,
            category="Kaggle"
        )

        return {
            "success": True,
            "files":   downloaded,
            "title":   dataset_title,
            "ref":     dataset_ref,
            "message": f"Downloaded {len(downloaded)} CSV file(s) successfully"
        }

    except Exception as e:
        temp_dir = os.path.join(DATASET_FOLDER, "_kaggle_temp")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return {"error": str(e)}