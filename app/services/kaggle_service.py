"""
Kaggle Service — searches and downloads datasets from Kaggle
using the official Kaggle API.
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


def search_kaggle_datasets(problem_statement: str, max_results: int = 6) -> list:
    try:
        api = get_kaggle_api()

        stop_words = {
            'i', 'want', 'to', 'a', 'an', 'the', 'and', 'or', 'for',
            'with', 'using', 'based', 'on', 'from', 'that', 'this',
            'will', 'can', 'how', 'what', 'which', 'is', 'are', 'my',
            'we', 'our', 'build', 'create', 'make', 'develop', 'predict',
            'analysis', 'analyze', 'find', 'get', 'data'
        }

        words        = problem_statement.lower().split()
        keywords     = [w for w in words if w not in stop_words and len(w) > 2]
        search_query = ' '.join(keywords[:4])

        if not search_query:
            search_query = problem_statement[:50]

        print(f"Searching Kaggle for: {search_query}")

        datasets = api.dataset_list(search=search_query)

        # Convert to list explicitly
        datasets_list = list(datasets)
        print(f"Found {len(datasets_list)} datasets")

        results = []
        for ds in datasets_list[:max_results]:
            try:
                # Get size safely — attribute name varies by kaggle version
                size_mb = 0
                for size_attr in ['totalBytes', 'size', 'total_bytes', 'datasetTotalBytes']:
                    val = getattr(ds, size_attr, None)
                    if val:
                        size_mb = round(val / (1024 * 1024), 1)
                        break

                results.append({
                    "ref":            str(ds.ref),
                    "title":          str(ds.title),
                    "size_mb":        size_mb,
                    "download_count": int(getattr(ds, 'downloadCount', 0) or 0),
                    "vote_count":     int(getattr(ds, 'voteCount', 0) or 0),
                    "usability":      float(getattr(ds, 'usabilityRating', 0) or 0),
                    "last_updated":   str(getattr(ds, 'lastUpdated', ''))[:10],
                })
            except Exception as e:
                print(f"Error processing dataset: {e}")
                continue

        print(f"Returning {len(results)} results")
        return results

    except Exception as e:
        print(f"Kaggle search error: {e}")
        return [{"error": str(e)}]


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

        zip_path   = os.path.join(temp_dir, zip_files[0])
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

        # ── Register in local search registry ──────────────────────
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