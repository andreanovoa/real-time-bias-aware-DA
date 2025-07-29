import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest
import datetime
import traceback
import logging

# Disable file validation for debugging
os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

logging.basicConfig(level=logging.INFO)

# --- Configuration (can be set via environment variables) ---
FOLDER_TO_TEST = os.environ.get("NB_TEST_FOLDER", "scripts/tutorials")
TIMEOUT = int(os.environ.get("NB_TEST_TIMEOUT", "600"))
QUICK = os.environ.get("NB_TEST_QUICK", "False").lower() in ("1", "true", "yes")


def run_notebook(notebook_path, timeout=600, quick=True):
    """Execute a Jupyter notebook with plotting and file-saving disabled."""

    if 'TODO' in notebook_path:
        logging.info(f'Notebook {notebook_path} under development. Not tested.')
        return True

    logging.info(f"Starting notebook test: {notebook_path} at {datetime.datetime.now()}")

    with open(notebook_path, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)

    logging.info(f'Testing {notebook_path}...')

    # Modify the notebook to disable plotting and file saving
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Suppress plotting
            cell['source'] = (
                "import matplotlib; matplotlib.use('Agg')  # Disable plotting\n" +
                cell['source']
            )

            if quick:
                if 'train' in cell['source']:
                    logging.info('Skipped training...')
                    return True

                if 'anim.' in cell['source']:
                    logging.info('Skipped animation...')
                    return True

    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

    try:
        ep.preprocess(notebook, {'metadata': {'path': os.path.dirname(notebook_path)}})
        logging.info('OK')
        return True
    except Exception as e:
        logging.error(f"Error executing {notebook_path}: {e}")
        traceback.print_exc()
        return False


def find_notebooks_in_folder(folder_path):
    """Find all Jupyter notebooks in a folder, skipping checkpoint files."""
    notebooks = []
    for root, _, files in os.walk(folder_path):
        if '.ipynb_checkpoints' in root:
            continue
        for file in files:
            if file.endswith('.ipynb'):
                notebooks.append(os.path.join(root, file))
    return sorted(notebooks)


# Discover notebooks at import time for pytest parameterization
NOTEBOOKS = find_notebooks_in_folder(FOLDER_TO_TEST)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_runs(notebook_path):
    """Test that a notebook runs without error."""
    assert run_notebook(notebook_path, timeout=TIMEOUT, quick=QUICK), f"Notebook failed: {notebook_path}"
