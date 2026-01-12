import pandas as pd

def csv_to_json(csv_file):
    """Read an uploaded CSV (or path-like) and return a list of records.

    Returns a Python list of dicts suitable to pass to `requests.post(..., json=...)`.
    """
    df = pd.read_csv(csv_file)
    return df.to_dict(orient='records')