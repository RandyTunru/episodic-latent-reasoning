import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import json
import numpy as np

def read_parquet_file(file_path: str) -> pd.DataFrame:
    """Read a Parquet file and return its contents as a pandas DataFrame.

    Args:
        file_path (str): The path to the Parquet file.
        columns (list, optional): A list of column names to read. If None, all columns are read.

    Returns:
        pd.DataFrame: The contents of the Parquet file as a pandas DataFrame.
    """
    table = pq.read_table(file_path)
    return table.to_pandas()

def handle_numpy(obj):
    """Handle numpy data types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def sample_to_jsonl(out_dir, sample: dict) -> dict:
    """Convert a sample dictionary to a JSON-serializable format.

    Args:
        sample (dict): A dictionary representing a sample.

    Returns:
        dict: A JSON-serializable dictionary.
    """
    out_dir = out_dir / "samples.jsonl"

    with open(out_dir, 'a') as f:
        for row in sample.to_dict(orient='records'):
            f.write(json.dumps(row, default=handle_numpy) + '\n')

    return out_dir

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) != 2:
        print("Usage: python parquet_reader.py <parquet_file_path>")
        sys.exit(1)

    parquet_file_path = sys.argv[1]
    df = read_parquet_file(parquet_file_path)

    output = sample_to_jsonl(Path(parquet_file_path).parent, df)

    print(f"Converted Parquet file to JSONL and saved to: {output}")