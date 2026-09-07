"""
Structured Telemetry & Report Exporter Module
=============================================
Exports extracted DSP metrics and modulation classification results
to standardized JSON and CSV formats for downstream SIGINT sensor handoff.
"""

from typing import Dict, Any
import json
import os
import numpy as np
import pandas as pd


def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively converts numpy data types, complex numbers, and arrays
    into JSON-serializable standard Python data structures.
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.complexfloating, complex)):
        return {"real": float(np.real(obj)), "imag": float(np.imag(obj))}
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def export_results_to_json(
    results_dict: Dict[str, Any],
    output_path: str,
    indent: int = 2
) -> str:
    """
    Exports results dictionary to an indented JSON file.

    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary of extracted parameters and metadata.
    output_path : str
        Target file path.

    Returns:
    --------
    str
        JSON string content.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sanitized = _sanitize_for_json(results_dict)
    json_str = json.dumps(sanitized, indent=indent)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json_str)

    return json_str


def export_results_to_csv(
    results_dict: Dict[str, Any],
    output_path: str
) -> pd.DataFrame:
    """
    Flattens nested results dictionary and writes to a tabular CSV file.

    Parameters:
    -----------
    results_dict : Dict[str, Any]
        Dictionary of extracted parameters.
    output_path : str
        Target CSV file path.

    Returns:
    --------
    pd.DataFrame
        Pandas DataFrame created from the flattened results.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        items: list = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, (list, tuple, np.ndarray)):
                items.append((new_key, str(v)))
            elif isinstance(v, (complex, np.complexfloating)):
                items.append((f"{new_key}_real", float(np.real(v))))
                items.append((f"{new_key}_imag", float(np.imag(v))))
            else:
                items.append((new_key, v))
        return dict(items)

    flat_dict = _flatten_dict(_sanitize_for_json(results_dict))
    df = pd.DataFrame([flat_dict])
    df.to_csv(output_path, index=False)
    return df
