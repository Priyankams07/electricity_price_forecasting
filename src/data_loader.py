# src/data_loader.py
import pandas as pd
import re
from src.config import DATA_PATH, TARGET_COLUMN

def _extract_start_time_from_range(s):
    """
    Given strings like "00:00 - 01:00" or "00:00-01:00", return "00:00".
    """
    if pd.isna(s):
        return None
    s = str(s)
    # split on '-' (with possible spaces) and take first token that looks like HH:MM
    parts = re.split(r'\s*-\s*', s)
    for p in parts:
        m = re.search(r'\d{1,2}:\d{2}', p)
        if m:
            return m.group(0)
    # fallback: search entire string
    m = re.search(r'\d{1,2}:\d{2}', s)
    return m.group(0) if m else None

def load_data():
    df = pd.read_csv(DATA_PATH, dtype=str)  # read as strings to inspect format

    # CASE A: separate 'date' and 'hour' columns
    if 'date' in df.columns and 'hour' in df.columns:
        # hour may be like "00:00 - 01:00" -> extract start time
        start_times = df['hour'].apply(_extract_start_time_from_range)
        df['timestamp'] = pd.to_datetime(df['date'].astype(str).str.strip() + ' ' + start_times)
        # drop the original columns we used
        df = df.drop(columns=['date', 'hour'])
        df = df.set_index('timestamp').sort_index()
    else:
        # CASE B: maybe the first column contains "2022/01/01 00:00 - 01:00" or similar
        first_col = df.columns[0]
        sample = df[first_col].iloc[0] if len(df) > 0 else ""
        # try pattern: date + time-range in same field
        # extract a date-like token and a time-like token (the start time)
        date_match = re.search(r'(\d{4}[/\-]\d{2}[/\-]\d{2})', str(sample))
        time_match = re.search(r'(\d{1,2}:\d{2})', str(sample))
        if date_match and time_match:
            # extract date and the first time occurrence in each row
            def parse_row_val(v):
                v = str(v)
                date_m = re.search(r'(\d{4}[/\-]\d{2}[/\-]\d{2})', v)
                time_m = re.search(r'(\d{1,2}:\d{2})', v)
                if date_m and time_m:
                    return f"{date_m.group(0)} {time_m.group(0)}"
                else:
                    return None
            df['timestamp'] = df[first_col].apply(parse_row_val)
            if df['timestamp'].isnull().any():
                # try alternative: maybe comma separated like "2022/01/01,00:00 - 01:00"
                def alt_parse(v):
                    v = str(v)
                    parts = v.split(',')
                    if len(parts) >= 2:
                        date = parts[0].strip()
                        time_part = parts[1].strip()
                        start = _extract_start_time_from_range(time_part)
                        if start:
                            return f"{date} {start}"
                    return None
                df['timestamp'] = df[first_col].apply(lambda v: df['timestamp'] if False else None)  # noop to keep structure
            # try to coerce timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().all():
                raise ValueError("Tried to parse combined date+hour column but failed. "
                                 "Open CSV and check the exact format of the first column.")
            # drop the original first column(s)
            df = df.drop(columns=[first_col])
            df = df.set_index('timestamp').sort_index()
        else:
            # CASE C: check for a dedicated 'timestamp' column
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                if df['timestamp'].isnull().any():
                    raise ValueError("Found 'timestamp' column but failed to parse its values. Check formats.")
                df = df.set_index('timestamp').sort_index()
            else:
                raise ValueError("Could not detect date/hour columns. Make sure CSV has either "
                                 "'date' and 'hour' columns or a combined first column like 'YYYY/MM/DD HH:MM - HH:MM' "
                                 "or a 'timestamp' column.")

    # now df indexed by timestamp. convert numeric price column(s)
    # Ensure target column exists (case-sensitive)
    available_cols = list(df.columns)
    if TARGET_COLUMN not in available_cols:
        raise ValueError(f"TARGET_COLUMN '{TARGET_COLUMN}' not found in CSV columns: {available_cols}")

    # convert selected column to numeric, coerce errors to NaN
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df = df[[TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "price"})
    # interpolate and fill edges
    df = df.interpolate(method='time').ffill().bfill()
    return df
