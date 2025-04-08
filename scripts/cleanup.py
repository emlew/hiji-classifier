def enforce_file_bounds(file, df):
    """
    Combats irregular sensor behavior by ensuring that each data frame contains only data from the intended interval.
    
    Parameters:
    - file: filename to select interval
    - df: DataFrame with time-series data
    
    Returns:
    - DataFrame with bounds enforced
    """
    if file == './data/raw/W-2-28.csv':
        return df[df['isoTimestamp'].between('2025-02-28 23:24:00.000000+00:00', '2025-02-28 23:27:30.000000+00:00')]
    if file == './data/raw/2-21.6.csv':
        return df[df['isoTimestamp'].between('2025-02-21 19:50:00.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]
    if file == './data/raw/2-21.8.csv':
        return df[df['isoTimestamp'].between('2025-02-21 19:50:00.000000+00:00', '2025-02-21 19:50:30.000000+00:00')]
    if file == './data/raw/2-21.10.csv':
        return df[df['isoTimestamp'].between('2025-02-21 20:04:30.000000+00:00', '2025-02-21 20:05:00.000000+00:00')]
    if file == './data/raw/3-11.csv':
        return df[df['isoTimestamp'].between('2025-03-10 10:19:25.422000+00:00','2025-03-12 21:19:25.422000+00:00')]
    return df