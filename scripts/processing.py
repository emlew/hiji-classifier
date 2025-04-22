import glob
import numpy as np
import pandas as pd

from scripts.cleanup import enforce_file_bounds

def get_impacts():
    """
    Reads and returns impact list from impacts.txt in data folder.
    
    :return: array of DateTimeIndex objects corresponding to each impact
    """
    datetimes = np.array([])
    with open('./data/impacts.txt', 'r') as file:
        for line in file:
            line = line.strip()
            if line:  # Ensure the line is not empty
                if line.startswith('#'):
                    continue
                try:
                    dt = pd.to_datetime(line)
                    datetimes = np.append(datetimes, dt)
                except ValueError:
                    print(f"Invalid datetime format: {line}. Skipping.")

    return pd.to_datetime(datetimes)

def get_interval_binned_data(data_folder,bin_size,aggregates,use_resultant=False):
    """
    Generates processed and binned data in final_preprocessed_data.csv
    
    :param data_folder: regex string for all data to include (ex. "./data/raw/*.csv").
    :param bin_size: time interval string for width of intervals (ex. "1.5s").
    :param aggregates: array of aggregate measures used to bin the data. Can be strings or functions.
    :param use_resultant: boolean specifying whether to user the resultant acceleration or axis breakdown.
    """
    output_files = []  # Store intermediate files

    impact_times = get_impacts()

    # Process each file separately
    for i, file in enumerate(glob.glob(data_folder)):
        
        if file in ['./data/raw/back.csv','./data/raw/neck.csv','./data/raw/side.csv']:
            continue
        
        print(f"Processing {file}...")
        
        df = pd.read_csv(file, dtype={"accelX": np.float32, "accelY": np.float32, "accelZ": np.float32, 
                                "rateX": np.float32, "rateY": np.float32, "rateZ": np.float32})
        
        if use_resultant:
            df['resLA'] = np.abs(np.sqrt(df['rateX']**2+df['rateY']**2+df['rateZ']**2)) # resultant linear acceleration
            df['resAA'] = np.abs(np.sqrt(df['accelX']**2+df['accelY']**2+df['accelZ']**2)) # resultant angular acceleration
            
            df.drop(columns=['accelX','accelY','accelZ','rateX','rateY','rateZ'],inplace=True)
            
        df["isoTimestamp"] = pd.to_datetime(df["isoTimestamp"], utc=True) 
        df = enforce_file_bounds(file, df) 
        df = df.set_index("isoTimestamp").resample(bin_size).agg(aggregates)  

        # Flatten column names
        df.columns = ["_".join(col) for col in df.columns]
        df.reset_index(inplace=True)

        # Convert timestamps to numeric format for efficient lookup
        df["timestamp_num"] = df["isoTimestamp"].astype("int64")  # Nanoseconds since epoch
        impact_nums = impact_times.astype("int64")  # Convert impact times too

        # Use searchsorted to efficiently check impact range
        df["impact"] = 0  # Default to 0
        idxs = np.searchsorted(df["timestamp_num"].values, impact_nums)

        # Ensure we stay in bounds and mark impact bins
        for idx in idxs:
            if idx < len(df):
                df.at[idx, "impact"] = 1  # Mark only relevant rows

        # Drop the temporary numeric timestamp column
        df.drop(columns=["timestamp_num"], inplace=True)

        # Save each file separately
        output_file = f"data/intermediate/processed_part_{i}.csv"
        df.to_csv(output_file, index=False)
        output_files.append(output_file)

    print("Processing complete! Now merging files...")

    # Merge all processed files
    df_final = pd.concat([pd.read_csv(f) for f in output_files])
    df_final.to_csv("./final_preprocessed_data.csv", index=False)

    print("Final dataset saved as final_preprocessed_data.csv.")

def iqr(column):
    """
    Returns Interquartile Range for a column (list of values)
    
    :param column: The column to find the IQR for.
    :return: size of IQR.
    """
    return column.quantile(0.75) - column.quantile(0.25)