import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

class GetGasData:
    def __init__(self):
        pass

    def load_data(self, gas_path, start_datetime=None, end_datetime=None, resolution=None) -> pd.DataFrame:

        print(f"Loading data from {gas_path}...")
        gas_data = self.concatenate(gas_path, start_datetime=start_datetime,
                                              end_datetime=end_datetime)

        if gas_data.empty:
            print("No gas data rows available after loading/filtering.")
            return gas_data

        if not isinstance(gas_data.index, pd.DatetimeIndex):
            raise TypeError("Gas data must be indexed by datetime before resampling.")

        if resolution:
            gas_data = gas_data.resample(resolution).median()

        gas_data = gas_data.dropna()
        return gas_data

    def concatenate(self, directory, start_datetime=None, end_datetime=None) -> pd.DataFrame:
        """
        Concatenate CSV files in a directory, filtering rows by datetime inside the file.

        :param directory: Path to directory of CSV files
        :param start_datetime: datetime object or string (e.g., '2025-07-01 12:00')
        :param end_datetime: datetime object or string
        :return: Filtered and concatenated DataFrame
        """
        directory = Path(directory)
        if not directory.is_dir():
            parent = directory.parent
            available = []
            if parent.is_dir():
                available = sorted(path.name for path in parent.iterdir() if path.is_dir())
            hint = f" Available directories in {parent}: {available}" if available else ""
            raise FileNotFoundError(
                f"Gas data directory does not exist: {directory}.{hint} "
                "Check config.json storage_path and the experiment/gas folder names."
            )

        data_frames = []
        files = sorted(os.listdir(directory))

        # Convert string inputs to datetime
        if isinstance(start_datetime, str):
            start_datetime = datetime.strptime(start_datetime, '%Y-%m-%d %H:%M')
        if isinstance(end_datetime, str):
            end_datetime = datetime.strptime(end_datetime, '%Y-%m-%d %H:%M')
            end_datetime = end_datetime + timedelta(hours= 2, minutes=10)

        for file in files:
            if file.endswith('.csv'):

                if (file.startswith('C') or file.startswith('n') or file.startswith('O')):
                    file_shortened = file[file.find('_') + 1:]  # removes 'P1_' → '2025-07-10 11:55:15:945758.csv'
                else:
                    file_shortened = file
                # Detect whether the datetime part starts after a space or underscore
                if ' ' in file_shortened:
                    date_str = file_shortened.split(' ')[0]
                elif '_' in file_shortened:
                    date_str = file_shortened.split('_')[0]
                else:
                    print(f"Skipping file {file}: no recognizable separator before datetime")
                    continue  # or handle differently
                file_date = datetime.strptime(date_str, '%Y-%m-%d')

                # Filter by file-level date to reduce unnecessary file reads
                if (start_datetime is None or file_date.date() >= start_datetime.date()) and \
                        (end_datetime is None or file_date.date() <= end_datetime.date()):

                    file_path = os.path.join(directory, file)
                    columns = pd.read_csv(file_path, nrows=0).columns
                    if 'datetime' in columns:
                        index_col = 'datetime'
                    elif 'timestamp' in columns:
                        index_col = 'timestamp'
                    else:
                        raise KeyError(
                            f"{file_path} must contain either a 'datetime' or 'timestamp' column."
                        )

                    df = pd.read_csv(file_path, index_col=index_col)
                    raw_index = pd.Index(df.index)
                    parsed_index = pd.to_datetime(raw_index, format='%Y-%m-%d %H:%M:%S:%f', errors='coerce')
                    needs_fallback = parsed_index.isna()
                    if needs_fallback.any():
                        fallback_index = pd.to_datetime(raw_index[needs_fallback], errors='coerce')
                        parsed_index = pd.Series(parsed_index, index=df.index)
                        parsed_index.loc[needs_fallback] = fallback_index
                        parsed_index = pd.DatetimeIndex(parsed_index)
                    df.index = parsed_index

                    invalid_rows = df.index.isna()
                    if invalid_rows.any():
                        invalid_count = int(invalid_rows.sum())
                        print(f"Skipping {invalid_count} rows with invalid timestamps in {file_path}.")
                        df = df.loc[~invalid_rows]
                    # print(df)

                    # Ensure datetime column is parsed
                    ##try:
                    #    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S:%f')
                    # except:
                    #    df['datetime'] = pd.to_datetime(df['datetime'])  # fallback

                    # Filter rows by full datetime
                    if start_datetime:
                        df = df[df.index >= start_datetime]
                    if end_datetime:
                        df = df[df.index <= end_datetime]

                    if not df.empty:
                        data_frames.append(df)


        if not data_frames:
            print("No data files found in the specified datetime range.")
            return pd.DataFrame(index=pd.DatetimeIndex([], name="datetime"))

        concatenated_df = pd.concat(data_frames).sort_index()
        return concatenated_df

    def plot_data(self, SM_data):

        fig, ax = plt.subplots(figsize=(10, 6))
        for column in SM_data.columns:
            if column != 'timestamp':
                ax.plot(SM_data.index, SM_data[column], label=column)
        str_index = SM_data.index.strftime("%Y-%m-%d %H:%M")
        ax.set(xticks=str_index[::int(len(SM_data.index) / 10)],
               xticklabels=str_index[::int(len(SM_data.index) / 10)].values)

        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Soil Moisture')
        ax.legend()
        plt.show()

    def plot_mean_of_groups(self, SM_data):

        # Get numeric columns
        numeric_cols = [col for col in SM_data.columns]
        print("Numeric columns:", numeric_cols)

        # Sort columns just in case
        numeric_cols = sorted(numeric_cols)

        # Define group size (e.g., 4)
        group_size = 4

        # Loop over column groups and compute means
        group_means = pd.DataFrame(index=SM_data.index)

        for i in range(0, len(numeric_cols), group_size):
            group = numeric_cols[i:i + group_size]
            group_label = f"CH_{group[0]}-{group[-1]}"
            group_means[group_label] = SM_data[group].mean(axis=1)

        # Plot all group means
        group_means.plot(title="Mean of Plant Groups", ylabel="Mean Moisture", xlabel="Time")
        plt.tight_layout()
        plt.show()
