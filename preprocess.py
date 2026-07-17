import json
from pathlib import Path
from Get_Gas_Data import GetGasData
from DataProcessing.PhytoNode.PhytoNode import PhytoNode
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import multiprocessing as mp
from tsfresh import extract_features
matplotlib.use('Qt5Agg')
from tsfresh import extract_features
from tsfresh.utilities.distribution import MultiprocessingDistributor
from tsfresh.feature_extraction import MinimalFCParameters




class Preprocess:
    def __init__(self, experiments_file=None):
        self.experiments_file = experiments_file or Path(__file__).with_name("gas_experiments.json")
        self.base_dir = Path(__file__).resolve().parent
        self.PNs = ['P1','P3']
        self.gases = ["CO2", "N2", "O3"]
        self.read_experiment_config()
        with open(self.base_dir / "config.json", "r") as file:
            self.config = json.load(file)
        self.config_paths = self.config['paths']

    def resolve_config_path(self, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.base_dir / path


    def read_experiment_config(self):


        with open(self.experiments_file, "r") as file:
            self.experiment_config = json.load(file)

        #self.datetime_format = self.experiment_config.get("datetime_format")

        self.experiment_start_times = self.experiment_config['experiment_time']
        self.CO2_experiment_info = self.experiment_config['CO2']
        self.N2_experiment_info = self.experiment_config['N2']
        self.O3_experiment_info = self.experiment_config['O3']

        print(self.CO2_experiment_info)
        print(self.N2_experiment_info)
        print(self.O3_experiment_info)

    def get_experiment_datetime(self, experiment, field_name):
        value = experiment.get(field_name)
        if value is None:
            bad_key = f"{field_name[:6]} zdatetime"
            if bad_key in experiment:
                raise KeyError(
                    f"{experiment['name']} uses invalid config key {bad_key!r}; "
                    f"rename it to {field_name!r} in {self.experiments_file}."
                )
            raise KeyError(
                f"{experiment['name']} is missing required config key {field_name!r} "
                f"in {self.experiments_file}."
            )
        if value == "None":
            return None
        return value

    def compress_data(self, path, gas = False, EDP = False, start_datetime=None, end_datetime=None, resolution=None):
        """
        Load gas experiment data from CSV files in gas_path directory,
        filtering by full datetime range (not just by filename).

        :param gas_path: Path to folder containing CSVs
        :param start_datetime: Start datetime (str or datetime object)
        :param end_datetime: End datetime (str or datetime object)
        :param resolution: Resampling frequency (e.g., '10min' for 10 minutes, '1H' for 1 hour, etc.)
        :return: Filtered DataFrame
        """
        if gas:
            GGD = GetGasData()
            data = GGD.load_data(path, start_datetime=start_datetime, end_datetime=end_datetime, resolution=resolution)
        #GGD.plot_data(gas_data)
        if EDP:
            # One DataFrame per node (P1, P3, ...); keyed by node id so each
            # node can be written to its own folder downstream.
            data = {}
            for node in self.PNs:
                print(f"Compressing node {node} ")
                PN = PhytoNode(node, path / node)

                data[node] = PN.load_data(PN.directory, start_datetime=start_datetime, end_datetime=end_datetime,
                                         resolution=resolution, inVolt=True, save=False)

        return data



    def get_experiment_time_range(self, gas, experiment):
        """
        Return (start_datetime, end_datetime) spanning one experiment's
        applications, so window CSVs (which only carry a window_start
        timestamp, not the experiment they came from) can be attributed
        back to the experiment that produced them.

        CO2/N2 experiments carry an explicit start/end datetime. O3
        applications are event-triggered (start/end are "None" in the
        config), so the range is derived from that experiment's times.csv
        instead - padded by the same -1h/+2h10m offsets used when the
        windows were extracted (see get_window_starts/read_raw_experiment_data).
        """
        start_value = experiment.get('start_datetime')
        end_value = experiment.get('end_datetime')
        if start_value and start_value != "None":
            start_dt = datetime.strptime(start_value.strip(), "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(end_value.strip(), "%Y-%m-%d %H:%M")
            return start_dt, end_dt

        storage_path = self.resolve_config_path(self.config_paths['storage_path'])
        times_file = Path(storage_path, gas, experiment['name'], "times.csv")
        times = pd.to_datetime(pd.read_csv(times_file)['times'])
        start_dt = times.min() - timedelta(hours=1)
        end_dt = times.max() - timedelta(hours=1) + timedelta(hours=2, minutes=10)
        return start_dt, end_dt

    def get_window_starts(self, gas, experiment_dir, gas_data=None, potential_data=None):
        """
        Determine the application-window start times for one experiment.

        O3 applications are event-triggered at irregular times, recorded per
        application in <experiment_dir>/times.csv (one hour is subtracted so
        the window starts before the trigger). CO2/N2 applications instead
        follow the fixed daily schedule in experiment_time, applied to every
        date present in the loaded gas/potential data.
        """
        if gas == "O3":
            times_df = pd.read_csv(experiment_dir / "times.csv")
            return [
                datetime.strptime(row["times"], "%Y-%m-%d %H:%M:%S") - timedelta(hours=1)
                for _, row in times_df.iterrows()
            ]

        dates = set()
        if gas_data is not None:
            dates.update(gas_data.index.normalize().unique())
        if potential_data is not None:
            for node_data in potential_data.values():
                if not node_data.empty:
                    dates.update(node_data.index.normalize().unique())

        return [
            datetime.combine(date.date(), datetime.strptime(exp_start_time, "%H:%M").time())
            for date in sorted(dates)
            for exp_start_time in self.experiment_start_times.values()
        ]

    def read_raw_experiment_data(self, split_gases=None, split_potential=None):

        window_duration = timedelta(hours=2, minutes=10)

        for gas in self.gases:
            for experiment in self.experiment_config[gas]:
                storage_path = self.resolve_config_path(self.config_paths['storage_path'])
                time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
                start_datetime = self.get_experiment_datetime(experiment, "start_datetime")
                end_datetime = self.get_experiment_datetime(experiment, "end_datetime")
                experiment_dir = Path(storage_path, gas, experiment['name'])

                gas_data = None
                potential_data = None

                if split_gases:
                    directory_gas = experiment_dir / gas  # note change back to gasftime
                    gas_data = self.compress_data(directory_gas, split_gases, split_potential, start_datetime=start_datetime, end_datetime=end_datetime, resolution=None)
                    gas_output_dir = time_window_path / gas / "gasdata"
                    gas_output_dir.mkdir(parents=True, exist_ok=True)
                    print(gas_data)
                if split_potential:
                    directory_EP = experiment_dir / 'PN'
                    potential_data = self.compress_data(directory_EP, split_gases, split_potential, start_datetime=start_datetime, end_datetime=end_datetime, resolution="1s")
                    potential_output_dir = time_window_path / gas / "EDP"
                    # One sub-folder per node (EDP/P1, EDP/P3, ...).
                    for node in potential_data:
                        (potential_output_dir / node).mkdir(parents=True, exist_ok=True)
                    print(potential_data)

                window_starts = self.get_window_starts(gas, experiment_dir, gas_data, potential_data)

                for window_start in window_starts:
                    window_end = window_start + window_duration

                    if split_gases:
                        gas_window = gas_data[
                            (gas_data.index >= window_start) &
                            (gas_data.index < window_end)
                            ]

                        if not gas_window.empty:
                            gas_window.to_csv(gas_output_dir / f"{gas}_{window_start}.csv")

                    if split_potential:
                        # Split each node's potential trace into the same
                        # application windows and write them to EDP/<node>/.
                        for node, node_data in potential_data.items():
                            if node_data.empty:
                                continue
                            node_output_dir = potential_output_dir / node
                            node_window = node_data[
                                (node_data.index >= window_start) &
                                (node_data.index < window_end)
                                ]

                            if not node_window.empty:
                                node_window.to_csv(node_output_dir / f"{node}_{window_start}.csv")
    def plot_gas_data(self, show=True, save=True):
        """
        Read the per-window CSVs written under time_window_path (00_time_windows)
        and plot them with matplotlib. For each gas, every application window is
        overlaid on a shared x-axis of minutes elapsed from the window start, so
        responses across applications can be compared. One figure per gas is
        produced (saved under figures_path when save=True).
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        # columns that describe the control state, not a measured concentration
        state_columns = {'valve_state', 'plugstate', 'goal_concentration'}

        for gas in self.gases:
            gas_dir = time_window_path / gas
            if not gas_dir.is_dir():
                print(f"No time-window directory for {gas}: {gas_dir}")
                continue

            window_files = sorted(gas_dir.glob(f"{gas}_*.csv"))
            if not window_files:
                print(f"No window CSVs found in {gas_dir}")
                continue

            windows = []
            measurement_columns = []
            for file in window_files:
                columns = pd.read_csv(file, nrows=0).columns
                index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                df = pd.read_csv(file, index_col=index_col)
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df[~df.index.isna()]
                if df.empty:
                    continue
                # minutes elapsed from the start of the window
                df['elapsed_min'] = (df.index - df.index[0]).total_seconds() / 60.0
                windows.append(df)
                for col in df.columns:
                    if col not in state_columns and col != 'elapsed_min' \
                            and col not in measurement_columns:
                        measurement_columns.append(col)

            if not windows or not measurement_columns:
                print(f"No plottable rows for {gas}")
                continue

            fig, axes = plt.subplots(
                len(measurement_columns), 1,
                figsize=(12, 3 * len(measurement_columns)),
                sharex=True, squeeze=False,
            )
            for ax, col in zip(axes[:, 0], measurement_columns):
                for df in windows:
                    if col in df.columns:
                        ax.plot(df['elapsed_min'], df[col], linewidth=0.8, alpha=0.5)
                ax.set_ylabel(col)
                ax.grid(True, alpha=0.3)
            axes[-1, 0].set_xlabel("Minutes from window start")
            fig.suptitle(f"{gas} — {len(windows)} application windows")
            fig.tight_layout()

            if save:
                out = figures_path / f"{gas}_time_windows.png"
                fig.savefig(out, dpi=150)
                print(f"Saved {out}")

        if show:
            plt.show()
    def plot_EDP_data(self, show=True, save=True):
        """
        Read the per-window EDP CSVs written under time_window_path/<gas>/EDP/<node>
        and plot them with matplotlib. One figure per gas, with one subplot per
        node (P1, P3, ...), each overlaying every application window on a shared
        x-axis of minutes elapsed from the window start. CH2 is ignored in every
        CSV. Each window is robust-z-score normalized (median / MAD) before
        plotting so traces with different baselines/amplitudes are comparable
        without being distorted by outliers.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        ignored_columns = {'CH2'}

        for gas in self.gases:
            edp_dir = time_window_path / gas / "EDP"
            if not edp_dir.is_dir():
                print(f"No EDP directory for {gas}: {edp_dir}")
                continue

            fig, axes = plt.subplots(
                len(self.PNs), 1,
                figsize=(12, 3 * len(self.PNs)),
                sharex=True, squeeze=False,
            )

            any_plotted = False
            for ax, node in zip(axes[:, 0], self.PNs):
                node_dir = edp_dir / node
                window_files = sorted(node_dir.glob(f"{node}_*.csv")) if node_dir.is_dir() else []
                if not window_files:
                    print(f"No EDP window CSVs found in {node_dir}")
                    continue

                for file in window_files:
                    columns = pd.read_csv(file, nrows=0).columns
                    index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                    df = pd.read_csv(file, index_col=index_col)
                    df = df.drop(columns=[c for c in ignored_columns if c in df.columns])
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[~df.index.isna()]
                    if df.empty:
                        continue
                    # minutes elapsed from the start of the window
                    elapsed_min = (df.index - df.index[0]).total_seconds() / 60.0
                    for col in df.columns:
                        median = df[col].median()
                        mad = (df[col] - median).abs().median()
                        if mad == 0:
                            continue
                        # 1.4826 scales MAD to be consistent with the standard
                        # deviation under a normal distribution.
                        # normalized = (df[col] - median) / (1.4826 * mad)  # robust z-score
                        normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())  # min-max
                        ax.plot(elapsed_min, normalized, linewidth=0.8, alpha=0.5)
                    any_plotted = True

                ax.set_title(node)
                ax.set_ylabel("electrical potential (robust z-score)")
                ax.grid(True, alpha=0.3)

            if not any_plotted:
                print(f"No plottable EDP rows for {gas}")
                plt.close(fig)
                continue

            axes[-1, 0].set_xlabel("Minutes from window start")
            fig.suptitle(f"{gas} EDP — application windows")
            fig.tight_layout()

            if save:
                out = figures_path / f"{gas}_EDP_time_windows.png"
                fig.savefig(out, dpi=150)
                print(f"Saved {out}")

        if show:
            plt.show()

    def plot_EDP_mean_std(self, show=True, save=True):
        """
        Like plot_EDP_data, but instead of overlaying every raw window trace,
        aligns all windows for each node onto a common per-second time grid
        (0 .. window_duration) and plots the mean trace with +/-1 standard
        deviation shaded as a band around it. Values are robust-z-score
        normalized (median / MAD) per window before aggregating, and CH2 is
        ignored in every CSV.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        ignored_columns = {'CH2'}
        window_duration = timedelta(hours=2, minutes=10)
        grid_seconds = pd.RangeIndex(0, int(window_duration.total_seconds()) + 1)
        elapsed_min = grid_seconds / 60.0

        for gas in self.gases:
            edp_dir = time_window_path / gas / "EDP"
            if not edp_dir.is_dir():
                print(f"No EDP directory for {gas}: {edp_dir}")
                continue

            fig, axes = plt.subplots(
                len(self.PNs), 1,
                figsize=(12, 3 * len(self.PNs)),
                sharex=True, squeeze=False,
            )

            any_plotted = False
            for ax, node in zip(axes[:, 0], self.PNs):
                node_dir = edp_dir / node
                window_files = sorted(node_dir.glob(f"{node}_*.csv")) if node_dir.is_dir() else []
                if not window_files:
                    print(f"No EDP window CSVs found in {node_dir}")
                    continue

                aligned_traces = []
                for file in window_files:
                    columns = pd.read_csv(file, nrows=0).columns
                    index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                    df = pd.read_csv(file, index_col=index_col)
                    df = df.drop(columns=[c for c in ignored_columns if c in df.columns])
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[~df.index.isna()]
                    if df.empty:
                        continue

                    elapsed_seconds = ((df.index - df.index[0]).total_seconds()).round().astype(int)
                    for col in df.columns:
                        median = df[col].median()
                        mad = (df[col] - median).abs().median()
                        if mad == 0:
                            continue
                        normalized = (df[col] - median) / (1.4826 * mad)  # robust z-score
                        #normalized = (df[col] - df[col].min()) / (df[col].max() - df[col].min())  # min-max
                        trace = pd.Series(normalized.values, index=elapsed_seconds)
                        # keep the first sample of any duplicated second so it
                        # can be aligned onto the common per-second grid
                        trace = trace[~trace.index.duplicated(keep='first')]
                        aligned_traces.append(trace.reindex(grid_seconds))

                if not aligned_traces:
                    continue

                stacked = pd.concat(aligned_traces, axis=1)
                mean_trace = stacked.mean(axis=1, skipna=True)
                std_trace = stacked.std(axis=1, skipna=True)

                ax.plot(elapsed_min, mean_trace, linewidth=1.2, color='C0', label='mean')
                ax.fill_between(
                    elapsed_min, mean_trace - std_trace, mean_trace + std_trace,
                    alpha=0.3, color='C0', label='±1 std',
                )
                ax.set_title(node)
                ax.set_ylabel("electrical potential (robust z-score)")
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                any_plotted = True

            if not any_plotted:
                print(f"No plottable EDP rows for {gas}")
                plt.close(fig)
                continue

            axes[-1, 0].set_xlabel("Minutes from window start")
            fig.suptitle(f"{gas} EDP — mean ± std across application windows")
            fig.tight_layout()

            if save:
                out = figures_path / f"{gas}_EDP_mean_std.png"
                fig.savefig(out, dpi=150)
                print(f"Saved {out}")

        if show:
            plt.show()

    def plot_EDP_individual(self, show=True, save=False, start_offset=50, end_offset=70):
        """
        Plot each EDP application window individually: one figure per window,
        with a gas-concentration subplot on top and one subplot per node
        (P1, P3, ...) below it, all sharing the same time interval/x-axis.
        The window's start timestamp is used as the plot title. Raw
        (non-normalized) values are shown. CH2 is ignored in every CSV.

        start_offset/end_offset (minutes from the window's own start), if
        given, restrict every subplot to that slice of the window - e.g.
        start_offset=50, end_offset=70 (the defaults) shows just the 10
        minutes before and the 10 minutes of the stimulus (the same
        sub-windows used by get_10min_calc_features). Pass None for either
        to leave that side unbounded, or both None for the full window.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        ignored_columns = {'CH2'}
        # columns that describe the control state, not a measured concentration
        state_columns = {'valve_state', 'plugstate', 'goal_concentration'}

        start_td = timedelta(minutes=start_offset) if start_offset is not None else None
        end_td = timedelta(minutes=end_offset) if end_offset is not None else None

        def clip_to_offsets(df):
            if df.empty or (start_td is None and end_td is None):
                return df
            window_start = df.index[0]
            lo = window_start + start_td if start_td is not None else df.index[0]
            hi = window_start + end_td if end_td is not None else df.index[-1]
            return df[(df.index >= lo) & (df.index <= hi)]

        file_suffix = ""
        if start_offset is not None or end_offset is not None:
            start_label = int(start_offset) if start_offset is not None else 0
            end_label = int(end_offset) if end_offset is not None else None
            file_suffix = f"_{start_label}to{end_label}min" if end_label is not None else f"_from{start_label}min"

        for gas in self.gases:
            edp_dir = time_window_path / gas / "EDP"
            gasdata_dir = time_window_path / gas / "gasdata"
            if not edp_dir.is_dir():
                print(f"No EDP directory for {gas}: {edp_dir}")
                continue

            # window start timestamp -> {node: file}
            windows = {}
            for node in self.PNs:
                node_dir = edp_dir / node
                if not node_dir.is_dir():
                    continue
                for file in sorted(node_dir.glob(f"{node}_*.csv")):
                    window_key = file.stem[len(node) + 1:]  # strip "<node>_" prefix
                    windows.setdefault(window_key, {})[node] = file

            if not windows:
                print(f"No EDP window CSVs found for {gas}")
                continue

            n_rows = len(self.PNs) + 1
            for window_key in sorted(windows):
                node_files = windows[window_key]
                fig, axes = plt.subplots(
                    n_rows, 1,
                    figsize=(12, 3 * n_rows),
                    sharex=True, squeeze=False,
                )

                any_plotted = False

                gas_ax = axes[0, 0]
                gas_file = gasdata_dir / f"{gas}_{window_key}.csv"
                if gas_file.is_file():
                    columns = pd.read_csv(gas_file, nrows=0).columns
                    index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                    gas_df = pd.read_csv(gas_file, index_col=index_col)
                    gas_df.index = pd.to_datetime(gas_df.index, errors='coerce')
                    gas_df = gas_df[~gas_df.index.isna()]
                    gas_df = clip_to_offsets(gas_df)
                    for col in gas_df.columns:
                        if col not in state_columns:
                            gas_ax.plot(gas_df.index, gas_df[col], linewidth=0.8, label=col)
                    if not gas_df.empty:
                        any_plotted = True
                    gas_ax.set_title(gas)
                    gas_ax.set_ylabel(f"{gas} concentration")
                    gas_ax.legend(loc='upper right')
                else:
                    gas_ax.set_title(f"{gas} (no data)")
                gas_ax.grid(True, alpha=0.3)

                for ax, node in zip(axes[1:, 0], self.PNs):
                    file = node_files.get(node)
                    if file is None:
                        ax.set_title(f"{node} (no data)")
                        ax.grid(True, alpha=0.3)
                        continue

                    columns = pd.read_csv(file, nrows=0).columns
                    index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                    df = pd.read_csv(file, index_col=index_col)
                    df = df.drop(columns=[c for c in ignored_columns if c in df.columns])
                    df.index = pd.to_datetime(df.index, errors='coerce')
                    df = df[~df.index.isna()]
                    df = clip_to_offsets(df)
                    if df.empty:
                        ax.set_title(f"{node} (empty)")
                        ax.grid(True, alpha=0.3)
                        continue

                    for col in df.columns:
                        ax.plot(df.index, df[col], linewidth=0.8, label=col)

                    ax.set_title(node)
                    ax.set_ylabel("electrical potential [mV]")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='upper right')
                    any_plotted = True

                if not any_plotted:
                    plt.close(fig)
                    continue

                axes[-1, 0].set_xlabel("Time")
                fig.suptitle(f"{gas} EDP — {window_key}")
                fig.tight_layout()

                if save:
                    out = figures_path / f"{gas}_EDP_{window_key}{file_suffix}.png"
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")

                if show:
                    plt.show()
                plt.close(fig)

    def plot_EDP_stimulus_overlay(self, show=True, save=True, start_offset=50, end_offset=70):
        """
        Like plot_EDP_individual, but instead of one figure per application
        window, produces one figure per experiment with every window from
        that experiment overlaid on the same subplot (gas concentration on
        top, one subplot per node below) - same layout as plot_EDP_data, but
        restricted to the stimulus slice [start_offset, end_offset] minutes
        into each window (defaults to the 10 minutes before + 10 minutes of
        stimulus, the same sub-windows used by get_10min_calc_features)
        instead of the full 2h10m window. Windows are attributed to their
        experiment via get_experiment_time_range (times.csv for O3, explicit
        start/end datetimes for CO2/N2), since window CSVs on disk only
        carry a window_start timestamp. Each trace's x-axis is minutes
        elapsed from the start of that slice, so all windows line up
        regardless of their real-world start time. EDP traces are z-scored
        per window/column before plotting; gas concentration is left in raw
        units. CH2 is ignored in every CSV.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        ignored_columns = {'CH2'}
        # columns that describe the control state, not a measured concentration
        state_columns = {'valve_state', 'plugstate', 'goal_concentration'}

        start_td = timedelta(minutes=start_offset) if start_offset is not None else None
        end_td = timedelta(minutes=end_offset) if end_offset is not None else None

        def clip_to_offsets(df):
            if df.empty or (start_td is None and end_td is None):
                return df
            window_start = df.index[0]
            lo = window_start + start_td if start_td is not None else df.index[0]
            hi = window_start + end_td if end_td is not None else df.index[-1]
            return df[(df.index >= lo) & (df.index <= hi)]

        file_suffix = ""
        if start_offset is not None or end_offset is not None:
            start_label = int(start_offset) if start_offset is not None else 0
            end_label = int(end_offset) if end_offset is not None else None
            file_suffix = f"_{start_label}to{end_label}min" if end_label is not None else f"_from{start_label}min"

        for gas in self.gases:
            edp_dir = time_window_path / gas / "EDP"
            gasdata_dir = time_window_path / gas / "gasdata"
            if not edp_dir.is_dir():
                print(f"No EDP directory for {gas}: {edp_dir}")
                continue

            # window start timestamp -> {node: file}
            all_windows = {}
            for node in self.PNs:
                node_dir = edp_dir / node
                if not node_dir.is_dir():
                    continue
                for file in sorted(node_dir.glob(f"{node}_*.csv")):
                    window_key = file.stem[len(node) + 1:]  # strip "<node>_" prefix
                    all_windows.setdefault(window_key, {})[node] = file

            if not all_windows:
                print(f"No EDP window CSVs found for {gas}")
                continue

            window_key_dt = {key: pd.to_datetime(key) for key in all_windows}

            for experiment in self.experiment_config[gas]:
                experiment_name = experiment['name']
                start_dt, end_dt = self.get_experiment_time_range(gas, experiment)
                windows = {
                    key: files for key, files in all_windows.items()
                    if start_dt <= window_key_dt[key] <= end_dt
                }
                if not windows:
                    print(f"No EDP window CSVs found for {gas} experiment {experiment_name}")
                    continue

                n_rows = len(self.PNs) + 1
                fig, axes = plt.subplots(
                    n_rows, 1,
                    figsize=(12, 3 * n_rows),
                    sharex=True, squeeze=False,
                )
                gas_ax = axes[0, 0]
                any_plotted = False

                gas_labeled_columns = set()
                for window_key in sorted(windows):
                    gas_file = gasdata_dir / f"{gas}_{window_key}.csv"
                    if not gas_file.is_file():
                        continue
                    columns = pd.read_csv(gas_file, nrows=0).columns
                    index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                    gas_df = pd.read_csv(gas_file, index_col=index_col)
                    gas_df.index = pd.to_datetime(gas_df.index, errors='coerce')
                    gas_df = gas_df[~gas_df.index.isna()]
                    gas_df = clip_to_offsets(gas_df)
                    if gas_df.empty:
                        continue
                    elapsed_min = (gas_df.index - gas_df.index[0]).total_seconds() / 60.0
                    for col in gas_df.columns:
                        if col in state_columns:
                            continue
                        label = col if col not in gas_labeled_columns else None
                        gas_ax.plot(elapsed_min, gas_df[col], linewidth=0.8, alpha=0.5,
                                    color=f"C{hash(col) % 10}", label=label)
                        gas_labeled_columns.add(col)
                    any_plotted = True

                gas_ax.set_title(gas)
                gas_ax.set_ylabel(f"{gas} concentration")
                gas_ax.grid(True, alpha=0.3)
                if gas_labeled_columns:
                    gas_ax.legend(loc='upper right')

                for ax, node in zip(axes[1:, 0], self.PNs):
                    node_labeled_columns = set()
                    for window_key in sorted(windows):
                        file = windows[window_key].get(node)
                        if file is None:
                            continue
                        columns = pd.read_csv(file, nrows=0).columns
                        index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                        df = pd.read_csv(file, index_col=index_col)
                        df = df.drop(columns=[c for c in ignored_columns if c in df.columns])
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        df = df[~df.index.isna()]
                        df = clip_to_offsets(df)
                        if df.empty:
                            continue
                        elapsed_min = (df.index - df.index[0]).total_seconds() / 60.0
                        for col in df.columns:
                            std = df[col].std()
                            if std == 0 or pd.isna(std):
                                continue
                            z_scored = (df[col] - df[col].mean()) / std
                            label = col if col not in node_labeled_columns else None
                            ax.plot(elapsed_min, z_scored, linewidth=0.8, alpha=0.5,
                                    color=f"C{hash(col) % 10}", label=label)
                            node_labeled_columns.add(col)
                        any_plotted = True

                    ax.set_title(node)
                    ax.set_ylabel("electrical potential (z-score)")
                    ax.grid(True, alpha=0.3)
                    if node_labeled_columns:
                        ax.legend(loc='upper right')

                if not any_plotted:
                    plt.close(fig)
                    print(f"No plottable rows for {gas} experiment {experiment_name}")
                    continue

                axes[-1, 0].set_xlabel("Minutes elapsed (from slice start)")
                fig.suptitle(f"{gas} {experiment_name} EDP — {len(windows)} application windows")
                fig.tight_layout()

                if save:
                    out = figures_path / f"{gas}_{experiment_name}_EDP_overlay{file_suffix}.png"
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")

                if show:
                    plt.show()
                plt.close(fig)

    def plot_EDP_stimulus_mean_std(self, show=True, save=True, start_offset=50, end_offset=70):
        """
        Like plot_EDP_stimulus_overlay, but instead of overlaying every raw
        window trace, aligns all of one experiment's windows onto a common
        per-second time grid within [start_offset, end_offset] (defaults to
        the 10 minutes before + 10 minutes of stimulus) and plots the mean
        trace with +/-1 standard deviation shaded as a band - one figure per
        experiment (gas concentration mean+-std on top, one subplot per node
        below). EDP traces are z-scored per window/column before
        aggregating; gas concentration is aggregated in raw units. CH2 is
        ignored in every CSV. Saved under a distinct "_EDP_mean_std"
        filename so it doesn't overwrite the raw-overlay figures from
        plot_EDP_stimulus_overlay.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        ignored_columns = {'CH2'}
        # columns that describe the control state, not a measured concentration
        state_columns = {'valve_state', 'plugstate', 'goal_concentration'}

        start_td = timedelta(minutes=start_offset) if start_offset is not None else None
        end_td = timedelta(minutes=end_offset) if end_offset is not None else None
        default_duration = timedelta(hours=2, minutes=10)
        slice_duration = (end_td - start_td) if (start_td is not None and end_td is not None) else default_duration
        grid_seconds = pd.RangeIndex(0, int(slice_duration.total_seconds()) + 1)
        elapsed_min = grid_seconds / 60.0

        def clip_to_offsets(df):
            if df.empty or (start_td is None and end_td is None):
                return df
            window_start = df.index[0]
            lo = window_start + start_td if start_td is not None else df.index[0]
            hi = window_start + end_td if end_td is not None else df.index[-1]
            return df[(df.index >= lo) & (df.index <= hi)]

        file_suffix = ""
        if start_offset is not None or end_offset is not None:
            start_label = int(start_offset) if start_offset is not None else 0
            end_label = int(end_offset) if end_offset is not None else None
            file_suffix = f"_{start_label}to{end_label}min" if end_label is not None else f"_from{start_label}min"

        for gas in self.gases:
            edp_dir = time_window_path / gas / "EDP"
            gasdata_dir = time_window_path / gas / "gasdata"
            if not edp_dir.is_dir():
                print(f"No EDP directory for {gas}: {edp_dir}")
                continue

            all_windows = {}
            for node in self.PNs:
                node_dir = edp_dir / node
                if not node_dir.is_dir():
                    continue
                for file in sorted(node_dir.glob(f"{node}_*.csv")):
                    window_key = file.stem[len(node) + 1:]
                    all_windows.setdefault(window_key, {})[node] = file

            if not all_windows:
                print(f"No EDP window CSVs found for {gas}")
                continue

            window_key_dt = {key: pd.to_datetime(key) for key in all_windows}

            for experiment in self.experiment_config[gas]:
                experiment_name = experiment['name']
                start_dt, end_dt = self.get_experiment_time_range(gas, experiment)
                windows = {
                    key: files for key, files in all_windows.items()
                    if start_dt <= window_key_dt[key] <= end_dt
                }
                if not windows:
                    print(f"No EDP window CSVs found for {gas} experiment {experiment_name}")
                    continue

                n_rows = len(self.PNs) + 1
                fig, axes = plt.subplots(
                    n_rows, 1,
                    figsize=(12, 3 * n_rows),
                    sharex=True, squeeze=False,
                )
                gas_ax = axes[0, 0]
                any_plotted = False

                gas_traces_by_col = {}
                for window_key in sorted(windows):
                    gas_file = gasdata_dir / f"{gas}_{window_key}.csv"
                    if not gas_file.is_file():
                        continue
                    columns = pd.read_csv(gas_file, nrows=0).columns
                    index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                    gas_df = pd.read_csv(gas_file, index_col=index_col)
                    gas_df.index = pd.to_datetime(gas_df.index, errors='coerce')
                    gas_df = gas_df[~gas_df.index.isna()]
                    gas_df = clip_to_offsets(gas_df)
                    if gas_df.empty:
                        continue
                    elapsed_seconds = ((gas_df.index - gas_df.index[0]).total_seconds()).round().astype(int)
                    for col in gas_df.columns:
                        if col in state_columns:
                            continue
                        trace = pd.Series(gas_df[col].values, index=elapsed_seconds)
                        trace = trace[~trace.index.duplicated(keep='first')]
                        gas_traces_by_col.setdefault(col, []).append(trace.reindex(grid_seconds))

                for col, traces in gas_traces_by_col.items():
                    stacked = pd.concat(traces, axis=1)
                    mean_trace = stacked.mean(axis=1, skipna=True)
                    std_trace = stacked.std(axis=1, skipna=True)
                    gas_ax.plot(elapsed_min, mean_trace, linewidth=1.2, label=col)
                    gas_ax.fill_between(elapsed_min, mean_trace - std_trace, mean_trace + std_trace, alpha=0.3)
                    any_plotted = True

                gas_ax.set_title(gas)
                gas_ax.set_ylabel(f"{gas} concentration")
                gas_ax.grid(True, alpha=0.3)
                if gas_traces_by_col:
                    gas_ax.legend(loc='upper right')

                for ax, node in zip(axes[1:, 0], self.PNs):
                    node_traces = []
                    for window_key in sorted(windows):
                        file = windows[window_key].get(node)
                        if file is None:
                            continue
                        columns = pd.read_csv(file, nrows=0).columns
                        index_col = 'datetime' if 'datetime' in columns else 'timestamp'
                        df = pd.read_csv(file, index_col=index_col)
                        df = df.drop(columns=[c for c in ignored_columns if c in df.columns])
                        df.index = pd.to_datetime(df.index, errors='coerce')
                        df = df[~df.index.isna()]
                        df = clip_to_offsets(df)
                        if df.empty:
                            continue
                        elapsed_seconds = ((df.index - df.index[0]).total_seconds()).round().astype(int)
                        for col in df.columns:
                            std = df[col].std()
                            if std == 0 or pd.isna(std):
                                continue
                            z_scored = (df[col] - df[col].mean()) / std
                            trace = pd.Series(z_scored.values, index=elapsed_seconds)
                            trace = trace[~trace.index.duplicated(keep='first')]
                            node_traces.append(trace.reindex(grid_seconds))

                    if node_traces:
                        stacked = pd.concat(node_traces, axis=1)
                        mean_trace = stacked.mean(axis=1, skipna=True)
                        std_trace = stacked.std(axis=1, skipna=True)
                        ax.plot(elapsed_min, mean_trace, linewidth=1.2, color='C0', label='mean')
                        ax.fill_between(
                            elapsed_min, mean_trace - std_trace, mean_trace + std_trace,
                            alpha=0.3, color='C0', label='±1 std',
                        )
                        ax.legend(loc='upper right')
                        any_plotted = True

                    ax.set_title(node)
                    ax.set_ylabel("electrical potential (z-score)")
                    ax.grid(True, alpha=0.3)

                if not any_plotted:
                    plt.close(fig)
                    print(f"No plottable rows for {gas} experiment {experiment_name}")
                    continue

                axes[-1, 0].set_xlabel("Minutes elapsed (from slice start)")
                fig.suptitle(f"{gas} {experiment_name} EDP — mean ± std across {len(windows)} application windows")
                fig.tight_layout()

                if save:
                    out = figures_path / f"{gas}_{experiment_name}_EDP_mean_std{file_suffix}.png"
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")

                if show:
                    plt.show()
                plt.close(fig)

    def plot_O3_data(self, show=True, save=False):
        """
        Plot every O3 window CSV under time_window_path/O3 one by one, using the
        file name as the plot title. Each file gets its own figure.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        figures_path = self.resolve_config_path(self.config_paths['figures_path'])
        o3_dir = time_window_path / "O3" / "gasdata"

        if not o3_dir.is_dir():
            print(f"No O3 time-window directory: {o3_dir}")
            return

        window_files = sorted(o3_dir.glob("O3_*.csv"))
        if not window_files:
            print(f"No O3 window CSVs found in {o3_dir}")
            return

        if save:
            figures_path.mkdir(parents=True, exist_ok=True)

        # columns that describe the control state, not a measured concentration
        state_columns = {'valve_state', 'plugstate', 'goal_concentration'}

        for file in window_files:
            columns = pd.read_csv(file, nrows=0).columns
            index_col = 'datetime' if 'datetime' in columns else 'timestamp'
            df = pd.read_csv(file, index_col=index_col)
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[~df.index.isna()]
            if df.empty:
                print(f"Skipping empty file {file.name}")
                continue

            fig, ax = plt.subplots(figsize=(12, 6))
            for col in df.columns:
                if col not in state_columns:
                    ax.plot(df.index, df[col], label=col)
            ax.set_xlabel("Time")
            ax.set_ylabel("O3")
            ax.set_title(file.name)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            if save:
                out = figures_path / f"{file.stem}.png"
                fig.savefig(out, dpi=150)
                print(f"Saved {out}")

            if show:
                plt.show()
            plt.close(fig)
    def extract_experiment_windows(self):
        print(f"Loaded experiments from: {self.experiments_file}")
        print(self.experiment_config)


    def get_10min_calc_features(self, background_subtraction=False):
        """
        Compute tsfresh time-series features on the CH1 signal for two
        10-minute sub-windows of every already-split EDP application window
        (P1/P3) - one right before the stimulus (50-60 minutes after the
        window start, prestimulus=True) and one right after (60-70 minutes,
        prestimulus=False) - and write one feature table per gas under
        features_path.

        If background_subtraction is True, the median CH1 value of a third,
        40-50 minute background sub-window is subtracted from the
        prestimulus and poststimulus signal values before features are
        extracted. The background sub-window itself is only used for this
        correction and is never written out as its own feature row.
        """
        time_window_path = self.resolve_config_path(self.config_paths['time_window_path'])
        features_path = self.resolve_config_path(self.config_paths['features_path'])
        features_path.mkdir(parents=True, exist_ok=True)

        signal_column = 'CH1'
        sub_window_duration = timedelta(minutes=10)
        sub_windows = [
            {'offset': timedelta(minutes=40), 'background': True},
            {'offset': timedelta(minutes=50), 'prestimulus': True},
            {'offset': timedelta(minutes=60), 'prestimulus': False},
        ]

        def load_signal(file):
            columns = pd.read_csv(file, nrows=0).columns
            index_col = 'datetime' if 'datetime' in columns else 'timestamp'
            if signal_column not in columns:
                return pd.DataFrame(columns=[signal_column])
            df = pd.read_csv(file, index_col=index_col, usecols=[index_col, signal_column])
            df.index = pd.to_datetime(df.index, errors='coerce')
            df = df[~df.index.isna()]
            df = df.dropna()
            return df

        def long_rows(df, node):
            # One (node, window_start, signal, sub_window_start, prestimulus)
            # id per sub-window, in the long format
            # tsfresh.extract_features expects.
            if df.empty:
                return []
            window_start = df.index[0]
            rows = []
            background_median = None
            for sub_window in sub_windows:
                sub_start = window_start + sub_window['offset']
                sub_end = sub_start + sub_window_duration
                chunk = df[(df.index >= sub_start) & (df.index < sub_end)]
                if chunk.empty:
                    continue
                if sub_window.get('background'):
                    background_median = chunk[signal_column].median()
                    continue
                values = chunk[signal_column].values
                if background_subtraction and background_median is not None:
                    values = values - background_median
                rows.append(pd.DataFrame({
                    'datetime': chunk.index,
                    'value': values,
                    'id': f"{node}|{window_start}|{signal_column}|{sub_start}|{sub_window['prestimulus']}",
                }))
            return rows

        for gas in self.gases:
            edp_dir = time_window_path / gas / "EDP"

            parts = []
            for node in self.PNs:
                node_dir = edp_dir / node
                for file in sorted(node_dir.glob(f"{node}_*.csv")) if node_dir.is_dir() else []:
                    parts.extend(long_rows(load_signal(file), node))

            if not parts:
                print(f"No windows found to compute features for {gas}")
                continue

            long_df = pd.concat(parts, ignore_index=True)

            features = extract_features(
                            long_df,
                            column_id="id",
                            column_sort="datetime",
                            column_value='value',  # Extract features on the 'value' column
                            #default_fc_parameters=MinimalFCParameters(),  # or Efficient/Comprehensive
                            disable_progressbar=False,
                            n_jobs= mp.cpu_count() -1,  # Use 1 job to avoid memory issues
                            #chunksize= 150,
                            #distributor=dist,
                        )

            id_parts = features.index.to_series().str.split("|", expand=True)
            id_parts.columns = ['node', 'window_start', 'signal', 'sub_window_start', 'prestimulus']
            id_parts.insert(0, 'gas', gas)
            id_parts['sub_window_end'] = pd.to_datetime(id_parts['sub_window_start']) + sub_window_duration
            id_parts['prestimulus'] = id_parts['prestimulus'] == 'True'
            features = pd.concat([id_parts.reset_index(drop=True), features.reset_index(drop=True)], axis=1)

            suffix = "_bgsub" if background_subtraction else ""
            out = features_path / f"{gas}_10min_features{suffix}.csv"
            features.to_csv(out, index=False)
            print(f"Saved {out}")


if __name__ == "__main__":
    pp = Preprocess()

    #pp.read_raw_experiment_data(split_gases=False, split_potential=True)
    #pp.plot_gas_data(show=True, save=True)
    #pp.plot_EDP_data(show=True, save=True)
    #pp.plot_EDP_mean_std(show=True, save=True)
    #pp.plot_EDP_individual(show=True, save=True)
    #pp.plot_O3_data(show=True, save=True)
    #pp.plot_EDP_stimulus_overlay(show=True, save=True)
    #pp.plot_EDP_stimulus_mean_std(show=True, save=True)
    pp.get_10min_calc_features(background_subtraction=True)