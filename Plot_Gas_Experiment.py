import matplotlib.pyplot as plt
import argparse
import json

from utils import *
from pathlib import Path, PurePath
from PhytoNode.PhytoNode import PhytoNode
from Get_Gas_Data import GetGasData


EXPERIMENTS_FILE = Path(__file__).with_name("gas_experiments.json")



class GasExperiment:
    def __init__(self, directory=None):




        self.directory = directory

        self.PNs = ["P1", "P3"]

    def compress_data(self, gas_path, start_datetime=None, end_datetime=None, resolution=None) -> pd.DataFrame:
        """
        Load gas experiment data from CSV files in gas_path directory,
        filtering by full datetime range (not just by filename).

        :param gas_path: Path to folder containing CSVs
        :param start_datetime: Start datetime (str or datetime object)
        :param end_datetime: End datetime (str or datetime object)
        :param resolution: Resampling frequency (e.g., '10min' for 10 minutes, '1H' for 1 hour, etc.)
        :return: Filtered DataFrame
        """
        GGD = GetGasData()
        gas_data = GGD.load_data(self.directory / "Ozone", start_datetime=start_datetime, end_datetime=end_datetime, resolution=resolution)
        #GGD.plot_data(gas_data)
        for node in self.PNs:
            print(f"Compressing node {node} ")
            PN1 = PhytoNode(node, self.directory/ "PN" /node)

            PN1_data = PN1.load_data(PN1.directory, start_datetime=start_datetime, end_datetime=end_datetime,
                                     resolution=resolution, inVolt=True, save=True)
            #PN1.plot_single_PN(PN1_data)

        #GGD = GetGasData(self.directory / "Carbon")
        #gas_data = GGD.load_data(GGD.directory, start_datetime=start_datetime, end_datetime=end_datetime, resolution=resolution)
        #GGD.plot_data(gas_data)
    def read_compressed_data(self, gas_path, start_datetime=None, end_datetime=None, resolution=None) -> pd.DataFrame:
        """
        Load gas experiment data from CSV files in gas_path directory,
        filtering by full datetime range (not just by filename).

        :param gas_path: Path to folder containing CSVs
        :param start_datetime: Start datetime (str or datetime object)
        :param end_datetime: End datetime (str or datetime object)
        :param resolution: Resampling frequency (e.g., '10min' for 10 minutes, '1H' for 1 hour, etc.)
        :return: Filtered DataFrame
        """
        data = []
        for node in self.PNs:
            print(f"Compressing node {node} ")
            PN1 = PhytoNode(node, self.directory / "PN" /node / "resampled" / resolution)

            PN1_data = PN1.load_data(PN1.directory, start_datetime=start_datetime, end_datetime=end_datetime,
                                     resolution=False, inVolt=False, save=False)
            data.append(PN1_data)
            #PN1.plot_single_PN(PN1_data)

        GGD = GetGasData()
        gas_data = GGD.load_data(self.directory / "nitrogen", start_datetime=start_datetime, end_datetime=end_datetime, resolution=resolution)
        #GGD.plot_data(gas_data)
        print(gas_data)
        print(data[0])
        print(type(data[0].index))
        print(type(gas_data.index))
        print(data[0].index.equals(gas_data.index))  # Should return True
        print(data[0].index.difference(gas_data.index))  # Shows what's missing in df2
        print(gas_data.index.difference(data[0].index))  # Shows what's missing in df1

        self.plot_dataframes(3,1, [data[0]["CH1"]],[data[1]["CH1"]],[gas_data["O2_1"],gas_data["O2_2"]])#,[data[1]["CH1"]])
        return data, gas_data

    def plot_dataframes(self, n_rows, n_cols, *df_lists):

        titles = ["Leaf", "Stem", "nitrogen"]
        y_labels = ["electrical potential [mV]", "electrical potential [mV]", "CO2 concentration [ppm]"]
        total_plots = n_rows * n_cols
        num_lists = len(df_lists)

        if num_lists > total_plots:
            raise ValueError(f"Too many DataFrame groups ({num_lists}) for {total_plots} subplots.")

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
        axes = axes.flatten() if total_plots > 1 else [axes]

        for i, df_list in enumerate(df_lists):
            ax = axes[i]
            for df in df_list:
                df.plot(ax=ax)
            ax.set_title(titles[i])
            ax.set_xlabel("datetime")
            ax.set_ylabel(y_labels[i])
            ax.grid(True,which='both',axis='both')

        # Turn off unused subplots
        for j in range(num_lists, total_plots):
            axes[j].axis("off")
        #plt.grid()
        plt.tight_layout()
        plt.show()
        """
        read_leaf = 1
        read_stem = 1
        read_control = 0
        read_oxygen = 1
        read_MU = 0

        directory = PurePath(config['DATA_PATH']) / "GasSetup/CarbonDioxide/Exp25_06"

        resolution = 1

        if read_leaf:
            PN1 = readPN(directory / "PN/P1", 0, -1, resolution)
            PN1["CH1"] = getVolt(PN1["CH1"], 4)
            PN1["CH1"] = PN1["CH1"].rolling(window=200, min_periods=1).mean()
            ax2.plot(PN1["datetime"], PN1["CH1"] - 0, label="leaf")
            # daysmarking(PN1,ax2)

        if read_stem:
            PN1 = readPN(directory / "PN/P3", 0, -1, resolution)
            PN1["CH1"] = getVolt(PN1["CH1"], 4)
            ax3.plot(PN1["datetime"], PN1["CH1"] + 0, label="stem")

        if read_control:
            PN1 = readPN(directory / "PN/P9", 0, -1, resolution)
            PN1["CH1"] = getVolt(PN1["CH1"], 4)
            ax2.plot(PN1["datetime"], PN1["CH1"] - 0, label="control")

        if read_oxygen:
            oxygen = readOzone(directory / "Carbon")
            ax1.plot(oxygen['datetime'], oxygen['CO2_top'], label="top")
            ax1.plot(oxygen['datetime'], oxygen['CO2_bot'], label="bottom")
            ax1.plot(oxygen['datetime'], (oxygen['CO2_top'] + oxygen['CO2_bot']) / 2, label="mean", linestyle="--",
                     color="black")
            ax1.plot(oxygen['datetime'], oxygen['goal_concentration'], label="Generator")
            ax1.plot(oxygen['datetime'], oxygen['valve_state'] * 1000, label="Generator")

        if read_MU:
            BB = BlueBox()
            what_to_plot = [['temp-external']]
            BB_data = BB.getData(directory / "MU", what_to_plot)[0]
            ax3.plot(BB_data['timestamp'], BB_data['transpiration'], label="transpiration")

        ## plot formatting
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S:%f'))
        ax1.set_xticks([])
        ax1.legend(loc="lower right")
        ax1.set_ylabel("Oxygen Concentration [%]")

        # ax2.set(xticks=PN["timestamp"][::int(len(PN["timestamp"]) / 10)], xticklabels=PN["timestamp"][::int(len(PN["timestamp"]) / 10)])
        ax2.tick_params(axis='x', labelrotation=45)
        ax2.legend(loc="lower right")
        ax2.set_ylabel("potential [mV]")

        # plt.gca().format_coord = fmt
        plt.tight_layout()
        plt.show()

        print(f"Loading gas experiment data from {gas_path}...")
        concatenated_data = self.concatenate(gas_path, start_datetime=start_datetime, end_datetime=end_datetime)

        if resolution:
            concatenated_data = concatenated_data.resample(resolution).mean()

        return concatenated_data
        """

def parse_args():
    parser = argparse.ArgumentParser(description="Plot a configured gas experiment interval.")
    parser.add_argument(
        "--experiment",
        default="exp25_12_full",
        help="Experiment name from gas_experiments.json. Use None/empty to plot the full data range.",
    )
    parser.add_argument(
        "--experiments-file",
        default=EXPERIMENTS_FILE,
        type=Path,
        help="Path to the experiment interval JSON file.",
    )
    parser.add_argument(
        "--resolution",
        default="1min",
        help="Resampling resolution, for example '1min', '10min', or '1H'.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    experiment_name = args.experiment or None

    hostname = socket.gethostname()
    print(hostname)
    if hostname == "MacBook-Pro-Eddy.local":
        config = loadyaml(Path("../config_MAC.yaml"))
    elif hostname == "watchplant":
        config = loadyaml(Path("../config_linux.yaml"))

    GE = GasExperiment(PurePath(config['DATA_PATH']) / "watchplant/GasSetup/O2/Exp44_Ivy2")
    GE.compress_data(GE.directory / "PN", start_datetime=None, end_datetime=None, resolution=args.resolution)
    GE.read_compressed_data(
        GE.directory / "PN",
        start_datetime=None,
        end_datetime=None,
        resolution=args.resolution,
    )
    print("END")
