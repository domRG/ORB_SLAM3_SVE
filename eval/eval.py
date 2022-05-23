import argparse
import sys
import numpy as np
import pandas as pd
import json

import matplotlib

matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'

import manipulations


class Evaluator:
    def __init__(self, args_dict):

        """ arg_dict should be something like:
        {
            'groundtruth_traj_file': '~/Documents/fyp/datasets/euroc/MH04/mav0/state_groundtruth_estimate0/data.csv',
            'estimated_traj_file': '../mh04_res0-25_ros_output_traj.txt',
            'verbose': './mh04_res0-25_metrics.json',
            'plot': './mh04_res0-25_ros_traj_plot',
            'sve': ['../mh04_res0-25_ros_output_sve.txt', './mh04_res0-25_sve_plot']
        }
        """

        self.args = args_dict
        self.gt_traj = None
        self.est_traj = None
        self.total_df = None
        self.gt_column_map = None
        self.gt_xyz = None
        self.est_xyz = None
        self.rot = None
        self.transGT = None
        self.trans_errorGT = None
        self.trans = None
        self.trans_error = None
        self.scale = None
        self.est_xyz_aligned = None

        self.output_details = {}
        self.verbose_precision = 6

    def __enter__(self):
        self.load_data()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            plt.close('all')
        except:
            print("Failed to close figures")

    def load_data(self):
        self.gt_traj = manipulations.csv_to_df(self.args["groundtruth_traj_file"])
        self.est_traj = manipulations.csv_to_df(self.args["estimated_traj_file"], scale_time_df=self.gt_traj)

        self.total_df, self.gt_column_map = manipulations.merge_traj(self.gt_traj, self.est_traj)

    def preprocess(self):
        if self.total_df is None:
            raise RuntimeError("data not yet loaded, call this.load_data() first or use a context manager")
        self.gt_xyz = self.total_df[[self.gt_column_map[i] for i in ["px", "py", "pz"]]].to_numpy().transpose()
        self.est_xyz = self.total_df[["t_1", "t_2", "t_0"]].to_numpy().transpose()

        self.rot, self.transGT, self.trans_errorGT, self.trans, self.trans_error, self.scale = manipulations.align(
            self.est_xyz, self.gt_xyz)

        self.est_xyz_aligned = np.asarray(((self.scale * self.rot * self.est_xyz).transpose() + self.trans).transpose())
        self.total_df["t_1"] = self.est_xyz_aligned[0]
        self.total_df["t_2"] = self.est_xyz_aligned[1]
        self.total_df["t_0"] = self.est_xyz_aligned[2]

    def main(self):
        self.preprocess()

        if "plot" in self.args.keys():
            self.plot(self.args["plot"])

        if "sve" in self.args.keys():
            self.sve(*self.args["sve"])

        if "verbose" in self.args.keys():
            self.verbose(self.args["verbose"])
        else:
            print(
                f"{np.sqrt(np.dot(self.trans_error, self.trans_error) / len(self.trans_error))}, {self.scale}, {np.sqrt(np.dot(self.trans_errorGT, self.trans_errorGT) / len(self.trans_errorGT)):.6f}")

        if "display" in self.args.keys():
            self.display(self.args["display"])

    def plot(self, filename):
        traj_fig = plt.figure()
        ax = traj_fig.add_subplot(1, 1, 1, projection="3d")

        ax.plot(self.total_df[self.gt_column_map["px"]], self.total_df[self.gt_column_map["py"]],
                self.total_df[self.gt_column_map["pz"]], '-',
                color='black', label='ground truth', linewidth=0.5)
        ax.plot(self.total_df["t_1"], self.total_df["t_2"], self.total_df["t_0"], '-', color='blue',
                label='estimated',
                linewidth=0.5)

        ax.legend()

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')

        ax.set_title(input("Enter Trajectory Title\n> "))

        traj_fig.savefig(f"{filename}.pdf", format="pdf")
        print(f"Plot saved to {filename}.pdf")

    def sve(self, data_file=None, filename=None, do_plot=True):
        if data_file is None:
            raise ValueError("No SVE datafile provided")

        self.sve_df = manipulations.csv_to_df(data_file, scale_time_df=self.gt_traj)

        self.output_details["sve"] = {}
        self.output_details["sve"]["len"] = len(self.sve_df["sve"])
        self.output_details["sve"]["rms"] = np.round(np.sqrt(np.mean(np.square(self.sve_df["sve"]))), self.verbose_precision)
        self.output_details["sve"]["mean"] = np.round(np.mean(self.sve_df["sve"]), self.verbose_precision)
        self.output_details["sve"]["median"] = np.round(np.median(self.sve_df["sve"]), self.verbose_precision)
        self.output_details["sve"]["std"] = np.round(np.std(self.sve_df["sve"]), self.verbose_precision)
        self.output_details["sve"]["min"] = np.round(np.min(self.sve_df["sve"]), self.verbose_precision)
        self.output_details["sve"]["max"] = np.round(np.max(self.sve_df["sve"]), self.verbose_precision)

        self.output_details["sve"]["time"] = {}
        self.output_details["sve"]["time"]["start"] = np.round(np.min(self.sve_df.norm_time), self.verbose_precision)
        self.output_details["sve"]["time"]["end"] = np.round(np.max(self.sve_df.norm_time), self.verbose_precision)
        self.output_details["sve"]["time"]["description"] = "start and end time values in normalised time (ie relative to groundtruth start and finish)"

        if do_plot:
            if filename is None:
                raise ValueError("No SVE output filename provided")
            print("plotting...")

            sve_fig = plt.figure()
            sve_ax = sve_fig.add_subplot(1, 2, 1)
            sve_ax.plot(self.sve_df["norm_time"], self.sve_df["sve"], linewidth=0.5)
            sve_ax.set_title("Combined SVE Metric")
            sve_ax.set_ylabel("Estimated Visibility (Magnitude)")
            sve_ax.set_xlabel("Normalised Time")

            for i, val in enumerate(['a', 'b', 'c']):
                sve_i_ax = sve_fig.add_subplot(3, 2, (i + 1) * 2)
                sve_i_ax.plot(self.sve_df["norm_time"], self.sve_df["sve_%s" % val], linewidth=0.5)
                sve_i_ax.set_title("SVE Metric %s" % val.upper())
                sve_i_ax.set_ylabel("Magnitude")
                sve_i_ax.set_xlabel("Normalised Time")

            plt.tight_layout()
            sve_fig.suptitle(input("Enter SVE Title\n> "))

            sve_fig.savefig(f"{filename}.pdf", format="pdf")
            print(f"Plot saved to {filename}.eps")

    def verbose(self, filename):
        self.output_details["traj"] = {}
        self.output_details["traj"]["len"] = (len(self.trans_error))

        self.output_details["traj"]["error"] = {}
        self.output_details["traj"]["error"]["rmse"] = np.round(
            np.sqrt(np.dot(self.trans_error, self.trans_error) / len(self.trans_error)), self.verbose_precision)
        self.output_details["traj"]["error"]["mean"] = np.round(np.mean(self.trans_error), self.verbose_precision)
        self.output_details["traj"]["error"]["median"] = np.round(np.median(self.trans_error), self.verbose_precision)
        self.output_details["traj"]["error"]["std"] = np.round(np.std(self.trans_error), self.verbose_precision)
        self.output_details["traj"]["error"]["min"] = np.round(np.min(self.trans_error), self.verbose_precision)
        self.output_details["traj"]["error"]["max"] = np.round(np.max(self.trans_error), self.verbose_precision)
        self.output_details["traj"]["error"]["description"] = "absolute translational error"

        self.output_details["traj"]["errorGT"] = {}
        self.output_details["traj"]["errorGT"]["rmse"] = np.round(
            np.sqrt(np.dot(self.trans_errorGT, self.trans_errorGT) / len(self.trans_errorGT)), self.verbose_precision)
        self.output_details["traj"]["errorGT"]["description"] = "absolute translational errorGT"

        self.output_details["traj"]["time"] = {}
        self.output_details["traj"]["time"]["start"] = np.round(np.min(self.total_df.norm_time), self.verbose_precision)
        self.output_details["traj"]["time"]["end"] = np.round(np.max(self.total_df.norm_time), self.verbose_precision)
        self.output_details["traj"]["time"]["description"] = "start and end time values in normalised time (ie relative to groundtruth start and finish)"

        with open(filename, "w") as of:
            json.dump(self.output_details, of)

    def display(self, enable):
        if enable:
            try:
                plt.show(block=True)
            except:
                print("Failed to display figures")

    def get_results(self):
        return self.output_details


if __name__ == "__main__":
    print(sys.version)

    """
    =======================
    Configure CLI Arguments
    =======================
    """
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('groundtruth_traj_file',
                        help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('estimated_traj_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose',
                        help='print and save verbose trajectory metrics')
    parser.add_argument('--sve', nargs=2,
                        help='include Scene Visibility Estimation analysis, args: sve_data, output_filename')
    parser.add_argument('--display',
                        help='enable figure plot display (hangs for figure to be closed)',
                        action='store_true')
    args = parser.parse_args()
    arg_dict = vars(args)

    with Evaluator(arg_dict) as ev:
        ev.main()
