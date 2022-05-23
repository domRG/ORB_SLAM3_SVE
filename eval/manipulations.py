import re
import numpy as np
from scipy import interpolate
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


def interp_groundtruth(groundtruth_list, estimated_list):
    """
    adds interpolated points to the groundtruth data list
    to match the times of the estimated data
    """

    # create dict with keys: t & 0->len(item value in groundtruth)
    gt_data = {}
    for i in ["t"] + range(len(groundtruth_list.items()[0][1])):
        gt_data[i] = []

    # populate with gt data
    for gt_t, gt_v in sorted(groundtruth_list.items()):
        gt_data["t"].append(gt_t)
        for i, v in enumerate(gt_v):
            gt_data[i].append(v)

    # generate interpolators (linear)
    f = {}
    for k, v in gt_data.items():
        if k is not "t":
            f[k] = interpolate.interp1d(gt_data["t"], gt_data[k], kind='linear')

    # for each timestamp in estimated trajectory
    # add/update entries in groundtruth based on the interpolators
    for est_t in estimated_list.keys():
        try:
            groundtruth_list[est_t] = [f[i](est_t) for i in sorted(f.keys())]
        except ValueError:
            # print("Time out of range:\t%f" % est_t)
            pass

    return groundtruth_list


def find_col_titles(traj_df):
    titles = {}
    for item in list(traj_df.columns.values):
        if re.findall(r'time', item, flags=re.IGNORECASE):
            titles["time"] = item
        elif re.findall(r'p.+x', item, flags=re.IGNORECASE):
            titles["px"] = item
        elif re.findall(r'p.+y', item, flags=re.IGNORECASE):
            titles["py"] = item
        elif re.findall(r'p.+z', item, flags=re.IGNORECASE):
            titles["pz"] = item
        elif re.findall(r'q.+x', item, flags=re.IGNORECASE):
            titles["qx"] = item
        elif re.findall(r'q.+y', item, flags=re.IGNORECASE):
            titles["qy"] = item
        elif re.findall(r'q.+z', item, flags=re.IGNORECASE):
            titles["qz"] = item
        elif re.findall(r'q.+w', item, flags=re.IGNORECASE):
            titles["qw"] = item
    return titles


def align(model, data):
    """Align two trajectories using the method of Horn (closed-form).

    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)

    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """

    np.set_printoptions(precision=3, suppress=True)
    model_zerocentered = (model.transpose() - model.mean(1)).transpose()
    data_zerocentered = (data.transpose() - data.mean(1)).transpose()

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity(3))
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U * S * Vh

    rotmodel = rot * model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += np.dot(data_zerocentered[:, column].transpose(), rotmodel[:, column])
        normi = np.linalg.norm(model_zerocentered[:, column])
        norms += normi * normi

    s = float(dots / norms)

    transGT = data.mean(1) - (s * rot).dot(model.mean(1))
    trans = data.mean(1) - rot.dot(model.mean(1))

    model_alignedGT = (s * rot.dot(model).transpose() + transGT).transpose()
    model_aligned = ((rot * model).transpose() + trans).transpose()

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = np.sqrt(np.sum(np.multiply(alignment_errorGT, alignment_errorGT), 0)).A[0]
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error, alignment_error), 0)).A[0]

    return rot, transGT, trans_errorGT, trans, trans_error, s


def csv_to_df(traj_file, scale_time_df=None):
    traj_df = pd.read_csv(traj_file)
    traj_df.rename(str.lower, axis='columns', inplace=True)
    col_titles = find_col_titles(traj_df)
    traj_df.rename({col_titles["time"]: "time"}, axis='columns', inplace=True)

    if scale_time_df is not None:
        # find order of magnitude time difference (usually 10E^9)
        t_mag_order = 10 ** np.round(np.log10(scale_time_df["time"].values[0] / traj_df["time"][0]))
        traj_df["time"] = traj_df["time"].apply(lambda x: x * t_mag_order)
        traj_df = add_normal_to_gt(traj_df, scale_time_df)
    else:
        t_start = np.min(traj_df["time"])
        t_end = np.max(traj_df["time"])
        traj_df["norm_time"] = np.apply_along_axis(lambda x: (x - t_start) / (t_end - t_start), 0, traj_df["time"])

    return traj_df


def merge_traj(gt_traj, est_traj):
    merged_df = gt_traj.merge(est_traj, how='outer').sort_values(by="time")
    merged_df.set_index("time", inplace=True)

    gt_col_titles = find_col_titles(gt_traj)
    gt_col_titles.pop("time")
    gt_important_columns = list(gt_col_titles.values())
    merged_df[gt_important_columns] = merged_df[gt_important_columns].interpolate(method='slinear',
                                                                                limit=1,
                                                                                limit_direction="backward")
    merged_df.dropna(subset=["t_0", gt_col_titles["px"]], inplace=True)
    merged_df.dropna(axis="columns", inplace=True)

    # t_start = np.min(gt_traj.time)
    # t_end = np.max(gt_traj.time)
    # merged_df["norm_time"] = np.apply_along_axis(lambda x: (x - t_start)/(t_end - t_start), 0, merged_df.index.values)

    merged_df = add_normal_to_gt(merged_df, gt_traj)

    return merged_df, gt_col_titles

def add_normal_to_gt(main_df, gt_df):
    try:
        tmp = main_df.norm_time
    except:
        t_start = np.min(gt_df.time)
        t_end = np.max(gt_df.time)
        main_df["norm_time"] = np.apply_along_axis(lambda x: (x - t_start) / (t_end - t_start), 0, main_df["time"])

    return main_df
