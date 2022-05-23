#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Modified by Raul Mur-Artal
# Automatically compute the optimal scale factor for monocular VO/SLAM.

# Software License Agreement (BSD License)
#
# Copyright (c) 2013, Juergen Sturm, TUM
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of TUM nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Requirements: 
# sudo apt-get install python-argparse

"""
This script computes the absolute trajectory error from the ground truth
trajectory and the estimated trajectory.
"""

from hashlib import new
from multiprocessing.sharedctypes import Value
import sys
import numpy
from scipy import interpolate
import argparse
import associate
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.patches import Ellipse


class TimestampError(Exception):
    def __init__(self, message=None):
        if not message:
            message = "Couldn't find matching timestamp pairs between groundtruth and estimated trajectory! Did you choose the correct sequence?"
        super(Exception, self).__init__(message)


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
            f[k] = interpolate.interp1d(gt_data["t"], gt_data[k], kind = 'linear')
    
    # for each timestamp in estimated trajectory
    # add/update entries in groundtruth based on the interpolators
    for est_t in estimated_list.keys():
        try:
            groundtruth_list[est_t] = [f[i](est_t) for i in sorted(f.keys())]
        except ValueError:
            # print("Time out of range:\t%f" % est_t)
            pass
    
    return groundtruth_list

def remove_outliers(data):
    pass

def align(model,data):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    """


    numpy.set_printoptions(precision=3,suppress=True)
    model_zerocentered = (model.transpose() - model.mean(1)).transpose()
    data_zerocentered = (data.transpose() - data.mean(1)).transpose()

    W = numpy.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += numpy.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = numpy.linalg.linalg.svd(W.transpose())
    S = numpy.matrix(numpy.identity( 3 ))
    if(numpy.linalg.det(U) * numpy.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    rotmodel = rot*model_zerocentered
    dots = 0.0
    norms = 0.0

    for column in range(data_zerocentered.shape[1]):
        dots += numpy.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
        normi = numpy.linalg.norm(model_zerocentered[:,column])
        norms += normi*normi

    s = float(dots/norms)

    transGT = data.mean(1) - (s*rot).dot(model.mean(1))
    trans = data.mean(1) - rot.dot(model.mean(1))

    model_alignedGT = (s*rot.dot(model).transpose() + transGT).transpose()
    model_aligned = ((rot * model).transpose() + trans).transpose()

    alignment_errorGT = model_alignedGT - data
    alignment_error = model_aligned - data

    trans_errorGT = numpy.sqrt(numpy.sum(numpy.multiply(alignment_errorGT,alignment_errorGT),0)).A[0]
    trans_error = numpy.sqrt(numpy.sum(numpy.multiply(alignment_error,alignment_error),0)).A[0]

    return rot,transGT,trans_errorGT,trans,trans_error, s

def plot_traj(ax,stamps,traj,style,color,label,linewidth):
    """
    Plot a trajectory using matplotlib. 
    
    Input:
    ax -- the plot
    stamps -- time stamps (1xn)
    traj -- trajectory (3xn)
    style -- line style
    color -- line color
    label -- plot legend
    
    """
    # not made for use with dataframes
    # attempts some amount of windowing on data to present nicely?)

    stamps.sort()
    interval = numpy.median([s-t for s,t in zip(stamps[1:],stamps[:-1])])
    x = []
    y = []
    last = stamps[0]
    for i in range(len(stamps)):
        if stamps[i]-last < 2*interval:
            x.append(traj[i][0])
            y.append(traj[i][1])
        elif len(x)>0:
            ax.plot(x,y,style,color=color,label=label,linewidth=linewidth)
            label=""
            x=[]
            y=[]
        last= stamps[i]
    if len(x)>0:
        ax.plot(x,y,style,color=color,label=label,linewidth=linewidth)

"""
good standard use:
evaluation/Ground_truth/EuRoC_left_cam/MH_GT.txt Examples/Stereo-Inertial/f_dataset-MH01-05_stereoi.txt --verbose --verbose2 --plot MH01_to_MH05_stereo.pdf
"""
if __name__=="__main__":
    print(sys.version)
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script computes the absolute trajectory error from the ground truth trajectory and the estimated trajectory. 
    ''')
    parser.add_argument('groundtruth_traj_file', help='ground truth trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('estimated_traj_file', help='estimated trajectory (format: timestamp tx ty tz qx qy qz qw)')
    parser.add_argument('--offset', help='time offset added to the timestamps of the second file (default: 0.0)',default=0.0)
    parser.add_argument('--scale', help='scaling factor for the second trajectory (default: 1.0)',default=1.0)
    parser.add_argument('--max_difference', help='maximally allowed time difference for matching entries (default: 10000000 ns)',default=20000000)
    parser.add_argument('--save', help='save aligned second trajectory to disk (format: stamp2 x2 y2 z2)')
    parser.add_argument('--save_associations', help='save associated first and aligned second trajectory to disk (format: stamp1 x1 y1 z1 stamp2 x2 y2 z2)')
    parser.add_argument('--plot', help='plot the first and the aligned second trajectory to an image (format: png)')
    parser.add_argument('--verbose', help='print all evaluation data (otherwise, only the RMSE absolute translational error in meters after alignment will be printed)', action='store_true')
    parser.add_argument('--verbose2', help='print scale eror and RMSE absolute translational error in meters after alignment with and without scale correction', action='store_true')
    parser.add_argument('--sve', nargs=2, help='include Scene Visibility Estimation analysis, args: sve_data, output_filename')
    args = parser.parse_args()

    gt_traj = associate.read_dataframe(args.groundtruth_traj_file)
    est_traj = associate.read_dataframe(args.estimated_traj_file, t_factor=1.0)

    matches = associate.associate_df(gt_traj, est_traj,float(args.offset),float(args.max_difference))
    # try:
    #     # if more than half the estimations can be associated with gt data
    #     if len(matches) < 0.5 * len(est_traj.index.values):
    #         raise TimestampError
    # except TimestampError:
    #     # TODO: implement dataframe support
    #     # handle data interpolation to match timestamps
    #     print("Interpolating provided groundtruth data...")
        
    #     gt_traj_list = interp_groundtruth(gt_traj_list, est_traj_list)
    #     matches = associate.associate_df(gt_traj, est_traj,float(args.offset),float(args.max_difference))

    #     if len(matches)<2:
    #         print("Still insufficient matches, cancelling evalutation")
    #         sys.exit()
    #     else:
    #         print("Interpolation successful, continuing")

    first_xyz = gt_traj.loc[[matches[i][0] for i in range(len(matches))], ['Local_X', 'Local_Y', 'Local_Z']].to_numpy().transpose()
    second_xyz = est_traj.loc[[matches[i][1] for i in range(len(matches))], ['t_0', 't_1', 't_2']].to_numpy().transpose()
    # dictionary_items = est_traj_list.items()
    # sorted_second_list = sorted(dictionary_items)

    second_xyz_full = est_traj[['t_0', 't_1', 't_2']].to_numpy().transpose()
    rot,transGT,trans_errorGT,trans,trans_error, scale = align(second_xyz,first_xyz)

    second_xyz_aligned = ((scale * rot * second_xyz).transpose() + trans).transpose()
    second_xyz_notscaled = ((rot * second_xyz).transpose() + trans).transpose()
    second_xyz_notscaled_full = ((rot * second_xyz_full).transpose() + trans).transpose()
    first_xyz_full = gt_traj.loc[:, ['Local_X', 'Local_Y', 'Local_Z']].to_numpy().transpose()

    second_xyz_full = est_traj.loc[:, ['t_0', 't_1', 't_2']].to_numpy().transpose()
    second_xyz_full_aligned = numpy.asarray(((scale * rot * second_xyz_full).transpose() + trans).transpose())

    if args.verbose:
        print "compared_pose_pairs %d pairs"%(len(trans_error))

        print "absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error, trans_error) / len(trans_error))
        print "absolute_translational_error.mean %f m"%numpy.mean(trans_error)
        print "absolute_translational_error.median %f m"%numpy.median(trans_error)
        print "absolute_translational_error.std %f m"%numpy.std(trans_error)
        print "absolute_translational_error.min %f m"%numpy.min(trans_error)
        print "absolute_translational_error.max %f m"%numpy.max(trans_error)
        print "max idx: %i" %numpy.argmax(trans_error)
    else:
        # print "%f, %f " % (numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)),  scale)
        # print "%f,%f" % (numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)),  scale)
        print "%f,%f,%f" % (numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error)), scale, numpy.sqrt(numpy.dot(trans_errorGT,trans_errorGT) / len(trans_errorGT)))
        # print "%f" % len(trans_error)
    if args.verbose2:
        print "compared_pose_pairs %d pairs"%(len(trans_error))
        print "absolute_translational_error.rmse %f m"%numpy.sqrt(numpy.dot(trans_error,trans_error) / len(trans_error))
        print "absolute_translational_errorGT.rmse %f m"%numpy.sqrt(numpy.dot(trans_errorGT,trans_errorGT) / len(trans_errorGT))

    if args.save_associations:
        file = open(args.save_associations,"w")
        file.write("\n".join(["%f %f %f %f %f %f %f %f"%(a,x1,y1,z1,b,x2,y2,z2) for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches,first_xyz.transpose().A,second_xyz_aligned.transpose().A)]))
        file.close()

    if args.save:
        file = open(args.save,"w")
        file.write("\n".join(["%f "%stamp+" ".join(["%f"%d for d in line]) for stamp,line in zip(second_stamps,second_xyz_notscaled_full.transpose().A)]))
        file.close()

    if args.plot:
        traj_fig = plt.figure()
        ax = traj_fig.add_subplot(111)
        label="difference"
        for (a,b),(x1,y1,z1),(x2,y2,z2) in zip(matches, first_xyz.transpose().tolist(), second_xyz_aligned.transpose().tolist()):
            ax.plot([x1,x2],[y1,y2],'-',color="red",label=label,linewidth=0.25)
            label=""
        # plot_traj(ax, gt_traj.index.values, first_xyz_full.transpose(),'-',"black","ground truth",linewidth=0.5)
        ax.plot(first_xyz_full[0], first_xyz_full[1], '-', color='black', label='ground truth', linewidth=0.5)
        # plot_traj(ax, est_traj.index.values, second_xyz_full_aligned.transpose(),'-',"blue","estimated",linewidth=0.5)
        ax.plot(second_xyz_full_aligned[0], second_xyz_full_aligned[1], '-', color='blue', label='estimated', linewidth=0.5)

        ax.legend()

        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        plt.axis('equal')
        plt.savefig("%s.pdf" % args.plot, format="pdf")
        plt.show(block=True)
        print("Plot saved to %s.pdf" % args.plot)

    if len(args.sve) == 2:
        sve_fig = plt.figure()

        # sve_dict = associate.read_dataframe(args.sve[0])
        sve_df = associate.read_dataframe(args.sve[0])

        # remove offset for timestamps? normalise?
        # time_start = timestamps[0]
        # for i in range(len(timestamps)):
        #     timestamps[i] = timestamps[i] - time_start

        print("plotting...")

        sve_ax = sve_fig.add_subplot(1, 2, 1)
        sve_ax.plot(sve_df.index.values, sve_df["SVE"], linewidth=0.25)

        for i, val in enumerate(['a', 'b', 'c']):
            sve_i_ax = sve_fig.add_subplot(3, 2, (i+1)*2)
            sve_i_ax.plot(sve_df.index.values, sve_df["SVE_%s" % val], linewidth=0.25)

        print("done")

        plt.tight_layout()

        print("saving...")
        plt.savefig("%s.pdf" % args.sve[1], format="pdf")
        plt.show(block=True)
        print("done")



