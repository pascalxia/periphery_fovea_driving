from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset import Dataset
import tensorflow as tf
import numpy as np
import os, random
import util
import util_car
import scipy.misc as misc
import glob
import multiprocessing
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
from PIL import Image
import cv2
import ctypes
import math, copy
from scipy import interpolate


ctx_channel = 19
ctx_height = 60
ctx_width = 90
len_seg_ctx = 100
#FRAMES_IN_SEG=570

city_im_channel = 3
city_seg_channel = 1
city_frames = 5
city_lock = multiprocessing.Lock()

NO_SLIGHT_TURN = True
DECELERATION_THRES = 1
HZ = 3

# the newly designed class has to have those methods
# especially the reader() that reads the binary record and the
# parse_example_proto that parse a single record into an instance
class MyDataset(Dataset):
    def __init__(self, subset):
        super(MyDataset, self).__init__('nexar', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        # TODO: not useful
        return 2

    def download_message(self):
        print('Failed to find any nexar %s files' % self.subset)

    @staticmethod
    def future_smooth(actions, naction, nfuture):
        # TODO: could add weighting differently between near future and far future
        # given a list of actions, for each time step, return the distribution of future actions
        l = len(actions) # action is a list of integers, from 0 to naction-1, negative values are ignored
        out = np.zeros((l, naction), dtype=np.float32)
        for i in range(l):
            # for each output position
            total = 0
            for j in range(min(nfuture, l-i)):
                # for each future position
                # current deal with i+j action
                acti = i + j
                if actions[acti]>=0:
                    out[i, actions[acti]] += 1
                    total += 1
            if total == 0:
                out[i, MyDataset.turn_str2int['straight']] = 1.0
            else:
                out[i, :] = out[i, :] / total
        return out

    @staticmethod
    def speed_to_future_has_stop(speed, nfuture, speed_limit_as_stop):
        # expect the stop_label to be 1 dimensional, representing the stop labels along time
        # nfuture is how many future stop labels to consider
        speed = np.linalg.norm(speed, axis=1)
        stop_label = np.less(speed, speed_limit_as_stop)
        stop_label = stop_label.astype(np.int32)

        # the naction=2 means: the number of valid actions are 2
        smoothed = MyDataset.future_smooth(stop_label, 2, nfuture)
        out = np.less(0, smoothed[:, 1]).astype(np.int32)
        return out

    @staticmethod
    def no_stop_dropout_valid(stop_label, drop_prob):
        nbatch = stop_label.shape[0]
        ntime  = stop_label.shape[1]
        out = np.ones(nbatch, dtype=np.bool)
        for i in range(nbatch):
            # determine whether this seq has stop
            has_stop = False
            for j in range(ntime):
                if stop_label[i, j]:
                    has_stop = True
                    break
            if not has_stop:
                if np.random.rand() < drop_prob:
                    out[i] = False
        
        return out

    @staticmethod
    def speed_to_course(speed):
        pi = math.pi
        if speed[1] == 0:
            if speed[0] > 0:
                course = pi / 2
            elif speed[0] == 0:
                course = None
            elif speed[0] < 0:
                course = 3 * pi / 2
            return course
        course = math.atan(speed[0] / speed[1])
        if course < 0:
            course = course + 2 * pi
        if speed[1] > 0:
            course = course
        else:
            course = pi + course
            if course > 2 * pi:
                course = course - 2 * pi
        assert not math.isnan(course)
        return course

    @staticmethod
    def to_course_list(speed_list):
        l = speed_list.shape[0]
        course_list = []
        for i in range(l):
            speed = speed_list[i,:]
            course_list.append(MyDataset.speed_to_course(speed))
        return course_list

    turn_str2int={'not_sure': -1, 'straight': 0, 'slow_or_stop': 1,
                  'turn_left': 2, 'turn_right': 3,
                  'turn_left_slight': 4, 'turn_right_slight': 5,}
                  #'acceleration': 6, 'deceleration': 7}

    turn_int2str={y: x for x, y in turn_str2int.items()}
    naction = np.sum(np.less_equal(0, np.array(list(turn_str2int.values()))))

    @staticmethod
    def turning_heuristics(speed_list, speed_limit_as_stop=0):
        course_list = MyDataset.to_course_list(speed_list)
        speed_v = np.linalg.norm(speed_list, axis=1)
        l = len(course_list)
        action = np.zeros(l).astype(np.int32)
        course_diff = np.zeros(l).astype(np.float32)

        enum = MyDataset.turn_str2int

        thresh_low = (2*math.pi / 360)*1
        thresh_high = (2*math.pi / 360)*35
        thresh_slight_low = (2*math.pi / 360)*3

        def diff(a, b):
            # return a-b \in -pi to pi
            d = a - b
            if d > math.pi:
                d -= math.pi * 2
            if d < -math.pi:
                d += math.pi * 2
            return d

        for i in range(l):
            if i == 0:
                action[i] = enum['not_sure']
                continue

            # the speed_limit_as_stop should be small,
            # this detect strict real stop
            if speed_v[i] < speed_limit_as_stop + 1e-3:
                # take the smaller speed as stop
                action[i] = enum['slow_or_stop']
                continue

            course = course_list[i]
            prev = course_list[i-1]

            if course is None or prev is None:
                action[i] = enum['slow_or_stop']
                course_diff[i] = 9999
                continue

            course_diff[i] = diff(course, prev)*360/(2*math.pi)
            if thresh_high > diff(course, prev) > thresh_low:
                if diff(course, prev) > thresh_slight_low:
                    action[i] = enum['turn_right']
                else:
                    action[i] = enum['turn_right_slight']

            elif -thresh_high < diff(course, prev) < -thresh_low:
                if diff(course, prev) < -thresh_slight_low:
                    action[i] = enum['turn_left']
                else:
                    action[i] = enum['turn_left_slight']
            elif diff(course, prev) >= thresh_high or diff(course, prev) <= -thresh_high:
                action[i] = enum['not_sure']
            else:
                action[i] = enum['straight']

            if NO_SLIGHT_TURN:
                if action[i] == enum['turn_left_slight']:
                    action[i] = enum['turn_left']
                if action[i] == enum['turn_right_slight']:
                    action[i] = enum['turn_right']

            # this detect significant slow down that is not due to going to turn
            if DECELERATION_THRES > 0 and action[i] == enum['straight']:
                hz = HZ
                acc_now = (speed_v[i] - speed_v[i - 1]) / (1.0 / hz)
                if acc_now < - DECELERATION_THRES:
                    action[i] = enum['slow_or_stop']
                    continue

        # avoid the initial uncertainty
        action[0] = action[1]
        return action

    @staticmethod
    def turn_future_smooth(speed, nfuture, speed_limit_as_stop):
        # this function takes in the speed and output a smooth future action map
        turn = MyDataset.turning_heuristics(speed, speed_limit_as_stop)
        smoothed = MyDataset.future_smooth(turn, MyDataset.naction, nfuture)
        return smoothed

    @staticmethod
    def fix_none_in_course(course_list):
        l = len(course_list)

        # fix the initial None value
        not_none_value = 0
        for i in range(l):
            if not (course_list[i] is None):
                not_none_value = course_list[i]
                break
        for i in range(l):
            if course_list[i] is None:
                course_list[i] = not_none_value
            else:
                break

        # a course could be None, use the previous course in that case
        for i in range(1, l):
            if course_list[i] is None:
                course_list[i] = course_list[i - 1]
        return course_list

    @staticmethod
    def relative_future_location(speed, nfuture, sample_rate):
        # given the speed vectors, calculate the future location relative to
        # the current location, with facing considered
        course_list = MyDataset.to_course_list(speed)
        course_list = MyDataset.fix_none_in_course(course_list)

        # integrate the speed to get the location
        loc = util_car.integral(speed, 1.0 / sample_rate)

        # project future motion on to the current facing direction
        # this is counter clock wise
        def rotate(vec, theta):
            c = math.cos(theta)
            s = math.sin(theta)
            xp = c * vec[0] - s * vec[1]
            yp = s * vec[0] + c * vec[1]
            return np.array([xp, yp])

        out = np.zeros_like(loc)
        l = out.shape[0]
        for i in range(l):
            future = loc[min(i+nfuture, l-1), :]
            delta = future - loc[i, :]
            out[i, :] = rotate(delta, course_list[i])

        return out

    @staticmethod
    def relative_future_course_speed(speed, nfuture, sample_rate):
        def norm_course_diff(course):
            if course > math.pi:
                course = course - 2*math.pi
            if course < -math.pi:
                course = course + 2*math.pi
            return course

        # given the speed vectors, calculate the future location relative to
        # the current location, with facing considered
        course_list = MyDataset.to_course_list(speed)
        course_list = MyDataset.fix_none_in_course(course_list)

        # integrate the speed to get the location
        loc = util_car.integral(speed, 1.0 / sample_rate)

        out = np.zeros_like(loc)
        l = out.shape[0]
        for i in range(l):
            if i+nfuture < l:
                fi = min(i + nfuture, l - 1)
                # first is course diff
                out[i, 0] = norm_course_diff(course_list[fi] - course_list[i])
                # second is the distance
                out[i, 1] = np.linalg.norm(loc[fi, :] - loc[i, :])
            else:
                # at the end of the video, just use what has before
                out[i,:] = out[i-1,:]

        # normalize the speed to be per second
        timediff = 1.0 * nfuture / sample_rate
        out = out / timediff

        return out

    
    @staticmethod
    def parse_array(array):

        type_code = np.asscalar(np.fromstring(array[0:4], dtype=np.int32))
        shape_size = np.asscalar(np.fromstring(array[4:8], dtype=np.int32))

        shape = np.fromstring(array[8: 8+4 * shape_size], dtype=np.int32)
        if type_code == 5:#cv2.CV_32F:
            dtype = np.float32
        if type_code == 6:#cv2.CV_64F:
            dtype = np.float64
        return np.fromstring(array[8+4 * shape_size:], dtype=dtype).reshape(shape)

    
    def read_array(self, array_buffer):
        fn = lambda array: MyDataset.parse_array(array)
        ctx_decoded = map(fn, array_buffer)                       
        return [ctx_decoded]


    def queue_cityscape(self, image_dir, seg_dir):
        city_im_queue = []
        city_seg_queue = []
        with open(image_dir,'r') as f:
            for content in f.readlines():
                city_im_queue.append(content)
        with open(seg_dir,'r') as f:
            for content in f.readlines():
                city_seg_queue.append(content)

        assert(len(city_im_queue) == len(city_seg_queue))

        return city_im_queue, city_seg_queue

    @staticmethod
    def generate_meshlist(arange1, arange2):
        return np.dstack(np.meshgrid(arange1, arange2, indexing='ij')).reshape((-1, 2))


    # the input should be bottom cropped image, i.e. no car hood
    @staticmethod
    def rotate_ground(original, theta, horizon=60, half_height=360 / 2, focal=1.0):
        height, width, channel = original.shape
        # the target grids
        yp = range(height - horizon, height)
        xp = range(0, width)

        # from pixel to coordinates
        y0 = (np.array(yp) - half_height) * 1.0 / half_height
        x0 = (np.array(xp) - width / 2) / (width / 2.0)

        # form the mesh
        mesh = MyDataset.generate_meshlist(x0, y0)
        # compute the source coordinates
        st = math.sin(theta)
        ct = math.cos(theta)
        deno = ct * focal + st * mesh[:, 0]
        out = np.array([(-st * focal + ct * mesh[:, 0]) / deno, mesh[:, 1] / deno])

        # interpolate
        vout = []
        for i in range(3):
            f = interpolate.RectBivariateSpline(y0, x0, original[- horizon:, :, i])
            values = f(out[1, :], out[0, :], grid=False)
            vout.append(values)

        lower = np.reshape(vout, (3, width, horizon)).transpose((2, 1, 0)).astype("uint8")

        # compute the upper part
        out = np.reshape(out[0, :], (width, horizon))
        out = out[:, 0]
        f = interpolate.interp1d(x0, original[:-horizon, :, :], axis=1,
                                 fill_value=(original[:-horizon, 0, :], original[:-horizon, -1, :]),
                                 bounds_error=False)
        ans = f(out)
        ans = ans.astype("uint8")

        return np.concatenate((ans, lower), axis=0)

