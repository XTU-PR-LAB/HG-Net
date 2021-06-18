import numpy as np
import cv2 as cv
import math
from collections import Counter
from path import Path
from PIL import Image
from matplotlib import pyplot as plt


def visualize_colormap(mat, colormap=cv.COLORMAP_JET, print_if=False):
    mat[mat > 80.] = 80.
    # mat=np.reciprocal(mat)
    # mat[np.nonzero(mat)]=1.0/mat
    min_val = 1.0  # np.amin(mat)
    max_val = 80.  # np.amax(mat)

    mat_view = (mat - min_val) / (max_val - min_val)
    mat_view *= 255
    mat_view = mat_view.astype(np.uint8)
    mat_view = cv.applyColorMap(mat_view, colormap)

    return mat_view


def post_process_disparity(ori_disp):
    disp = np.copy(ori_disp)
    disp[disp > 80.] = 80.
    # disp[np.logical_and(disp > 0, disp < 1)] = 1
    disp[disp < 4] = 4
    disp[disp > 0] = 1.0 / disp[disp > 0]
    # cmap = 'plasma'
    h, w = disp.shape
    # l_disp = disp[0, :, :]
    # r_disp = np.fliplr(disp[1, :, :])
    l_disp = disp
    r_disp = l_disp
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    disp_view = r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp
    return disp_view


def load_velodyne_points(file_name):
    # adapted from https://github.com/hunse/kitti
    points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1
    return points


def read_calib_file(path):
    # taken from https://github.com/hunse/kitti
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' '))))
                except ValueError:
                    # casting error: data[key] already eq. value, so pass
                    pass

    return data


def sub2ind(matrixSize, rowSub, colSub):
    m, n = matrixSize
    return rowSub * (n - 1) + colSub - 1


def generate_depth_map(calib_dir, velo_file_name, im_shape, cam=3):
    # load calibration files
    cam2cam = read_calib_file(calib_dir / 'calib_cam_to_cam.txt')
    velo2cam = read_calib_file(calib_dir / 'calib_velo_to_cam.txt')
    velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

    # compute projection matrix velodyne->image plane
    R_cam2rect = np.eye(4)
    R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
    P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_file_name)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, -1:]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape))
    depth[velo_pts_im[:, 1].astype(np.int), velo_pts_im[:, 0].astype(np.int)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0
    return depth


def generate_mask(gt_depth, min_depth, max_depth):
    mask = np.logical_and(gt_depth > min_depth,
                          gt_depth < max_depth)
    # crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    # if used on gt_size 370x1224 produces a crop of [-218, -3, 44, 1180]
    gt_height, gt_width = gt_depth.shape
    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    return mask


def split(value):
    diff = value - int(value)
    if diff >= 0.5:
        value = int(value) + 1
    else:
        value = int(value)
    return value


def depth_scale(depth, H, W):
    ori_H, ori_W = depth.shape
    ratio_x = W / ori_W
    ratio_y = H / ori_H
    counter = [[[] for j in range(W)] for i in range(H)]
    new_depth = np.zeros((H, W), dtype=np.float32)
    for row in range(ori_H):
        for col in range(ori_W):
            new_x = split(col * ratio_x)
            new_y = split(row * ratio_y)
            if new_x < 0 or new_x >= W or new_y < 0 or new_y >= H:
                continue
            if depth[row][col] > 0:
                counter[new_y][new_x].append(depth[row][col])

    for row in range(H):
        for col in range(W):
            if len(counter[row][col]) > 0:
                new_depth[row][col] = np.min(counter[row][col])

    return new_depth


def depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert (np.max(depth_png) > 255)

    depth = depth_png.astype(np.float) / 256.
    mask = np.logical_or(depth < 1e-3, depth > 80)
    depth[mask] = 0
    # depth[depth_png == 0] = -1.
    return depth