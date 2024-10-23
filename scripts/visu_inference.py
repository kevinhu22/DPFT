import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from dprt.utils.visu import get_box_corners, get_transformation


class Object3D:
    def __init__(self, xc, yc, zc, xl, yl, zl, rot_rad):
        self.xc, self.yc, self.zc, self.xl, self.yl, self.zl, self.rot_rad = (
            xc,
            yc,
            zc,
            xl,
            yl,
            zl,
            rot_rad,
        )

        corners_x = np.array([xl, xl, xl, xl, -xl, -xl, -xl, -xl]) / 2
        corners_y = np.array([yl, yl, -yl, -yl, yl, yl, -yl, -yl]) / 2
        corners_z = np.array([zl, -zl, zl, -zl, zl, -zl, zl, -zl]) / 2

        self.corners = np.row_stack((corners_x, corners_y, corners_z))

        rotation_matrix = np.array(
            [
                [np.cos(rot_rad), -np.sin(rot_rad), 0.0],
                [np.sin(rot_rad), np.cos(rot_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        self.corners = rotation_matrix.dot(self.corners).T + np.array(
            [[self.xc, self.yc, self.zc]]
        )


# Define arrays for intrinsic and extrinsic parameters
# intrinsic (fx, fy, px, py)
# extrinsic (roll, pitch, yaw, x, y, z) [degree, m]

intrinsic_params = [600.0, 600.0, 650.0, 390.0]
extrinsic_params = [0.0, -1.8, 1.8, 4.0, 0.0, -0.8]


# Define the function to handle the button click
def visualize(image_path, preds, gts, intrinsic, extrinsic, output_path):
    alpha = 0.5
    lthick = 2

    # cv_img = cv2.imread(image_path)  # Replace with the actual path
    cv_img = cv2.imread("data/Samsung 8TB 1/3/cam-front/cam-front_00464.png")
    cv_img = cv_img[:, 1280:, :]
    cv_img_ori = cv_img.copy()

    rot, tra = get_rotation_and_translation_from_extrinsic(extrinsic)

    list_line_order = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
    ]

    # K-Radar gts
    for gt_object in gts:
        x, y, z, l, w, h, theta = gt_object[2]
        obj3d = Object3D(x, y, z, l, w, h, theta)
        bbox_corners = obj3d.corners
        arr_points = get_pointcloud_with_rotation_and_translation(
            bbox_corners, rot, tra
        )

        arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
        for idx_1, idx_2 in list_line_order:
            p1_x, p1_y = arr_pix[idx_1]
            p2_x, p2_y = arr_pix[idx_2]
            p1_x = int(np.round(p1_x))
            p1_y = int(np.round(p1_y))
            p2_x = int(np.round(p2_x))
            p2_y = int(np.round(p2_y))

            color = [23, 208, 253]  # yellow
            cv_img = cv2.line(
                cv_img, (p1_x, p1_y), (p2_x, p2_y), color, thickness=lthick
            )

    # for gt_object in gts:

    #     bbox_corners = get_box_corners(gt_object["bbox"])

    #     arr_points = get_pointcloud_with_rotation_and_translation(
    #         bbox_corners, rot, tra
    #     )

    #     arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
    #     for idx_1, idx_2 in list_line_order:
    #         p1_x, p1_y = arr_pix[idx_1]
    #         p2_x, p2_y = arr_pix[idx_2]
    #         p1_x = int(np.round(p1_x))
    #         p1_y = int(np.round(p1_y))
    #         p2_x = int(np.round(p2_x))
    #         p2_y = int(np.round(p2_y))

    #         color = [23, 208, 253]
    #         cv_img = cv2.line(
    #             cv_img, (p1_x, p1_y), (p2_x, p2_y), color, thickness=lthick
    #         )

    for pred_object in preds:
        arr_points = get_box_corners(pred_object["bbox"])

        arr_points = get_pointcloud_with_rotation_and_translation(arr_points, rot, tra)
        arr_pix = get_pixel_from_point_cloud_in_camera_coordinate(arr_points, intrinsic)
        for idx_1, idx_2 in list_line_order:
            p1_x, p1_y = arr_pix[idx_1]
            p2_x, p2_y = arr_pix[idx_2]
            p1_x = int(np.round(p1_x))
            p1_y = int(np.round(p1_y))
            p2_x = int(np.round(p2_x))
            p2_y = int(np.round(p2_y))

            color = [0, 50, 255]
            cv_img = cv2.line(
                cv_img, (p1_x, p1_y), (p2_x, p2_y), color, thickness=lthick
            )

    # cv_img = cv2.addWeighted(cv_img, alpha, cv_img_ori, 1 - alpha, 0)
    # cv2.imshow("front_image", cv_img)
    cv2.imwrite(output_path, cv_img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_pointcloud_with_rotation_and_translation(point_cloud_xyz, rot, tra):
    pc_xyz = point_cloud_xyz.copy()
    num_points = len(pc_xyz)

    for i in range(num_points):
        point_temp = pc_xyz[i, :]
        point_temp = np.reshape(point_temp, (3, 1))

        point_processed = np.dot(rot, point_temp) + tra
        point_processed = np.reshape(point_processed, (3,))

        pc_xyz[i, :] = point_processed

    return pc_xyz


def get_label_bboxes(path_label, calib_info):
    with open(path_label, "r") as f:
        lines = f.readlines()
        f.close()
    # print('* lines : ', lines)
    line_objects = lines[1:]
    # print('* line_objs: ', line_objects)
    list_objects = []

    # print(dict_temp['meta']['path_label'])
    for line in line_objects:
        temp_tuple = get_tuple_object(
            line, calib_info, is_heading_in_rad=True, path_label=path_label
        )
        if temp_tuple is not None:
            list_objects.append(temp_tuple)

    return list_objects


def get_tuple_object(line, calib_info, is_heading_in_rad=True, path_label=None):
    """
    * in : e.g., '*, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> One Example
    * in : e.g., '*, 0, 0, Sedan, 3.8, 5.535, -1.0, -93.1155, 2.112, 1.0347, 0.95' --> There are labels like this too
    * out: tuple ('Sedan', idx_cls, [x, y, z, theta, l, w, h], idx_obj)
    *       None if idx_cls == -1 or header != '*'
    """
    list_values = line.split(",")

    if list_values[0] != "*":
        return None

    offset = 0
    if (len(list_values)) == 11:
        # print('* Exception error (Dataset): length of values is 11')
        offset = 1
    else:
        print("* Exception error (Dataset): length of values is 10")
        print(path_label)

    cls_name = list_values[2 + offset][1:]

    CLASS_ID = {
        "Sedan": 1,
        "Bus or Truck": -1,
        "Motorcycle": -1,
        "Bicycle": -1,
        "Bicycle Group": -1,
        "Pedestrian": -1,
        "Pedestrian Group": -1,
        "Background": 0,
    }

    idx_cls = CLASS_ID[cls_name]

    if idx_cls == -1:  # Considering None as -1
        return None

    idx_obj = int(list_values[1 + offset])
    x = float(list_values[3 + offset])
    y = float(list_values[4 + offset])
    z = float(list_values[5 + offset])
    theta = float(list_values[6 + offset])
    if is_heading_in_rad:
        theta = theta * np.pi / 180.0
    l = 2 * float(list_values[7 + offset])
    w = 2 * float(list_values[8 + offset])
    h = 2 * float(list_values[9 + offset])

    x = x + calib_info[0]
    y = y + calib_info[1]
    z = z + calib_info[2]

    # Check if the label is in roi (For Radar, checking azimuth angle)
    # print('* x, y, z: ', x, y, z)
    # print('* roi_label: ', self.roi_label)
    ROI_DEFAULT = [
        0,
        120,
        -100,
        100,
        -50,
        50,
    ]  # x_min_max, y_min_max, z_min_max / Dim: [m]
    x_min, x_max, y_min, y_max, z_min, z_max = ROI_DEFAULT
    if (
        (x > x_min)
        and (x < x_max)
        and (y > y_min)
        and (y < y_max)
        and (z > z_min)
        and (z < z_max)
    ):
        # print('* debug 1')

        ### RDR: Check azimuth angle if it is valid ###
        if True:
            min_azi, max_azi = [-50, 50]
            # print('* min, max: ', min_azi, max_azi)
            if True:  # center
                azimuth_center = np.arctan2(y, x)
                if (azimuth_center < min_azi) or (azimuth_center > max_azi):
                    # print(f'* debug 2-1, azimuth = {azimuth_center}')
                    return None
                # for pt in pts:
                #     azimuth_apex = np.arctan2(pt[1], pt[0])
                #     # print(azimuth_apex)
                #     if (azimuth_apex < min_azi) or (azimuth_apex > max_azi):
                #         # print('* debug 2-2')
                #         return None
        ### RDR: Check azimuth angle if it is valid ###

        # print('* debug 3')
        return (cls_name, idx_cls, [x, y, z, theta, l, w, h], idx_obj)
    else:
        # print('* debug 4')
        return None


def get_pixel_from_point_cloud_in_camera_coordinate(point_cloud_xyz, intrinsic):
    """
    * in : pointcloud in np array (nx3)
    * out: projected pixel in np array (nx2)
    """

    process_pc = point_cloud_xyz.copy()
    if np.shape(point_cloud_xyz) == 1:
        num_points = 0
    else:
        # Temporary fix for when shape = (0.)
        try:
            num_points, _ = np.shape(point_cloud_xyz)
        except:
            num_points = 0
    fx, fy, px, py = intrinsic

    pixels = []
    for i in range(num_points):
        xc, yc, zc = process_pc[i, :]
        y_pix = py - fy * zc / xc
        x_pix = px - fx * yc / xc

        pixels.append([x_pix, y_pix])
    pixels = np.array(pixels)

    return pixels


# def vis_infer(self, sample_indices, conf_thr=0.7, is_nms=True, vis_mode=['lpc', 'spcube', 'cube'], is_train=False):
#         '''
#         * sample_indices: e.g. [0, 1, 2, 3, 4]
#         * assume batch_size = 1 for convenience
#         * vis_mode (TBD)
#         '''
#         self.network.eval()

#             ### Labels ###
#             labels = dict_out['label'][0]
#             list_obj_label = []
#             for label_obj in labels:
#                 cls_name, cls_id, (xc, yc, zc, rot, xl, yl, zl), obj_idx = label_obj
#                 obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
#                 list_obj_label.append(obj)
#             ### Labels ###

#             ### Preds: post processing bbox ###
#             list_obj_pred = []
#             list_cls_pred = []
#             if dict_datum['pp_num_bbox'] == 0:
#                 pass
#             else:
#                 pp_cls = dict_datum['pp_cls']
#                 for idx_pred, pred_obj in enumerate(dict_datum['pp_bbox']):
#                     conf_score, xc, yc, zc, xl, yl, zl, rot = pred_obj
#                     obj = Object3D(xc, yc, zc, xl, yl, zl, rot)
#                     list_obj_pred.append(obj)
#                     list_cls_pred.append('Sedan')
#                     # list_cls_pred.append(self.dict_cls_id_to_name[pp_cls[idx_pred]])
#             ### Preds: post processing bbox ###

#             ### Vis for open3d ###
#             lines = [[0, 1], [1, 2], [2, 3], [0, 3],
#                     [4, 5], [6, 7], #[5, 6],[4, 7],
#                     [0, 4], [1, 5], [2, 6], [3, 7],
#                     [0, 2], [1, 3], [4, 6], [5, 7]]
#             colors_label = [[0, 0, 0] for _ in range(len(lines))]
#             list_line_set_label = []
#             list_line_set_pred = []
#             for label_obj in list_obj_label:
#                 line_set = o3d.geometry.LineSet()
#                 line_set.points = o3d.utility.Vector3dVector(label_obj.corners)
#                 line_set.lines = o3d.utility.Vector2iVector(lines)
#                 line_set.colors = o3d.utility.Vector3dVector(colors_label)
#                 list_line_set_label.append(line_set)

#             for idx_pred, pred_obj in enumerate(list_obj_pred):
#                 line_set = o3d.geometry.LineSet()
#                 line_set.points = o3d.utility.Vector3dVector(pred_obj.corners)
#                 line_set.lines = o3d.utility.Vector2iVector(lines)
#                 # colors_pred = [self.dict_cls_name_to_rgb[list_cls_pred[idx_pred]] for _ in range(len(lines))]
#                 colors_pred = [[1.,0.,0.] for _ in range(len(lines))]
#                 line_set.colors = o3d.utility.Vector3dVector(colors_pred)
#                 list_line_set_pred.append(line_set)

#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(pc_lidar[:, :3])
#             o3d.visualization.draw_geometries([pcd] + list_line_set_label + list_line_set_pred)
#             ### Vis for open3d ###

#         return list_obj_label, list_obj_pred


def get_rotation_and_translation_from_extrinsic(extrinsic, is_deg=True):
    ext_copy = extrinsic.copy()
    if is_deg:
        ext_copy[:3] = list(map(lambda x: x * np.pi / 180.0, extrinsic[:3]))

    roll, pitch, yaw = ext_copy[:3]
    x, y, z = ext_copy[3:]

    ### Roll-Pitch-Yaw Convention
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    c_r = np.cos(roll)
    s_r = np.sin(roll)

    R_yaw = np.array([[c_y, -s_y, 0.0], [s_y, c_y, 0.0], [0.0, 0.0, 1.0]])
    R_pitch = np.array([[c_p, 0.0, s_p], [0.0, 1.0, 0.0], [-s_p, 0.0, c_p]])
    R_roll = np.array([[1.0, 0.0, 0.0], [0.0, c_r, -s_r], [0.0, s_r, c_r]])

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)
    trans = np.array([[x], [y], [z]])

    return R, trans


def read_log_file(log_file):
    """Reads the log file and returns a list of steps and image paths."""
    steps = []
    image_paths = []
    with open(log_file, "r") as file:
        for line in file:
            parts = line.strip().split(", ")
            step = int(parts[0].split(": ")[1])
            image_path = parts[1].split(": ")[1]
            steps.append(step)
            image_paths.append(image_path)
    return steps, image_paths


def read_txt_file(txt_file):
    """Reads the txt file and returns a list of bounding box objects."""
    objects = []
    with open(txt_file, "r") as file:
        for line in file:
            parts = line.strip().split(" ")
            obj = {
                "name": parts[0],
                "truncated": float(parts[1]),
                "occluded": float(parts[2]),
                "alpha": float(parts[3]),
                "bbox": {
                    "x": float(parts[13]),
                    "y": float(parts[11]),
                    "z": float(parts[12]),
                    "theta": float(parts[14]),
                    "l": float(parts[10]),
                    "w": float(parts[9]),
                    "h": float(parts[8]),
                },
            }
            objects.append(obj)
    return objects


def project_3d_to_2d(x, y, z, h, w, l, theta):
    """Projects 3D bounding box to 2D image plane with a scaling factor."""
    # Define the eight corners of the 3D bounding box
    corners = np.array(
        [
            [l / 2, w / 2, h / 2],
            [l / 2, -w / 2, h / 2],
            [-l / 2, -w / 2, h / 2],
            [-l / 2, w / 2, h / 2],
            [l / 2, w / 2, -h / 2],
            [l / 2, -w / 2, -h / 2],
            [-l / 2, -w / 2, -h / 2],
            [-l / 2, w / 2, -h / 2],
        ]
    )

    # Rotation matrix around the z-axis
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    # Rotate and translate the corners
    corners = np.dot(corners, R.T)
    corners += np.array([x, y, z])

    # Project the 3D corners to 2D (ignoring z)
    corners_2d = corners[:, :2]

    return corners_2d


def visualize_inference(log_file, base_dir, output_dir):
    """Visualizes the inference results."""
    steps, image_paths = read_log_file(log_file)
    processed_folders = set()

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for step, image_path in zip(steps, image_paths):
        # if step != 862:
        #     continue
        folder_number = image_path.split("/")[
            6
        ]  # Extract the folder number from the image path

        if folder_number in processed_folders:
            continue

        processed_folders.add(folder_number)

        step_str = str(step).zfill(6)

        preds_file = os.path.join(base_dir, "preds", f"{step_str}.txt")
        gts_file_dpft = os.path.join(base_dir, "gts", f"{step_str}.txt")
        labels_file = os.path.join(os.path.dirname(image_path), "labels.npy")

        if (
            not os.path.exists(preds_file)
            or not os.path.exists(gts_file_dpft)
            or not os.path.exists(image_path)
            or not os.path.exists(labels_file)
        ):
            print(f"Files for step {step_str} not found.")
            continue

        preds = read_txt_file(preds_file)
        gts_dpft = read_txt_file(gts_file_dpft)
        output_path = os.path.join(output_dir, f"{step_str}.png")
        gts_file = "data/Samsung 8TB 1/7/info_label_v2/00182_00149.txt"
        calib_info = np.array([33, -2.54, 0.3])
        gts = get_label_bboxes(gts_file, calib_info)

        visualize(
            image_path, preds, gts, intrinsic_params, extrinsic_params, output_path
        )


if __name__ == "__main__":
    log_file = "app/log/20241017-081549-441/exports/kradar/0.7/export_log.txt"  # Path to the log file
    base_dir = "app/log/20241017-081549-441/exports/kradar/0.7/all"  # Base directory containing preds and gts folders
    output_dir = "app/output/inference_images"  # Directory to save the output images
    visualize_inference(log_file, base_dir, output_dir)
