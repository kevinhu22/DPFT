from __future__ import annotations  # noqa: F407

import os
import os.path as osp
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import cache
from glob import glob
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from scipy import ndimage
import scipy.ndimage.filters as filters
from pypcd import pypcd

from dprt.datasets.kradar.utils import radar_info
from dprt.datasets.kradar.utils import split

import matplotlib.pyplot as plt

class CFAR:
    def __init__(self, roi, type="pointcloud"):
        """
        * type in ['pointcloud', 'index', 'both']
        """

        ### Design parameters ###
        # self.LARGE_VALUE = 1e+15
        self.grid_size = 0.4  # [m]
        self.n_half_guard_cell_zyx = [1, 2, 4]
        self.n_half_train_cell_zyx = [4, 8, 16]

        self.guard_cell_range_zyx = (
            (2 * (np.array(self.n_half_guard_cell_zyx)) + 1) * self.grid_size
        ).tolist()
        self.boundary_cell_range_zyx = (
            (
                2 * (np.array(self.n_half_train_cell_zyx))
                + 2 * (np.array(self.n_half_guard_cell_zyx))
                + 1
            )
            * self.grid_size
        ).tolist()

        self.fa_rate = 0.05  # [m] for CA-CFAR
        self.thr_rate = 0.02  # for OS-CFAR
        ### Design parameters ###

        self.roi = roi

        self.arr_z_cb = np.arange(roi["z"][0], roi["z"][1], 0.4)
        self.arr_y_cb = np.arange(roi["y"][0], roi["y"][1], 0.4)
        self.arr_x_cb = np.arange(roi["x"][0], roi["x"][1], 0.4)

        self.min_values = [np.min(self.arr_z_cb), np.min(self.arr_y_cb), np.min(self.arr_x_cb)]

        ### Return mode ###
        if type == "pointcloud":
            self.mode = 0
        elif type == "index":
            self.mode = 1
        elif type == "both":
            self.mode = 2
        ### Return mode ###

        self.CA_CFAR = {
            'FALSE_ALARM_RATE': 0.0005,
            'GUARD_CELL_RAE': [4, 2, 2],
            'TRAIN_CELL_RAE': [8, 4, 4],
        }

        self.OS_CFAR = {
            'RATE': 0.05,
            'PADDING_HALF_RA': [2, 1],
        }

        self.CA_CFAR_RA = {
            'FALSE_ALARM_RATE': 0.0005,
            'GUARD_CELL_RA': [4, 2],
            'TRAIN_CELL_RA': [8, 4],
            'VAL_Z': 0.5,
        }

    def _get_ca_cfar_idx_from_cube(self, cube_pw:np.array)->tuple:
        cfg_cfar_ra = self.CA_CFAR_RA
        cube_pw_norm = cube_pw/1.e+13 # preventing overflow

        nh_g_x, nh_g_y = cfg_cfar_ra["GUARD_CELL_RA"]
        nh_t_x, nh_t_y = cfg_cfar_ra["TRAIN_CELL_RA"] 
        mask_size = (2*(nh_g_x+nh_t_x)+1, 2*(nh_g_y+nh_t_y)+1) # 1 for own
        mask = np.ones(mask_size)
        mask[nh_t_x:nh_t_x+2*nh_g_x+1, nh_t_y:nh_t_y+2*nh_g_y+1] = 0
        num_total_train_cells = np.count_nonzero(mask)
        mask = mask/num_total_train_cells
        
        conv_out = ndimage.convolve(cube_pw_norm, mask, mode='constant')
        alpha = num_total_train_cells * (cfg_cfar_ra["FALSE_ALARM_RATE"]**(-1/num_total_train_cells)-1)
        conv_out = alpha * conv_out
        bool_cfar_target = np.greater(cube_pw_norm, conv_out)
        
        return np.where(bool_cfar_target==True)

    def ca_cfar(self, tesseract):
        # tesseract_norm = tesseract.copy()

        # nh_g, nh_t = self.n_half_guard_cell_zyx[0], self.n_half_train_cell_zyx[0]
        # mask_size = (2 * (nh_g + nh_t) + 1, 2 * (nh_g + nh_t) + 1)  # 1 for own
        # mask = np.ones(mask_size)
        # mask[nh_t:nh_t + 2 * nh_g + 1, nh_t:nh_t + 2 * nh_g + 1] = 0
        # num_total_train_cells = np.count_nonzero(mask)
        # mask = mask / num_total_train_cells

        # alpha = num_total_train_cells * (self.fa_rate ** (-1 / num_total_train_cells) - 1)
        
        # conv_out = ndimage.convolve(tesseract_norm, mask, mode='mirror')
        # conv_out = alpha * conv_out

        # out = np.greater(tesseract_norm, conv_out)
        # pc_idx = np.where(out == True)
        # correp_power = tesseract[pc_idx]  # Unnormalized

        # ### To point cloud ###
        # min_x, min_y = self.min_values[:2]
        # indices_x, indices_y = pc_idx
        # pc_x = min_x + indices_x * self.grid_size + self.grid_size / 2.0
        # pc_y = min_y + indices_y * self.grid_size + self.grid_size / 2.0
        
        # total_values = np.concatenate([pc_x.reshape(-1, 1), pc_y.reshape(-1, 1), correp_power.reshape(-1, 1)], axis=1)
        # # Filter the power is -1.
        # total_values = np.array(list(filter(lambda x: x[2] != -1.0, total_values.tolist())))
        # ### To point cloud ###

        # if self.mode == 0:
        #     return total_values  # X, Y, PW
        # elif self.mode == 1:
        #     return pc_idx
        # elif self.mode == 2:
        #     return total_values, pc_idx

        local_maxima_reference_cells = 5
        cfar_factor = 0.99
        cfar_offset = 20
        cfar_reference_cells = 30
        cfar_guard_cells = 5
        power_threshold = -150

        # Apply maximum filter to find local maxima
        data_max = filters.maximum_filter(tesseract,
                                        size=(local_maxima_reference_cells, local_maxima_reference_cells),
                                        mode="mirror")

        local_maxima = np.array(np.where(tesseract == data_max)).astype(int)

        predetections = []
        rows, cols = local_maxima.shape
        filtered_indices = []

        for c in range(cols):
            index_r, index_c = local_maxima[:, c]

            # Ensure indices are within bounds
            r_start = max(index_r - cfar_reference_cells - cfar_guard_cells, 0)
            r_end = max(index_r - cfar_guard_cells, 0)
            c_start = max(index_c - cfar_reference_cells - cfar_guard_cells, 0)
            c_end = max(index_c - cfar_guard_cells, 0)

            # Calculate mean of reference cells
            reference_cells = tesseract[r_start:r_end, c_start:c_end]
            if reference_cells.size == 0:
                continue

            mean = np.mean(reference_cells)
            cell_under_test = tesseract[index_r, index_c]

            # Apply CFAR thresholding
            if cell_under_test > (mean * cfar_factor + cfar_offset) and cell_under_test > power_threshold:
                filtered_indices.append([index_r, index_c])

        predetections = np.array(filtered_indices)

        return predetections
        # # generating 3D mask
        # nh_g_z, nh_g_y, nh_g_x = self.n_half_guard_cell_zyx
        # nh_t_z, nh_t_y, nh_t_x = self.n_half_train_cell_zyx
        # mask_size = (
        #     2 * (nh_g_z + nh_t_z) + 1,
        #     2 * (nh_g_y + nh_t_y) + 1,
        #     2 * (nh_g_x + nh_t_x) + 1,
        # )  # 1 for own
        # mask = np.ones(mask_size)
        # mask[
        #     nh_t_z : nh_t_z + 2 * nh_g_z + 1,
        #     nh_t_y : nh_t_y + 2 * nh_g_y + 1,
        #     nh_t_x : nh_t_x + 2 * nh_g_x + 1,
        # ] = 0
        # num_total_train_cells = np.count_nonzero(mask)
        # mask = mask / num_total_train_cells

        # alpha = num_total_train_cells * (
        #     self.fa_rate ** (-1 / num_total_train_cells) - 1
        # )

        # conv_out = ndimage.convolve(cube_norm, mask, mode="mirror")
        # conv_out = alpha * conv_out

        # out = np.greater(cube_norm, conv_out)
        # pc_idx = np.where(out == True)
        # correp_power = cube[pc_idx]  # Unnormalized

        # ### To point cloud ###
        # min_z, min_y, min_x = self.min_values
        # indices_z, indices_y, indices_x = pc_idx
        # pc_z = min_z + indices_z * self.grid_size + self.grid_size / 2.0
        # pc_y = min_y + indices_y * self.grid_size + self.grid_size / 2.0
        # pc_x = min_x + indices_x * self.grid_size + self.grid_size / 2.0

        # total_values = np.concatenate(
        #     [
        #         pc_x.reshape(-1, 1),
        #         pc_y.reshape(-1, 1),
        #         pc_z.reshape(-1, 1),
        #         correp_power.reshape(-1, 1),
        #     ],
        #     axis=1,
        # )
        # # fliter the power is -1.
        # total_values = np.array(
        #     list(filter(lambda x: x[3] != -1.0, total_values.tolist()))
        # )
        # ### To point cloud ###
        
        # if self.mode == 0:
        #     return total_values  # X, Y, Z, PW
        # elif self.mode == 1:
        #     return pc_idx
        # elif self.mode == 2:
        #     return total_values, pc_idx

    # def os_cfar(self, cube):
    #     invalid_idx = np.where(cube == -1.0)

    #     # normalize cube & make power level realy high or (set as representive value: mean) to invalid idx to make threshold high to edge part
    #     cube_norm = cube.copy()
    #     cube_norm[invalid_idx] = 0
    #     cube_norm = (cube_norm) / 1e13
    #     # cube_norm[invalid_idx] = self.LARGE_VALUE
    #     cube_norm[invalid_idx] = np.mean(cube_norm)

    #     # generating 3D mask
    #     nh_g_z, nh_g_y, nh_g_x = self.n_half_guard_cell_zyx
    #     nh_t_z, nh_t_y, nh_t_x = self.n_half_train_cell_zyx

    #     margin_z = nh_g_z + nh_t_z
    #     margin_y = nh_g_y + nh_t_y
    #     margin_x = nh_g_x + nh_t_x

    #     out = np.zeros_like(cube)

    #     n_z, n_y, n_x = out.shape

    #     for idx_z in range(margin_z, n_z - margin_z):
    #         for idx_y in range(margin_y, n_y - margin_y):
    #             for idx_x in range(margin_x, n_x - margin_x):
    #                 mask = cube_norm[
    #                     idx_z - margin_z : idx_z + margin_z + 1,
    #                     idx_y - margin_y : idx_y + margin_y + 1,
    #                     idx_x - margin_x : idx_x + margin_x + 1,
    #                 ].copy()
    #                 mask[
    #                     nh_t_z : nh_t_z + 2 * nh_g_z + 1,
    #                     nh_t_y : nh_t_y + 2 * nh_g_y + 1,
    #                     nh_t_x : nh_t_x + 2 * nh_g_x + 1,
    #                 ] = -1
    #                 arr = mask[np.where(mask != -1.0)]
    #                 thr = np.quantile(arr, 1 - self.thr_rate)
    #                 out[idx_z, idx_y, idx_x] = (
    #                     1 if cube_norm[idx_z, idx_y, idx_x] > thr else 0
    #                 )

    #     pc_idx = np.where(out == 1)
    #     correp_power = cube[pc_idx]  # Unnormalized

    #     ### To point cloud ###
    #     min_z, min_y, min_x = self.min_values
    #     indices_z, indices_y, indices_x = pc_idx
    #     pc_z = min_z + indices_z * self.grid_size + self.grid_size / 2.0
    #     pc_y = min_y + indices_y * self.grid_size + self.grid_size / 2.0
    #     pc_x = min_x + indices_x * self.grid_size + self.grid_size / 2.0

    #     total_values = np.concatenate(
    #         [
    #             pc_x.reshape(-1, 1),
    #             pc_y.reshape(-1, 1),
    #             pc_z.reshape(-1, 1),
    #             correp_power.reshape(-1, 1),
    #         ],
    #         axis=1,
    #     )
    #     # fliter the power is -1.
    #     total_values = np.array(
    #         list(filter(lambda x: x[3] != -1.0, total_values.tolist()))
    #     )
    #     ### To point cloud ###

    #     if self.mode == 0:
    #         return total_values  # X, Y, Z, PW
    #     elif self.mode == 1:
    #         return pc_idx
    #     elif self.mode == 2:
    #         return total_values, pc_idx


class KRadarProcessor:
    """K-Radar dataset preprocessor.

    Arguments:
        version: Dataset version. One of either mini or '',
            where '' is the full dataset.
        revision: Dataset revision. One of either '' or 'v2'.
        categories: Category mapping of the dataset classes.
            Maps a dataset class to a numerical category.
        dtype: Global data type used for the preprocessing
            (must be numpy compatible).
    """

    def __init__(
        self,
        version: str = "",
        revision: str = "",
        categories: Dict[str, int] = None,
        road_structures: Dict[str, int] = None,
        weather_conditions: Dict[str, int] = None,
        time_zone: Dict[str, int] = None,
        workers: int = 1,
        dtype: str = "float32",
        **kwargs,
    ):
        self.version = version
        self.revision = revision
        self.categories = categories
        self.road_structures = road_structures
        self.weather_conditions = weather_conditions
        self.time_zone = time_zone
        self.workers = workers
        self.dtype = dtype

        # Define dataset splits (based on the version)
        self.splits = ["train", "val", "test"]
        if self.version:
            self.splits = [f"{self.version}_{s}" for s in self.splits]

        # Set desired jpg quality
        self.jpg_quality = [cv2.IMWRITE_JPEG_QUALITY, 98]

    @property
    def categories(self):
        return self._categories

    @categories.setter
    def categories(self, value: Dict[str, int]):
        if value is None:
            # Default categories of the K-Radar dataset
            self._categories = {
                "Sedan": 0,
                "Bus or Truck": 1,
                "Motorcycle": 2,
                "Bicycle": 3,
                "Bicycle Group": 4,
                "Pedestrian": 5,
                "Pedestrian Group": 6,
                "Background": 7,
            }

        elif len(value) != 8:
            raise ValueError(
                f"The categories property must provide a unique mapping "
                f"for each of the 8 classes but an input with "
                f"{len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._categories = value

        else:
            raise TypeError(
                f"The categories property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def road_structures(self):
        return self._road_structures

    @road_structures.setter
    def road_structures(self, value: Dict[str, int]):
        if value is None:
            # Default road structures of the K-Radar dataset
            self._road_structures = {
                "urban": 0,
                "highway": 1,
                "alleyway": 2,
                "suburban": 3,
                "university": 4,
                "mountain": 5,
                "parking_lots": 6,
                "parkinglots": 6,
                "shoulder": 7,
                "countryside": 8,
            }

        elif len(value) != 8:
            raise ValueError(
                f"The road structures property must provide a unique mapping "
                f"for each of the 8 road structures but an input with "
                f"{len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._road_structures = value

        else:
            raise TypeError(
                f"The road structures property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def weather_conditions(self):
        return self._weather_conditions

    @weather_conditions.setter
    def weather_conditions(self, value: Dict[str, int]):
        if value is None:
            # Default weather conditions of the K-Radar dataset
            self._weather_conditions = {
                "normal": 0,
                "overcast": 1,
                "fog": 2,
                "rain": 3,
                "sleet": 4,
                "light_snow": 5,
                "lightsnow": 5,
                "heavy_snow": 6,
                "heavysnow": 6,
            }

        elif len(value) != 7:
            raise ValueError(
                f"The weather conditions property must provide a unique "
                f"mapping for each of the 7 weather conditions but an input "
                f"with {len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._weather_conditions = value

        else:
            raise TypeError(
                f"The weather conditions property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def time_zone(self):
        return self._time_zone

    @time_zone.setter
    def time_zone(self, value: Dict[str, int]):
        if value is None:
            # Default time zones of the K-Radar dataset
            self._time_zone = {
                "day": 0,
                "night": 1,
            }

        elif len(value) != 2:
            raise ValueError(
                f"The time zone property must provide a unique mapping "
                f"for each of the 2 time zones but an input with"
                f"{len(value)} elements was given!"
            )

        elif isinstance(value, dict):
            self._time_zone = value

        else:
            raise TypeError(
                f"The time zone property must be of type 'dict' "
                f"but an input of type {type(value)} was given!"
            )

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        self._workers = 1 if value < 1 else value

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, value: str):
        self._dtype = np.dtype(value)

    @classmethod
    def from_config(cls, config: Dict) -> KRadarProcessor:  # noqa: F821
        return cls(**dict(config["computing"] | config["data"]))

    def __call__(self, *args: Any, **kwargs: Any):
        self.prepare(*args, **kwargs)

    @staticmethod
    def get_data_indices(label_path: str) -> Tuple[str, ...]:
        """Returns the indices of the sensor data belonging to a label.

        Arguments:
            label_path: Path to the label info file in question.

        Returns:
            seq_idx: Index of the associated sequence.
            radar_idx: Index of the associated radar data.
            os2_idx: Index of the associated os2-64 lidar data.
            camf_idx: Index of the associated front camera data.
            os1_idx: Index of the associated os1-128 lidar data.
            camlrr_idx: Index of the associated left, right, rear camera data.
        """
        with open(label_path, "r") as f:
            line = f.readline()

        seq_idx = label_path.split("/")[-3]
        radar_idx, os2_idx, camf_idx, os1_idx, camlrr_idx = (
            line.split(",")[0].split("=")[1].split("_")
        )

        return seq_idx, radar_idx, os2_idx, camf_idx, os1_idx, camlrr_idx

    @staticmethod
    def get_description(filename: str) -> List[str]:
        """Returns the sequence description tags from a given description filename.

        Arguments:
            filename: Filename of the sequence description file.

        Returns:
            List of sequence description tags.
        """
        with open(filename, "r") as f:
            line = f.readline()

        road_type, capture_time, climate = line.split(",")

        return [road_type, capture_time, climate]

    def get_dataset_paths(self, src_list: List) -> Dict[str, Dict[str, List[str]]]:
        """Returns the paths of all train and test labels.

        These files serve as central information to link sensor data
        to labels and calibration information. The data is structured
        in a dictionary split into test and train and structured by
        sequence number.

        Arguments:
            src: Source path to the dataset folder.

        Returns:
            dataset_paths: Dictionary of paths to the individual
                label_info files of each sample.
        """
        # Initialize dataset paths
        dataset_paths = {split: {} for split in self.splits}

        # Add revision postfix
        info_label = f"info_label_{self.revision}" if self.revision else "info_label"

        # List all sequences in the dataset
        for src in src_list:
            for seq in os.listdir(src):
                # List all samples in the sequence
                samples = set(glob(osp.join(src, seq, info_label, "*.txt")))

                # Filter all samples according to split
                for s in self.splits:
                    # Get current split
                    c_split = getattr(split, s)

                    # Filter dataset paths
                    dataset_paths[s][seq] = sorted(
                        list(
                            filter(
                                lambda x: f"{seq}_{osp.splitext(osp.basename(x))[0]}"
                                in c_split,
                                samples,
                            )
                        )
                    )

        return dataset_paths

    def get_sequence_paths(
        self, sequence: List[str]
    ) -> Dict[str, Union[List[str], Dict[str, str]]]:
        """Returns the paths of all data belonging to a sequence.

        Arguments:
            sequence: List of label info paths belonging to a sequence.

        Returns:
            sequence_paths: Dictionary of sensors data, calibration
                and label file paths strctured by sample id.
        """
        sequence_paths = {}

        for sample in sequence:
            # Get base path of the sequence
            base_path = osp.abspath(osp.join(osp.dirname(sample), os.pardir))

            # Get sample id
            sample_id = osp.splitext(osp.basename(sample))[0]

            # Get sensor indices beloning the sample
            _, radar_idx, os2_idx, camf_idx, os1_idx, _ = self.get_data_indices(sample)

            # Construct sensor data paths
            sequence_paths[sample_id] = {}
            sequence_paths[sample_id]["label"] = sample
            sequence_paths[sample_id]["calib_radar_lidar"] = osp.join(
                base_path, "info_calib", "calib_radar_lidar.txt"
            )
            sequence_paths[sample_id]["calib_camera_lidar"] = osp.join(
                base_path, "info_calib", "calib_camera_lidar.txt"
            )
            sequence_paths[sample_id]["camera_front"] = osp.join(
                base_path, "cam-front", "cam-front_" + camf_idx + ".png"
            )
            sequence_paths[sample_id]["radar_tesseract"] = osp.join(
                base_path, "radar_tesseract", "tesseract_" + radar_idx + ".mat"
            )
            sequence_paths[sample_id]["os1"] = osp.join(
                base_path, "os1-128", "os1-128_" + os1_idx + ".pcd"
            )
            sequence_paths[sample_id]["os2"] = osp.join(
                base_path, "os2-64", "os2-64_" + os2_idx + ".pcd"
            )

        if sequence:
            description_file = osp.join(base_path, "description.txt")
            sequence_paths["description"] = self.get_description(description_file)

        return sequence_paths

    @cache
    def get_camera_calibration(self, filename: str) -> np.ndarray:
        """Retruns a homogeneous transformation matrix from the given calibration file.

        Note: Stereo camera transformation is estimated due to missing calibration
        information.

        Arguments:
            filename: Filename of the calibration file.

        Returns:
            calibration: A homogeneous (4x4) transformation matrix.
        """
        # Load calibration file data
        with open(filename, "r") as f:
            lines = f.readlines()

        # Initialize homogeneous transformation matrix
        calibration_left = np.eye(4, dtype=self._dtype)

        # Assign camera transformation matrix
        calibration_left[:3, :] = np.array(
            list(map(float, lines[1].split(",")))
        ).reshape((3, 4))

        # Define stereo camera baseline according to spec sheet
        B = 0.12

        # Construct stereo camera transformation matrix (Tx = -fx * B)
        calibration_right = deepcopy(calibration_left)
        calibration_right[0, 3] += -calibration_right[0, 0] * B

        return calibration_left, calibration_right

    @cache
    def get_radar_calibration(self, filename: str) -> np.ndarray:
        """Retruns a homogeneous transformation matrix from the given calibration file.

        Arguments:
            filename: Filename of the calibration file.

        Returns:
            calibration: A homogeneous (4x4) transformation matrix.
        """
        # Load calibration file data
        with open(filename, "r") as f:
            lines = f.readlines()

        # Initialize homogeneous transformation matrix
        calibration = np.eye(4, dtype=self._dtype)

        # Initialize translation vector
        translation = np.zeros(3, dtype=self._dtype)

        # Map translation information from (frame difference, dx, dy) to (dx, dy, dz=0)
        translation[:2] = np.array(list(map(float, lines[1].split(",")))[-2:])
        calibration[:3, -1] = translation.T

        # Initialize transformation matrices for ra and ea projection
        T_ra = deepcopy(calibration)
        T_ea = deepcopy(calibration)

        return T_ra, T_ea

    @cache
    def get_translation(self, filename: str) -> np.ndarray:
        """Retruns a homogeneous transformation matrix from the given calibration file.

        This transformation matrix represents a translation without any rotation.

        Arguments:
            filename: Filename of the calibration file.

        Returns:
            calibration: A homogeneous (4x4) transformation matrix.
        """
        # Load calibration file data
        with open(filename, "r") as f:
            lines = f.readlines()

        # Initialize homogeneous transformation matrix
        calibration = np.eye(4, dtype=self._dtype)

        # Initialize translation vector
        translation = np.zeros(3, dtype=self._dtype)

        # Map translation information from (frame difference, dx, dy) to (dx, dy, dz=0)
        translation[:2] = np.array(list(map(float, lines[1].split(",")))[-2:])
        calibration[:3, -1] = translation.T

        return calibration

    @staticmethod
    def _transform_boxes(boxes: np.array, transformation: np.array) -> np.array:
        """Transfroms an array of bounding boxes by a given transformation.

        Arguments:
            boxes: Bounding boxes given as array with size (M, 9). Where M
                is the number of bounding boxes and 9 coorespodes to
                [x, y, z, theta, l, w, h, category index, object id].
            transformation: A homogeneous (4x4) transformation matrix.

        Returns:
            boxes: Array of transformed bounding boxes with size (M, 9).
        """
        # Transform bounding box center (translation)
        boxes[:, :3] = np.einsum(
            "ij,...j->...i",
            transformation,
            np.column_stack((boxes[:, :3], np.ones(boxes.shape[0]))),
        )[:, :3]

        # TODO: Transform bounding box heading (rotation)

        return boxes

    def get_boxes(self, filename: str) -> np.array:
        """Returns an array of bounding boxes from a given label file.

        Bounding box format: [x, y, z, theta, l, w, h, category index, object id]
            - x: Bounding box center position along the x-axis in m
            - y: Bounding box center position along the y-axis in m
            - z: Bounding box center position along the z-axis in m
            - theta: Bounding box heading (rotation around the z-axis) in deg
            - l: Bounding box lenght (extent along the x-axis) in m
            - w: Bounding box width (extent along the y-axis) in m
            - h: Bounding box height (extent along the z-axis) in m
            - category index: Category affiliation accoring to the class matching
            - object id: Unique object identifier across a sequence

        Arguments:
            filename: Filename of the label file.

        Returns:
            boxes: Array of bounding boxes (M, 9), with M being the number of boxes.
        """
        # Load label data
        with open(filename, "r") as f:
            lines = f.readlines()

        # Initialize bounding box array
        boxes = np.zeros([len(lines[1:]), 9], dtype=self._dtype)

        # Parse label data (skip first info line)
        for i, line in enumerate(lines[1:]):
            values = line.split(",")

            # Skip invalid labels (header missing)
            if values[0] != "*":
                continue

            # There a two types of label formats
            if len(values) == 10:
                _, obj_id, class_name, x, y, z, theta, l, w, h = values
            else:
                _, _, obj_id, class_name, x, y, z, theta, l, w, h = values

            # Map class names to category indices
            category_idx = self.categories[class_name.strip()]

            # Skip invalid categories (category index -1)
            if category_idx < 0:
                continue

            # Assign to bounding box array
            boxes[i, :] = np.array(
                [
                    float(x),
                    float(y),
                    float(z),
                    np.deg2rad(float(theta)),
                    2 * float(l),
                    2 * float(w),
                    2 * float(h),
                    category_idx,
                    obj_id,
                ]
            )

        # Filter invalid bounding boxes and return valid boxes
        return boxes[~np.all(boxes == 0, axis=1)]

    def get_camera_data(self, filename: str) -> Tuple[np.array, np.array]:
        """Returns two rgb matrices from a given stereo image filename.

        Arguments:
            filename: Stereo image data filename.

        Retruns:
            left: Array of rgb values from the left camera (h, w, 3).
            right: Array of rgb values from the right camera (h, w, 3).
        """
        # Load image data from file
        image = cv2.imread(filename)

        # Split stereo image into left and right image
        left, right = np.split(image, 2, axis=1)

        return left, right

    def get_lidar_data(self, filename: str) -> np.ndarray:
        """Returns a lidar point cloud from a given pcd filename.

        Lidar point cloud with shape (N, 9) and fileds:
        x, y, z, intensity, t, reflectivity, ring, ambient, range.

        Arguments:
            filename: Lidar data pcd filename.

        Returns:
            point_cloud: Lidar point cloud with
                shape (N, 9).
        """
        # Load lidar point cloud
        pc = pypcd.PointCloud.from_path(filename)
        pc_data = pc.pc_data

        # Convert point cloud to array
        point_cloud = np.array(
            [
                pc_data["x"],
                pc_data["y"],
                pc_data["z"],
                pc_data["intensity"],
                pc_data["t"],
                pc_data["reflectivity"],
                pc_data["ring"],
                pc_data["ambient"],
                pc_data["range"],
            ],
            dtype=self._dtype,
        ).T

        # Filter out missing values
        point_cloud = point_cloud[np.where(np.abs(point_cloud[:, 0]) > 0.01)]

        return point_cloud

    def get_radar_tesseract(self, filename: str) -> np.array:
        """Returns the raw 4D radar tesseract.

        Arguments:
            filename: Filename of the 4D radar tesseract mat file.

        Returns:
            tesseract: 4D radar tesseract with shape
                (doppler, range, elevation, azimuth)
        """
        # Load radar tesseract
        tesseract: np.ndarray = loadmat(filename)["arrDREA"]

        return tesseract.astype(self._dtype)

    def get_radar_data(self, filename: str) -> np.array:
        """Returns the RA and EA projection of the 4D radar tesseract.

        Arguments:
            filename: Filename of the 4D radar tesseract mat file.

        Returns:
            ra: Range-Azimuth projection of the 4D radar tesseract.
            ea: Elevation-Azimuth projection of the 4D radar tesseract.
        """
        try:
            # Load radar tesseract with shape (doppler, range, elevation, azimuth)
            tesseract = self.get_radar_tesseract(filename)

            tesseract = np.array(tesseract)
            # Convert radar responce to dB
            tesseract = 10 * np.log10(tesseract)

            # Reduce to range-azimuth plane
            ra_rcs_max = np.max(np.max(tesseract, axis=2), axis=0)
            ra_rcs_median = np.median(np.median(tesseract, axis=2), axis=0)
            ra_rcs_var = np.var(np.var(tesseract, axis=2), axis=0)

            ra_doppler_max_idx = np.argmax(np.max(tesseract, axis=2), axis=0)
            ra_doppler_max = np.asarray(radar_info.doppler_raster)[ra_doppler_max_idx]
            ra_doppler_median = np.median(np.max(tesseract, axis=2), axis=0)
            ra_doppler_var = np.var(np.max(tesseract, axis=2), axis=0)

            # Crop radar tesseract (to 4:252) in the range dimension due to fft artifacts
            tesseract = tesseract[:, 4:252, :, :]

            # Reduce to elevation-azimuth plane
            ea_rcs_max = np.max(np.max(tesseract, axis=1), axis=0)
            ea_rcs_median = np.median(np.median(tesseract, axis=1), axis=0)
            ea_rcs_var = np.var(np.var(tesseract, axis=1), axis=0)

            ea_doppler_max_idx = np.argmax(np.max(tesseract, axis=1), axis=0)
            ea_doppler_max = np.asarray(radar_info.doppler_raster)[ea_doppler_max_idx]
            ea_doppler_median = np.mean(np.max(tesseract, axis=1), axis=0)
            ea_doppler_var = np.var(np.max(tesseract, axis=1), axis=0)

            # Stack radar features
            ra = np.dstack(
                (
                    ra_rcs_max,
                    ra_rcs_median,
                    ra_rcs_var,
                    ra_doppler_max,
                    ra_doppler_median,
                    ra_doppler_var,
                )
            )
            ea = np.dstack(
                (
                    ea_rcs_max,
                    ea_rcs_median,
                    ea_rcs_var,
                    ea_doppler_max,
                    ea_doppler_median,
                    ea_doppler_var,
                )
            )

            return ra, ea
        except Exception as error:
            print("Caught this error: " + repr(error))

    def get_radar_data_revised(self, filename: str, cfg) -> np.array:
        """Returns the RA and EA projection of the 4D radar tesseract.

        Arguments:
            filename: Filename of the 4D radar tesseract mat file.

        Returns:
            ra: Range-Azimuth projection of the 4D radar tesseract.
            ea: Elevation-Azimuth projection of the 4D radar tesseract.
        """
        try:
            # Load radar tesseract with shape (doppler, range, elevation, azimuth)
            tesseract = self.get_radar_tesseract(filename)
            tesseract = np.array(tesseract)
            invalid_idx = np.where(tesseract == -1.0)
            if invalid_idx[0].size != 0:
                print("Cube has invalid values")

            if np.any(tesseract <= 0):
                print("Warning: Tesseract contains zero or negative values, which may cause log10 issues.")
                # tesseract = np.where(tesseract > 0, tesseract, np.nan)  # Replace non-positive values with NaN
            
            # Convert radar responce to dB
            # tesseract = 10 * np.log10(tesseract)
            
            RD_map = tesseract.mean(axis=(2, 3))    # D R

            # cfar(RD_map)
            roi = cfg["data"]["fov"]
            cls_cfar = CFAR(roi, type="index")
            
            predetections = cls_cfar._get_ca_cfar_idx_from_cube(RD_map)

            RD_map = 10 * np.log10(RD_map)

            predetections_v2 = cls_cfar.ca_cfar(RD_map)
            
            if len(predetections[0]) > 0:  # Check if there are any detections
                try:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(RD_map, aspect='auto', origin='lower', cmap='viridis')
                    plt.colorbar(label='Amplitude (dB)')
                    plt.xlabel('Range')
                    plt.ylabel('Doppler')
                    plt.title('2D Range-Doppler Map')

                    # Convert tuple of arrays to a 2D array for plotting
                    predetections_array = np.column_stack(predetections)
                    print(len(predetections_array), "CFAR 1 detections found")
                    print(len(predetections_v2[:, 1],), "CFAR 2 detections found")
                    # Add markers for the peaks
                    plt.scatter(predetections_array[:, 1], predetections_array[:, 0], color='red', marker='x') 
                    if len(predetections_v2) > 0:
                        plt.scatter(predetections_v2[:, 1], predetections_v2[:, 0], color='pink', marker='x')
                    # Save the entire plot with all decorations
                    plt.savefig("/app/outputs/" + filename.split("/radar_tesseract/")[0].split("/")[-1] + "_" + filename.split("tesseract_")[-1].split(".mat")[0] + "_2D_RD_map.png")
                    plt.show()
                except Exception as e:
                    print(f"{filename} failed to plot 2D RD map: {e}")
            else:
                print("No detections found")

            for rd_ind in predetections_v2:
                AE_map = tesseract[rd_ind[1], rd_ind[0], :, :]



            #### to do!
            # Reduce to range-azimuth plane
            ra_rcs_max = np.max(np.max(tesseract, axis=2), axis=0)
            ra_rcs_median = np.median(np.median(tesseract, axis=2), axis=0)
            ra_rcs_var = np.var(np.var(tesseract, axis=2), axis=0)

            ra_doppler_max_idx = np.argmax(np.max(tesseract, axis=2), axis=0)
            ra_doppler_max = np.asarray(radar_info.doppler_raster)[ra_doppler_max_idx]
            ra_doppler_median = np.median(np.max(tesseract, axis=2), axis=0)
            ra_doppler_var = np.var(np.max(tesseract, axis=2), axis=0)

            # Crop radar tesseract (to 4:252) in the range dimension due to fft artifacts
            tesseract = tesseract[:, 4:252, :, :]

            # Reduce to elevation-azimuth plane
            ea_rcs_max = np.max(np.max(tesseract, axis=1), axis=0)
            ea_rcs_median = np.median(np.median(tesseract, axis=1), axis=0)
            ea_rcs_var = np.var(np.var(tesseract, axis=1), axis=0)

            ea_doppler_max_idx = np.argmax(np.max(tesseract, axis=1), axis=0)
            ea_doppler_max = np.asarray(radar_info.doppler_raster)[ea_doppler_max_idx]
            ea_doppler_median = np.mean(np.max(tesseract, axis=1), axis=0)
            ea_doppler_var = np.var(np.max(tesseract, axis=1), axis=0)

            # Stack radar features
            ra = np.dstack(
                (
                    ra_rcs_max,
                    ra_rcs_median,
                    ra_rcs_var,
                    ra_doppler_max,
                    ra_doppler_median,
                    ra_doppler_var,
                )
            )
            ea = np.dstack(
                (
                    ea_rcs_max,
                    ea_rcs_median,
                    ea_rcs_var,
                    ea_doppler_max,
                    ea_doppler_median,
                    ea_doppler_var,
                )
            )

            return ra, ea
        except Exception as error:
            print("Caught this error: " + repr(error))

    def map_description(self, description: List[str]) -> np.ndarray:
        """Returns an encoded scene description.

        Arguments:
            description: List of scene description tags.

        Retruns:
            Array of numerical scene description values according
            to the defined mapping.
        """
        return np.array(
            [
                self._road_structures[description[0]],
                self._time_zone[description[1]],
                self._weather_conditions[description[2]],
            ],
            dtype=self._dtype,
        )

    def prepare_sample(self, sample: Dict[str, str], description, dst: str) -> None:
        """Pre-processes a single data sample and saves the results.

        Arguments:
            sample: Dictionary with all file paths belonging to a
                single sample.
            dst: Destiantion directory to save the processed sample
                data.
        """
        if os.path.exists(osp.join(dst, "ra.npy")):
            print(
                f"Directory {osp.join(dst, 'ra.npy')} already exists. Skipping sample."
            )
            return

        # Load lable data
        boxes = self.get_boxes(sample["label"])

        # Skip samples without bounding boxes
        if not boxes.size:
            return

        # Encode description
        description = self.map_description(description)

        # Load calibration data
        ra_to_lidar, ea_to_lidar = self.get_radar_calibration(
            sample["calib_radar_lidar"]
        )
        mono_to_lidar, stereo_to_lidar = self.get_camera_calibration(
            sample["calib_camera_lidar"]
        )

        # Transform bounding boxes to lidar frame
        radar_to_lidar = self.get_translation(sample["calib_radar_lidar"])
        boxes = self._transform_boxes(boxes, radar_to_lidar)

        # Load front camera data
        camera_front_left, camera_front_right = self.get_camera_data(
            sample["camera_front"]
        )

        # Load radar data (range-azimuth, elevation-azimuth)
        ra, ea = self.get_radar_data(sample["radar_tesseract"])

        # Load lidar data
        os1 = self.get_lidar_data(sample["os1"])
        os2 = self.get_lidar_data(sample["os1"])

        # Save data
        os.makedirs(dst, exist_ok=True)
        np.save(osp.join(dst, "labels.npy"), boxes, allow_pickle=False)
        np.save(osp.join(dst, "description.npy"), description, allow_pickle=False)
        cv2.imwrite(osp.join(dst, "mono.jpg"), camera_front_left, self.jpg_quality)
        np.save(osp.join(dst, "mono_info.npy"), mono_to_lidar, allow_pickle=False)
        cv2.imwrite(osp.join(dst, "stereo.jpg"), camera_front_right, self.jpg_quality)
        np.save(osp.join(dst, "stereo_info.npy"), stereo_to_lidar, allow_pickle=False)
        np.save(osp.join(dst, "ra.npy"), ra, allow_pickle=False)
        np.save(osp.join(dst, "ra_info.npy"), ra_to_lidar, allow_pickle=False)
        np.save(osp.join(dst, "ea.npy"), ea, allow_pickle=False)
        np.save(osp.join(dst, "ea_info.npy"), ea_to_lidar, allow_pickle=False)
        np.save(osp.join(dst, "os1.npy"), os1, allow_pickle=False)
        np.save(osp.join(dst, "os2.npy"), os2, allow_pickle=False)

    def prepare_radar_sample(
        self, sample: Dict[str, str], description, dst: str, cfg=None
    ) -> None:
        """Pre-processes a single radar data sample and saves the results.

        Arguments:
            sample: Dictionary with all file paths belonging to a
                single sample.
            dst: Destiantion directory to save the processed sample
                data.
        """

        # if os.path.exists(osp.join(dst, "ra.npy")):
        #     print(f"Directory {osp.join(dst, "ra.npy")} already exists. Skipping sample.")
        #     return
        # Load radar data (range-azimuth, elevation-azimuth)
        ra, ea = self.get_radar_data_revised(sample["radar_tesseract"], cfg)

        # Save data
        # os.makedirs(dst, exist_ok=True)
        # np.save(osp.join(dst, "ra.npy"), ra, allow_pickle=False)
        # np.save(osp.join(dst, "ea.npy"), ea, allow_pickle=False)

    def prepare_sequence(self, sequence: List[str], dst: str, cfg=None) -> None:
        """Pre-processes a single sequence by sample.

        Arguments:
            sequence: List of all sample path belonging
                to a single sequence.
            dst: Destination path to save the processed
                results of the sequence.
        """
        # Get data, label and calibration file paths for each sample
        sequence_paths = self.get_sequence_paths(sequence)

        # Separate sequence description form samples
        description = sequence_paths.pop("description")

        # Execute sample processing concurrently
        with ThreadPoolExecutor(max_workers=self._workers) as e:
            e.map(
                lambda item: self.prepare_radar_sample(
                    item[1], description, osp.join(dst, item[0]), cfg
                ),
                sequence_paths.items(),
            )

    def prepare(self, src_list: list, dst: str, cfg=None) -> None:
        """Pre-processes and saves the data of the given dataset.

        Arguments:
            src: Source path of the kradar dataset folder.
            dst: Destination path to save the processed dataset.
        """
        # Get dataset path
        dataset_paths = self.get_dataset_paths(src_list)

        # Get length of the full dataset
        full = f"{self.version}_full" if self.version else "full"
        total = len(getattr(split, full))

        with tqdm(total=total, desc="Processing dataset") as pbar:
            for s in self.splits:
                # Prepare data split
                for seq_id, sequence in dataset_paths[s].items():
                    # Check available memory in dst
                    statvfs = os.statvfs(dst)
                    free_space = statvfs.f_frsize * statvfs.f_bavail

                    # Estimate required space (this is a rough estimate, adjust as needed)
                    required_space = 3e9  # Example: 1GB required space

                    if free_space < required_space:
                        print(
                            f"Not enough space in {dst}. Required: {required_space}, Available: {free_space}. Stopping."
                        )
                        return

                    # if (
                    #     not os.path.isdir(osp.join(dst, s, seq_id))
                    #     and len(sequence) > 0
                    # ):
                    #     self.prepare_sequence(sequence, osp.join(dst, s, seq_id))
                    # else:
                    #     print(f"Skipping sequence {seq_id} in split {s}.")
                    if len(sequence) > 0:
                        self.prepare_sequence(sequence, osp.join(dst, s, seq_id), cfg)

                    # Update progress bar
                    pbar.update(len(sequence))


def prepare_kradar(*args, **kwargs):
    return KRadarProcessor.from_config(*args, **kwargs)
