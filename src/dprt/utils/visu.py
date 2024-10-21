from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from matplotlib.colors import (
    Colormap,
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
)
from matplotlib.cm import ScalarMappable

from dprt.utils.geometry import get_box_corners, get_transformation
from dprt.utils.project import cart2spher, polar2cart, spher2cart


from scipy.spatial import Delaunay
import copy
import torch

# from ops.roiaware_pool3d import roiaware_pool3d_utils

# Define TUM color map
TUMCM = LinearSegmentedColormap.from_list(
    "tum", [[0.0, 0.2, 0.34901960784313724], [1.0, 1.0, 1.0]], N=100
)


def get_tum_accent_cm() -> Colormap:
    return ListedColormap(
        np.array([[162, 173, 0], [227, 114, 34], [152, 198, 234], [218, 215, 203]])
        / 255
    )


def scalar2rgba(
    scalars: np.ndarray, cm: Colormap = None, norm: bool = True
) -> np.ndarray:
    """Returns RGBA values for scalar input values accoring to a colormap.

    Arguments:
        scalars: Scalar values with shape (n,) to map to
            RGBA values.
        cm: Colormap to map the scalar values to.
        norm: Whether to normalize the scalar values.

    Returns:
        rgba: Red, green, blue, alpha values with
            shape (n, 4) for all scalar values.
    """
    # Get data normalization function
    if norm:
        norm = Normalize(vmin=np.min(scalars), vmax=np.max(scalars), clip=True)
    else:
        norm = None

    # Define color map
    mapper = ScalarMappable(norm=norm, cmap=cm)

    # Map scalars to rgba values
    rgba = mapper.to_rgba(scalars.flatten())

    return rgba


def visu_camera_data(img, dst: str = None) -> None:
    """Visualizes a given image.

    Arguments:
        img: Image data with shape (w, h, 3).
        dst: Destination filename to save the figure
            to. If provided, the figure will not be
            displayed but only saved to file.
    """
    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Plot image to axes
    ax.imshow(img)

    if dst is not None:
        fig.savefig(dst)
    else:
        fig.show()


def visu_lidar_data(
    pc: np.ndarray,
    boxes: np.ndarray = None,
    xlim: Tuple[float, float] = None,
    ylim: Tuple[float, float] = None,
    cm: Colormap = None,
) -> None:
    """Visualizes a given lidar point cloud.

    Arguments:
        pc: Lidar point cloud with shape (N, ...).
            The second dimension of the point cloud
            can be of arbitrary size as long as the
            first 4 values represent the x, y, z and
            intensity values.
        boxes: Bounding boxes given as numerical array of shape (M, ...).
            The second dimension can be of arbitrary size as long as the
            x, y, z, theta, l, w, h, class values represent the first 8
            values of this dimension and are provided in the right order.
        xlim: Tuple (min, max) of x-coordinate limits
            to restrict the point cloud to.
        ylim: Tuple (min, max) of y-coordinate limits
            to restrict the point cloud to.
        cm: Color map to visualize the point
            cloud values.
    """
    # Get colormap
    if cm is None:
        cm = TUMCM

    # Limit the point cloud
    if xlim is not None:
        pc = pc[np.logical_and(pc[:, 0] > xlim[0], pc[:, 0] < xlim[1])]

    if ylim is not None:
        pc = pc[np.logical_and(pc[:, 1] > ylim[0], pc[:, 1] < ylim[1])]

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

    # Get colors based on intensity
    rgb = scalar2rgba(pc[:, 3], cm=cm, norm=True)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Create a visualization object and window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Display lidar point cloud
    vis.add_geometry(pcd)

    # Add bounding boxes
    if boxes is not None:
        cm = get_tum_accent_cm()

        for box in boxes:
            # Create bounding box
            bbox = o3d.geometry.OrientedBoundingBox()
            bbox.center = box[:3]
            bbox.extent = box[4:7]
            bbox.R = get_transformation(
                rotation=np.array([0.0, 0.0, box[3]]), degrees=True
            )[:3, :3]
            bbox.color = np.asarray(cm(int(box[7])))[:3]

            # Visualize bounding box
            vis.add_geometry(bbox)

    # Block thread until window is closed
    vis.run()


def visu_2d_lidar_points(
    ax: plt.Axes,
    points: np.ndarray,
    dims: Tuple[int, int],
    roi: Tuple[float, float, float, float] = None,
    cart: bool = True,
    r_max: float = None,
    flip: bool = True,
) -> None:
    """Visualizes the given points on the specified axis.

    Arguments:
        ax: Axis to visualize the points on.
        points: Points with shape (M, ...). The second
            dimension can be of arbitrary size as long as
            the first 4 values represent the x, y, z,
            intensity values.
        roi: Region of interest given as min and max
            values for both dimensions. The order is
            (min1, max1, min2, max2).
        cart: Wheter to visualize the provided points in
            cartesian coordinates or polar coordinates.
    """
    # Remove everything except x, y, z and intensity
    points = points[:, :4]

    # Limit point cloud to a given region of interest
    if roi is not None:
        points[:, 0], points[:, 1], points[:, 2] = cart2spher(
            points[:, 0], points[:, 1], points[:, 2], degrees=True
        )
        points[:, 0] = (
            np.ones_like(points[:, 0]) * r_max if r_max is not None else points[:, 0]
        )
        points = points[
            np.logical_and(points[:, dims[0]] > roi[0], points[:, dims[0]] < roi[1])
        ]
        points = points[
            np.logical_and(points[:, dims[1]] > roi[2], points[:, dims[1]] < roi[3])
        ]
        points[:, 0], points[:, 1], points[:, 2] = spher2cart(
            points[:, 0], points[:, 1], points[:, 2], degrees=True
        )

    if not cart:
        points[:, 0], points[:, 1], points[:, 2] = cart2spher(
            points[:, 0], points[:, 1], points[:, 2], degrees=True
        )
        points[:, 0] = (
            np.ones_like(points[:, 0]) * r_max if r_max is not None else points[:, 0]
        )

    if flip:
        ax.scatter(points[:, dims[0]], points[:, dims[1]], s=0.2, c="black")
    else:
        points[:, dims[0]] *= -1
        ax.scatter(points[:, dims[0]], points[:, dims[1]], s=0.2, c="black")


def visu_3d_radar_data(
    cube: np.ndarray,
    dims: str,
    raster: List[np.ndarray] = None,
    cart: bool = False,
    cm: Colormap = None,
    **kwargs,
) -> None:
    """Visualizes a given 3D radar cube.

    Arguments:
        cube: 3D cube representing a section of the 4D radar
            tesseract with shape (N, M, K).
        raster: Rasterization values (grid points) of
            of the associated radar dimensions with
            shape (N, M, K).
        cart: Wheter to project the provided
            grid values from polar to cartesian
            coordinates. If True, the second
            raster dimension has to represent the
            angular values.
        cm: Color map to visualize the gird values.
        jv: Whether to use a jupyter visualizer for
            visualizations within a jupyter notebook.
    """
    # Check input dimensions
    if cart and dims != "rae":
        raise ValueError(
            f"A cartesian transformation is only possible "
            f"if the data is provided in a 'rae' order. "
            f"However, the data was given as {dims}."
        )

    # Mesh grid based on sensor specifications
    if raster is not None:
        x, y, z = np.meshgrid(raster[0], raster[1], raster[2], indexing="ij")
    else:
        x, y, z = np.meshgrid(
            np.arange(cube.shape[0]),
            np.arange(cube.shape[1]),
            np.arange(cube.shape[2]),
            indexing="ij",
        )

    # Convert spherical to cartesian coordinate values
    if cart:
        x_shape, y_shape, z_shape = x.shape, y.shape, z.shape
        x, y, z = spher2cart(
            x.flatten("F"), y.flatten("F"), z.flatten("F"), degrees=True
        )
        x = x.reshape(x_shape, order="F")
        y = y.reshape(y_shape, order="F")
        z = z.reshape(z_shape, order="F")

    # Create point cloud representing the voxel center points
    xyz = np.zeros((np.size(x), 3))
    xyz[:, 0] = np.reshape(x, -1)
    xyz[:, 1] = np.reshape(y, -1)
    xyz[:, 2] = np.reshape(z, -1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # Get radar rcs values
    rcs = 10 * np.log10(cube)

    # Map rcs values to color
    rgb = scalar2rgba(rcs, cm=cm, norm=True)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Create voxel grid from grid points
    # voxel_size = np.min([np.min(np.diff(r)) for r in raster])
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.5)

    # Visualize 3d radar cube
    o3d.visualization.draw_geometries([voxel_grid])


def visu_2d_boxes(
    ax: plt.Axes,
    boxes: np.ndarray,
    dims: Tuple[int, int],
    cart: bool = True,
    r_max: float = None,
    flip: bool = False,
) -> None:
    """

    Order of the 2D bounding box corners.
        3------2
        |      |
        |      |
        0------1

    """
    # Get number of boxes
    M = boxes.shape[0]
    dims = sorted(list(dims))

    # Get 3D box corners form bounidng boxes
    corners3d = get_box_corners(boxes)

    # Project bounding boxes to 2D (but maintain the third coordinate for transformation)
    if 0 in dims:
        corners2d = corners3d[:, :4, :]
    else:
        corners2d = np.zeros((M, 4, 3))
        corners2d[:, 0, :] = corners3d[
            np.arange(M), np.argmin(corners3d[:, :4, dims[0]], axis=-1), :
        ]
        corners2d[:, 1, :] = corners3d[
            np.arange(M), np.argmax(corners3d[:, :4, dims[0]], axis=-1), :
        ]
        corners2d[:, 2, :] = corners3d[
            np.arange(M), 4 + np.argmax(corners3d[:, -4:, dims[0]], axis=-1), :
        ]
        corners2d[:, 3, :] = corners3d[
            np.arange(M), 4 + np.argmin(corners3d[:, -4:, dims[0]], axis=-1), :
        ]

    if flip:
        corners2d[:, :, 1] *= -1

    # Define box edges with shape (boxes, edges, dimensions, resolution)
    res = 50
    edges = np.zeros((M, 4, 3, res))
    for i in range(4):
        edges[:, i, 0, :] = np.linspace(
            corners2d[:, i % 4, 0], corners2d[:, (i + 1) % 4, 0], num=res
        ).T
        edges[:, i, 1, :] = np.linspace(
            corners2d[:, i % 4, 1], corners2d[:, (i + 1) % 4, 1], num=res
        ).T
        edges[:, i, 2, :] = np.linspace(
            corners2d[:, i % 4, 2], corners2d[:, (i + 1) % 4, 2], num=res
        ).T

    # Convert cartesian box edges to spherical box edges
    for i in range(4):
        x = edges[:, i, 0, :].flatten()
        y = edges[:, i, 1, :].flatten()
        z = edges[:, i, 2, :].flatten()
        r, phi, roh = cart2spher(x, y, z, degrees=True)
        r = np.ones_like(r) * r_max if r_max is not None else r
        edges[:, i, 0, :] = r.reshape((M, res))
        edges[:, i, 1, :] = phi.reshape((M, res))
        edges[:, i, 2, :] = roh.reshape((M, res))

    if cart:
        for i in range(4):
            r = edges[:, i, 0, :].flatten()
            phi = edges[:, i, 1, :].flatten()
            roh = edges[:, i, 2, :].flatten()
            x, y, z = spher2cart(r, phi, roh, degrees=True)
            edges[:, i, 0, :] = x.reshape((M, res))
            edges[:, i, 1, :] = y.reshape((M, res))
            edges[:, i, 2, :] = z.reshape((M, res))

    # Get colormap
    cm = get_tum_accent_cm()

    # Plot boxes
    for box_edges, box in zip(edges, boxes):
        for edge in box_edges:
            if flip:
                ax.plot(edge[dims[1], :], edge[dims[0], :], color=cm(int(box[-2])))
            else:
                ax.plot(edge[dims[0], :], edge[dims[1], :], color=cm(int(box[-2])))


def visu_2d_radar_grid(
    ax: plt.Axes,
    grid: np.ndarray,
    raster: List[np.ndarray] = None,
    cart: bool = False,
    dims: str = "ra",
    r_max: float = 1.0,
    cm: Colormap = None,
    flip: bool = False,
):
    """ """
    # Swap axis
    if flip:
        grid = grid.T
        raster = list(reversed(raster))

    # Mesh grid based on sensor specifications
    if raster is not None:
        x_mesh, y_mesh = np.meshgrid(raster[0], raster[1])
    else:
        x_mesh, y_mesh = np.meshgrid(
            np.arange(grid.shape[0] + 1), np.arange(grid.shape[1] + 1)
        )

    if cart and dims in {"ra", "ar"}:
        # Convert polar to cartesian coordinate values
        x_shape, y_shape = x_mesh.shape, y_mesh.shape
        x_mesh, y_mesh = polar2cart(x_mesh.flatten(), y_mesh.flatten(), degrees=True)
        x_mesh, y_mesh = x_mesh.reshape(x_shape), y_mesh.reshape(y_shape)

    if cart and dims in {"ae", "ea"}:
        # Convert spherical to cartesian coordinate values
        x_shape, y_shape = x_mesh.shape, y_mesh.shape
        _, y_mesh, x_mesh = spher2cart(
            np.ones_like(x_mesh).flatten() * r_max,
            y_mesh.flatten(),
            x_mesh.flatten(),
            degrees=True,
        )
        x_mesh, y_mesh = x_mesh.reshape(x_shape), y_mesh.reshape(y_shape)

    # Get radar rcs values (dB)
    rcs = 10 * np.log10(grid)

    # Plot radar data
    if flip:
        y_mesh *= -1
        p = ax.pcolormesh(y_mesh, x_mesh, rcs.T, cmap=cm, shading="nearest")
    else:
        p = ax.pcolormesh(x_mesh, y_mesh, rcs.T, cmap=cm, shading="nearest")

    # Add colorbar to the plot
    plt.colorbar(p, ax=ax, label="Power in dB")


def visu_2d_radar_data(
    grid: np.ndarray,
    dims: str,
    boxes: np.ndarray = None,
    points: np.ndarray = None,
    raster: List[np.ndarray] = None,
    roi: bool = True,
    label: Tuple[str, str] = None,
    cart: bool = False,
    r_max: float = 1.0,
    cm: Colormap = None,
    dst: str = None,
    **kwargs,
) -> None:
    """Visualizes a given 2D radar grid.

    Arguments:
        grid: 2D grid representing a slice of the 4D radar
            tesseract with shape (N, M).
        dims:
        boxes:
        points:
        raster: Rasterization values (grid points) of
            of the associated radar dimensions with
            shape (N, M).
        roi:
        label: Description of the provided radar
            dimensions.
        cart: Wheter to project the provided
            grid values from polar to cartesian
            coordinates. If True, the second
            raster dimension has to represent the
            angular values.
        cm: Color map to visualize the gird values.
        dst: Destination filename to save the figure
            to. If provided, the figure will not be
            displayed but only saved to file.
    """
    # Check input data
    valid_dims = {"ra", "ar", "ae", "ea"}
    if cart and dims not in valid_dims:
        raise ValueError(
            f"It is only possible to visualize projections "
            f"of spatial and non perpendicular dimensions. "
            f"Therefore, you can only visualize the "
            f"{valid_dims} dimensions but {dims} was given."
        )
    # Initialize parameters
    flip = False
    dims_to_xyz = {"r": 0, "a": 1, "e": 2}
    xyz = tuple((dims_to_xyz[d] for d in dims))

    # Adjust parameters
    if dims in {"ar", "ea"}:
        flip = True

    if "e" not in dims:
        r_max = None

    # Initialize figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cm = cm if cm is not None else "viridis"

    # Visualize 2D radar grid
    visu_2d_radar_grid(
        ax=ax,
        grid=grid,
        raster=raster,
        cart=cart,
        dims=dims,
        r_max=r_max,
        cm=cm,
        flip=flip,
    )

    if roi:
        roi = (
            np.min(raster[0]),
            np.max(raster[0]),
            np.min(raster[1]),
            np.max(raster[1]),
        )

    # Visualize 2D point cloud
    if points is not None:
        visu_2d_lidar_points(
            ax, points, dims=xyz, roi=roi, cart=cart, r_max=r_max, flip=not flip
        )

    # Visualize 2D bounding boxes
    if boxes is not None:
        visu_2d_boxes(ax, boxes, dims=xyz, cart=cart, r_max=r_max, flip=flip)

    # Add axis label
    if label is not None:
        ax.set_xlabel(label[0])
        ax.set_ylabel(label[1])

    # Set equal ax ratios
    ax.axis("equal")

    if dst is not None:
        fig.savefig(dst)
    else:
        fig.show()


def visu_radar_tesseract(
    tesseract: np.ndarray,
    dims: str,
    raster: Dict[str, np.ndarray] = None,
    aggregation_func: Callable = np.max,
    **kwargs,
) -> None:
    """Visualizes the specified dimensions of a given radar tesseract.

    Arguments:
        tesseract: Data of the 4D radar tesseract with shape
            (doppler, range, elevation, azimuth).
        dims: Dimensions to visualize. Can be any combination of
            either two or three dimensions expressed by thier
            abbriviation (r: range, d: doppler, a: azimuth,
            e: elevation), e.g. 'ra'.
        raster: Dictionary specifying the rasterization values
            (grid points) of all radar dimensions.
        aggregation_func: Aggregation function for the
            reduction of the radar dimensions.
    """
    # Get maximum range
    r_max = max(raster["r"])

    # Get raster of the data distribution
    if raster is not None:
        raster = [raster[d] for d in dims]

    # Map dim abbreviations to data dimensions
    dim_order = {"d": 0, "r": 1, "e": 2, "a": 3}
    dim_names = {"d": "doppler", "r": "range", "e": "elevation", "a": "azimuth"}
    names = [dim_names[d] for d in dims]
    dim_idx = [dim_order[d] for d in dims]

    # Reduce radar data dimensions
    data = aggregation_func(
        tesseract, axis=tuple(set(dim_order.values()).difference(set(dim_idx)))
    )

    # rcs = tesseract[0, ...]
    # doppler = doppler_raster[np.argmax(tesseract, axis=0)]

    # Restructure data accoring to the given order in dims
    data = np.moveaxis(data, np.arange(data.ndim), np.argsort(dim_idx))

    # Select plot based on the number of dimensions
    if not 1 < len(dims) < 4:
        raise ValueError(
            f"There must be either two or three dimensions "
            f"selected for visualization but {len(dims)} "
            f"were given!"
        )

    # Visualize 3D radar data
    if len(dims) == 3:
        visu_3d_radar_data(cube=data, dims=dims, raster=raster, cm=TUMCM, **kwargs)

    # Visualize 2D radar data
    if len(dims) == 2:
        visu_2d_radar_data(
            grid=data,
            dims=dims,
            raster=raster,
            r_max=r_max,
            label=names,
            cm=TUMCM,
            **kwargs,
        )


# from k-Radar


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print("Warning: not a hull %s" % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = (
        boxes3d.new_tensor(
            (
                [1, 1, -1],
                [1, -1, -1],
                [-1, -1, -1],
                [-1, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [-1, -1, 1],
                [-1, 1, 1],
            )
        )
        / 2
    )

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(
        -1, 8, 3
    )
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def corners_rect_to_camera(corners):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        corners:  (8, 3) [x0, y0, z0, ...], (x, y, z) is the point coordinate in image rect

    Returns:
        boxes_rect:  (7,) [x, y, z, l, h, w, r] in rect camera coords
    """
    height_group = [(0, 4), (1, 5), (2, 6), (3, 7)]
    width_group = [(0, 1), (2, 3), (4, 5), (6, 7)]
    length_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    vector_group = [(0, 3), (1, 2), (4, 7), (5, 6)]
    height, width, length = 0.0, 0.0, 0.0
    vector = np.zeros(2, dtype=np.float32)
    for index_h, index_w, index_l, index_v in zip(
        height_group, width_group, length_group, vector_group
    ):
        height += np.linalg.norm(corners[index_h[0], :] - corners[index_h[1], :])
        width += np.linalg.norm(corners[index_w[0], :] - corners[index_w[1], :])
        length += np.linalg.norm(corners[index_l[0], :] - corners[index_l[1], :])
        vector[0] += (corners[index_v[0], :] - corners[index_v[1], :])[0]
        vector[1] += (corners[index_v[0], :] - corners[index_v[1], :])[2]

    height, width, length = height * 1.0 / 4, width * 1.0 / 4, length * 1.0 / 4
    rotation_y = -np.arctan2(vector[1], vector[0])

    center_point = corners.mean(axis=0)
    center_point[1] += height / 2
    camera_rect = np.concatenate(
        [center_point, np.array([length, height, width, rotation_y])]
    )

    return camera_rect


def mask_boxes_outside_range_numpy(
    boxes, limit_range, min_num_corners=1, use_center_to_filter=True
):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    if use_center_to_filter:
        box_centers = boxes[:, 0:3]
        mask = (
            (box_centers >= limit_range[0:3]) & (box_centers <= limit_range[3:6])
        ).all(axis=-1)
    else:
        corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
        corners = corners[:, :, 0:2]
        mask = ((corners >= limit_range[0:2]) & (corners <= limit_range[3:5])).all(
            axis=2
        )
        mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask


# def remove_points_in_boxes3d(points, boxes3d):
#     """
#     Args:
#         points: (num_points, 3 + C)
#         boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

#     Returns:

#     """
#     boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
#     points, is_numpy = common_utils.check_numpy_to_torch(points)
#     point_masks = roiaware_pool3d_utils.points_in_boxes_cpu(points[:, 0:3], boxes3d)
#     points = points[point_masks.sum(dim=0) == 0]

#     return points.numpy() if is_numpy else points


def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_camera_copy = copy.deepcopy(boxes3d_camera)
    xyz_camera, r = boxes3d_camera_copy[:, 0:3], boxes3d_camera_copy[:, 6:7]
    l, h, w = (
        boxes3d_camera_copy[:, 3:4],
        boxes3d_camera_copy[:, 4:5],
        boxes3d_camera_copy[:, 5:6],
    )

    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
    """
    Args:
        boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    w, l, h = (
        boxes3d_lidar_copy[:, 3:4],
        boxes3d_lidar_copy[:, 4:5],
        boxes3d_lidar_copy[:, 5:6],
    )
    r = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] += h[:, 0] / 2
    return np.concatenate(
        [boxes3d_lidar_copy[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1
    )


def boxes3d_kitti_lidar_to_fakelidar(boxes3d_lidar):
    """
    Args:
        boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        boxes3d_fakelidar: [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    dx, dy, dz = (
        boxes3d_lidar_copy[:, 3:4],
        boxes3d_lidar_copy[:, 4:5],
        boxes3d_lidar_copy[:, 5:6],
    )
    heading = boxes3d_lidar_copy[:, 6:7]

    boxes3d_lidar_copy[:, 2] -= dz[:, 0] / 2
    return np.concatenate(
        [boxes3d_lidar_copy[:, 0:3], dy, dx, dz, -heading - np.pi / 2], axis=-1
    )


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """

    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d


def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    boxes3d_lidar_copy = copy.deepcopy(boxes3d_lidar)
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = (
        boxes3d_lidar_copy[:, 3:4],
        boxes3d_lidar_copy[:, 4:5],
        boxes3d_lidar_copy[:, 5:6],
    )
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)


def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [l / 2.0, l / 2.0, -l / 2.0, -l / 2.0, l / 2.0, l / 2.0, -l / 2.0, -l / 2],
        dtype=np.float32,
    ).T
    z_corners = np.array(
        [w / 2.0, -w / 2.0, -w / 2.0, w / 2.0, w / 2.0, -w / 2.0, -w / 2.0, w / 2.0],
        dtype=np.float32,
    ).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array(
            [
                h / 2.0,
                h / 2.0,
                h / 2.0,
                h / 2.0,
                -h / 2.0,
                -h / 2.0,
                -h / 2.0,
                -h / 2.0,
            ],
            dtype=np.float32,
        ).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(
        ry.size, dtype=np.float32
    )
    rot_list = np.array(
        [
            [np.cos(ry), zeros, -np.sin(ry)],
            [zeros, ones, zeros],
            [np.sin(ry), zeros, np.cos(ry)],
        ]
    )  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate(
        (
            x_corners.reshape(-1, 8, 1),
            y_corners.reshape(-1, 8, 1),
            z_corners.reshape(-1, 8, 1),
        ),
        axis=2,
    )  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = (
        rotated_corners[:, :, 0],
        rotated_corners[:, :, 1],
        rotated_corners[:, :, 2],
    )

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2
    )

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(
            boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1
        )
        boxes2d_image[:, 1] = np.clip(
            boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1
        )
        boxes2d_image[:, 2] = np.clip(
            boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1
        )
        boxes2d_image[:, 3] = np.clip(
            boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1
        )

    return boxes2d_image


def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(
        area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6
    )
    return iou


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(
        rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]]
    )
    aligned_bev_boxes = torch.cat(
        (boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1
    )
    return aligned_bev_boxes


def boxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def area(box) -> torch.Tensor:
    """
    Computes the area of all the boxes.

    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area


# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = area(boxes1)
    area2 = area(boxes2)

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def center_to_corner2d(center, dim):
    corners_norm = torch.tensor(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]], device=dim.device
    ).type_as(
        center
    )  # (4, 2)
    corners = dim.view([-1, 1, 2]) * corners_norm.view([1, 4, 2])  # (N, 4, 2)
    corners = corners + center.view(-1, 1, 2)
    return corners


def bbox3d_overlaps_diou(pred_boxes, gt_boxes):
    """
    https://github.com/agent-sgs/PillarNet/blob/master/det3d/core/utils/center_utils.py
    Args:
        pred_boxes (N, 7):
        gt_boxes (N, 7):

    Returns:
        _type_: _description_
    """
    assert pred_boxes.shape[0] == gt_boxes.shape[0]

    qcorners = center_to_corner2d(pred_boxes[:, :2], pred_boxes[:, 3:5])  # (N, 4, 2)
    gcorners = center_to_corner2d(gt_boxes[:, :2], gt_boxes[:, 3:5])  # (N, 4, 2)

    inter_max_xy = torch.minimum(qcorners[:, 2], gcorners[:, 2])
    inter_min_xy = torch.maximum(qcorners[:, 0], gcorners[:, 0])
    out_max_xy = torch.maximum(qcorners[:, 2], gcorners[:, 2])
    out_min_xy = torch.minimum(qcorners[:, 0], gcorners[:, 0])

    # calculate area
    volume_pred_boxes = pred_boxes[:, 3] * pred_boxes[:, 4] * pred_boxes[:, 5]
    volume_gt_boxes = gt_boxes[:, 3] * gt_boxes[:, 4] * gt_boxes[:, 5]

    inter_h = torch.minimum(
        pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5]
    ) - torch.maximum(
        pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5], gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5]
    )
    inter_h = torch.clamp(inter_h, min=0)

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    volume_inter = inter[:, 0] * inter[:, 1] * inter_h
    volume_union = volume_gt_boxes + volume_pred_boxes - volume_inter

    # boxes_iou3d_gpu(pred_boxes, gt_boxes)
    inter_diag = torch.pow(gt_boxes[:, 0:3] - pred_boxes[:, 0:3], 2).sum(-1)

    outer_h = torch.maximum(
        gt_boxes[:, 2] + 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] + 0.5 * pred_boxes[:, 5]
    ) - torch.minimum(
        gt_boxes[:, 2] - 0.5 * gt_boxes[:, 5], pred_boxes[:, 2] - 0.5 * pred_boxes[:, 5]
    )
    outer_h = torch.clamp(outer_h, min=0)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = outer[:, 0] ** 2 + outer[:, 1] ** 2 + outer_h**2

    dious = volume_inter / volume_union - inter_diag / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)

    return dious


# def main(src, cfg, dst):

#     # Load the image
#     image = plt.imread(image_path)

#     # Initialize figure
#     fig, ax = plt.subplots(1, 1, figsize=(12, 8))

#     # Display the image
#     ax.imshow(image)

#     # Plot bounding boxes
#     visu_2d_boxes(ax, bounding_boxes, dims=(0, 1))

#     # Show the plot
#     plt.show()


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="DPRT data preprocessing")

#     default_src = [
#         Path("/data/Samsung 8TB 1"),
#         Path("/data/Samsung 500GB"),
#         Path("/data/Samsung 8TB 2"),
#     ]

#     parser.add_argument(
#         "--src",
#         type=Path,
#         nargs="+",
#         default=default_src,
#         help="Paths to the raw dataset folders.",
#     )

#     parser.add_argument(
#         "--cfg",
#         type=Path,
#         default=Path("/app/config/kradar.json"),
#         help="Path to the configuration file.",
#     )
#     parser.add_argument(
#         "--dst",
#         type=Path,
#         default=Path("/data/Samsung 8TB 2/kradar/processed"),
#         help="Path to save the processed dataset.",
#     )
#     args = parser.parse_args()

#     main(src=args.src, cfg=args.cfg, dst=args.dst)
