import torch
import numpy as np
import torchvision
from torchvision.ops.boxes import batched_nms, box_area
import logging
from utils.inout import save_json, load_json, save_npz
from utils.bbox_utils import xyxy_to_xywh, xywh_to_xyxy, force_binary_mask
import time
from PIL import Image
import torch.nn.functional as F
from pytorch3d.structures import Pointclouds, utils as py3d_util
from pytorch3d.ops import (
    knn_points,
    sample_farthest_points,
    corresponding_points_alignment,
)
from pytorch3d.transforms import (
    transform3d,
    euler_angles_to_matrix,
    matrix_to_euler_angles,
)

lmo_object_ids = np.array(
    [
        1,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]
)  # object ID of occlusionLINEMOD is different


def mask_to_rle(binary_mask):
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")

    last_elem = 0
    running_length = 0

    for i, elem in enumerate(binary_mask.ravel(order="F")):
        if elem == last_elem:
            pass
        else:
            counts.append(running_length)
            running_length = 0
            last_elem = elem
        running_length += 1

    counts.append(running_length)

    return rle


class BatchedData:
    """
    A structure for storing data in batched format.
    Implements basic filtering and concatenation.
    """

    def __init__(self, batch_size, data=None, **kwargs) -> None:
        self.batch_size = batch_size
        if data is not None:
            self.data = data
        else:
            self.data = []

    def __len__(self):
        assert self.batch_size is not None, "batch_size is not defined"
        return np.ceil(len(self.data) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        assert self.batch_size is not None, "batch_size is not defined"
        return self.data[idx * self.batch_size : (idx + 1) * self.batch_size]

    def cat(self, data, dim=0):
        if len(self.data) == 0:
            self.data = data
        else:
            self.data = torch.cat([self.data, data], dim=dim)

    def append(self, data):
        self.data.append(data)

    def stack(self, dim=0):
        self.data = torch.stack(self.data, dim=dim)


class Detections:
    """
    A structure for storing detections.
    """

    def __init__(self, data) -> None:
        if isinstance(data, str):
            data = self.load_from_file(data)
        for key, value in data.items():
            setattr(self, key, value)
        self.keys = list(data.keys())
        if "boxes" in self.keys:
            if isinstance(self.boxes, np.ndarray):
                self.to_torch()
            self.boxes = self.boxes.long()

    def remove_very_small_detections(self, config):
        img_area = self.masks.shape[1] * self.masks.shape[2]
        box_areas = box_area(self.boxes) / img_area
        mask_areas = self.masks.sum(dim=(1, 2)) / img_area
        keep_idxs = torch.logical_and(
            box_areas > config.min_box_size**2, mask_areas > config.min_mask_size
        )
        # logging.info(f"Removing {len(keep_idxs) - keep_idxs.sum()} detections")
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms_per_object_id(self, nms_thresh=0.5):
        keep_idxs = BatchedData(None)
        all_indexes = torch.arange(len(self.object_ids), device=self.boxes.device)
        for object_id in torch.unique(self.object_ids):
            idx = self.object_ids == object_id
            idx_object_id = all_indexes[idx]
            keep_idx = torchvision.ops.nms(
                self.boxes[idx].float(), self.scores[idx].float(), nms_thresh
            )
            keep_idxs.cat(idx_object_id[keep_idx])
        keep_idxs = keep_idxs.data
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idxs])

    def apply_nms(self, nms_thresh=0.5):
        keep_idx = torchvision.ops.nms(
            self.boxes.float(), self.scores.float(), nms_thresh
        )
        for key in self.keys:
            setattr(self, key, getattr(self, key)[keep_idx])

    def add_attribute(self, key, value):
        setattr(self, key, value)
        self.keys.append(key)

    def __len__(self):
        return len(self.boxes)

    def check_size(self):
        mask_size = len(self.masks)
        box_size = len(self.boxes)
        score_size = len(self.scores)
        object_id_size = len(self.object_ids)
        assert (
            mask_size == box_size == score_size == object_id_size
        ), f"Size mismatch {mask_size} {box_size} {score_size} {object_id_size}"

    def to_numpy(self):
        for key in self.keys:
            setattr(self, key, getattr(self, key).cpu().numpy())

    def to_torch(self):
        for key in self.keys:
            a = getattr(self, key)
            setattr(self, key, torch.from_numpy(getattr(self, key)))

    def save_to_file(
        self, scene_id, frame_id, runtime, file_path, dataset_name, return_results=False
    ):
        """
        scene_id, image_id, category_id, bbox, time
        """
        boxes = xyxy_to_xywh(self.boxes)
        results = {
            "scene_id": scene_id,
            "image_id": frame_id,
            "category_id": (
                self.object_ids + 1
                if dataset_name != "lmo"
                else lmo_object_ids[self.object_ids]
            ),
            "score": self.scores,
            "bbox": boxes,
            "time": runtime,
            "segmentation": self.masks,
        }
        save_npz(file_path, results)
        if return_results:
            return results

    def load_from_file(self, file_path):
        data = np.load(file_path)
        masks = data["segmentation"]
        boxes = xywh_to_xyxy(np.array(data["bbox"]))
        data = {
            "object_ids": data["category_id"] - 1,
            "bbox": boxes,
            "scores": data["score"],
            "masks": masks,
        }
        logging.info(f"Loaded {file_path}")
        return data

    def filter(self, idxs):
        for key in self.keys:
            setattr(self, key, getattr(self, key)[idxs])

    def clone(self):
        """
        Clone the current object
        """
        return Detections(self.__dict__.copy())


def convert_npz_to_json(idx, list_npz_paths):
    npz_path = list_npz_paths[idx]
    detections = np.load(npz_path)
    results = []
    for idx_det in range(len(detections["bbox"])):
        result = {
            "scene_id": int(detections["scene_id"]),
            "image_id": int(detections["image_id"]),
            "category_id": int(detections["category_id"][idx_det]),
            "bbox": detections["bbox"][idx_det].tolist(),
            "score": float(detections["score"][idx_det]),
            "time": float(detections["time"]),
            "segmentation": mask_to_rle(
                force_binary_mask(detections["segmentation"][idx_det])
            ),
        }
        results.append(result)
    return results


def feature2pixel(features, target_size):
    B, _, H, W = target_size
    pixel_features = (
        features.reshape(B, 16, 16, 4, 4, 64).permute(0, 5, 1, 3, 2, 4).contiguous()
    )
    pixel_features = pixel_features.reshape(B, -1, 64, 64)
    pixel_features = F.interpolate(
        pixel_features, (H, W), mode="bilinear", align_corners=False
    )
    return pixel_features


def packed_to_padded(packed, lengths, pad_value=-1):
    list = py3d_util.packed_to_list(packed, lengths.tolist())
    return py3d_util.list_to_padded(list, pad_value=pad_value)


def mean_distance_nearest_neighbor_torch(points):
    # Use knn_points from PyTorch3D to find the nearest neighbors (k=2)
    # We request 2 neighbors because the first one will be the point itself
    knn_result = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=2)

    # Extract distances to the nearest neighbors
    distances = knn_result.dists.squeeze(0)[:, 1]  # Skip the 0th neighbor (itself)

    # Compute the maximum distance
    max_distance = torch.mean(distances).item()

    return max_distance


def mean_distance_to_centroid(points):

    # Calculate the centroid
    centroid = points.mean(dim=0)

    # Calculate the distances from each point to the centroid
    distances = torch.sqrt(torch.sum((points - centroid) ** 2, axis=1))
    distances = distances / distances.max()

    # Calculate the mean distance
    mean_distance = distances.mean()

    return mean_distance


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def match_pointsfeats(pointsfeats1, pointsfeats2, k):
    assert (
        len(pointsfeats1.shape) == 3 and len(pointsfeats2.shape) == 3
    ), "The points should be 3D"
    assert (
        pointsfeats1.shape[-1] > 3 and pointsfeats2.shape[-1] > 3
    ), "The points should be feature augmented"
    assert (
        pointsfeats1.shape[-1] == pointsfeats2.shape[-1]
    ), "The dimentionality of the features should be the same"

    feats1 = pointsfeats1[..., 3:]
    feats2 = pointsfeats2[..., 3:]
    feats1 /= feats1.norm(dim=-1, keepdim=True)
    feats2 /= feats2.norm(dim=-1, keepdim=True)

    sim = feats1 @ feats2.transpose(1, 2)
    B, p1, p2 = sim.shape
    reshaped_sim = sim.view(B, p1 * p2)
    top_values, flat_top_indices = torch.topk(reshaped_sim, k, dim=1)
    # Convert flat indices to 2D indices
    row_indices = flat_top_indices // p2
    col_indices = flat_top_indices % p2
    points1_matched = pointsfeats1[
        torch.arange(pointsfeats1.shape[0]).unsqueeze(-1), row_indices
    ][:, :, :3]
    points2_matched = pointsfeats2[
        torch.arange(pointsfeats2.shape[0]).unsqueeze(-1), col_indices
    ][:, :, :3]

    R, T, _ = corresponding_points_alignment(
        points1_matched.to(float), points2_matched.to(float)
    )

    return R, T
    # matched_pairs = torch.cat([points1_matched, points2_matched], dim=-1).view(-1, 2, 3)

    # np.savetxt(f"tmp/pcl/points1_matched.xyz", points1_matched.cpu().numpy())
    # np.savetxt(f"tmp/pcl/points2_matched.xyz", points2_matched.cpu().numpy())
    # np.savetxt(f"tmp/pcl/points1.xyz", pointsfeats1.cpu().numpy())
    # np.savetxt(f"tmp/pcl/points2.xyz", pointsfeats2.cpu().numpy())
    # for i in range(len(matched_pairs)):
    #     np.savetxt(f"tmp/pcl/pair{i}.xyz", matched_pairs[i].cpu().numpy())

    # return matched_pairs, sim


def mask_outliers_by_reference(reference_points, to_clean_points, threshold=None):
    assert len(reference_points.shape) == 3 and len(to_clean_points.shape) == 3
    assert reference_points.shape[0] == to_clean_points.shape[0]

    reference_points = reference_points[:, :, :3]
    to_clean_points = to_clean_points[:, :, :3]
    if threshold is None:
        threshold = mean_distance_to_centroid(reference_points) * 1e-5

    dist_idx = knn_points(to_clean_points, reference_points, K=1)
    if len(dist_idx.dists) == 1:
        dists = dist_idx.dists.squeeze(0).squeeze(-1)
    else:
        dists = dist_idx.dists.squeeze(-1)
    good_idx = dists < threshold

    return good_idx


def clean_point_average_features_to_refpoints(
    templates_points, ref_points, ref_points_size, threshold=None
):
    assert len(templates_points.shape) == 3
    assert len(ref_points.shape) == 2
    if threshold is None:
        threshold = mean_distance_to_centroid(ref_points) * 1e-5

    # cleaning the templates points and features
    aggregated_templates = templates_points.view(-1, templates_points.shape[-1])

    cleaned_templates_indexes = mask_outliers_by_reference(
        ref_points.unsqueeze(0), aggregated_templates.unsqueeze(0), threshold
    )
    aggregated_templates = aggregated_templates[
        cleaned_templates_indexes.nonzero().squeeze(-1)
    ]
    aggregated_templates = torch.cat(
        [
            aggregated_templates[:, :3],
            aggregated_templates[:, 3:]
            / aggregated_templates[:, 3:].norm(dim=-1, keepdim=True),
        ],
        dim=-1,
    )

    # cleaning the ref points
    cleaned_ref_points_indexes = mask_outliers_by_reference(
        aggregated_templates.unsqueeze(0), ref_points.unsqueeze(0), threshold
    )
    ref_points = ref_points[cleaned_ref_points_indexes.nonzero().squeeze(-1)]

    ref_points = sample_farthest_points(
        ref_points.unsqueeze(0),
        K=ref_points_size,
    )[
        0
    ][0]

    # np.savetxt(
    #     f"tmp/pcl/aggregated_templates_clean1.xyz", aggregated_templates.cpu().numpy()
    # )
    # np.savetxt(f"tmp/pcl/ref_points_clean1.xyz", ref_points.cpu().numpy())

    dist_idx = knn_points(
        ref_points.unsqueeze(0),
        aggregated_templates[:, :3].unsqueeze(0),
        K=int(len(aggregated_templates) / len(ref_points)),
    )
    # dists = dist_idx.dists.squeeze(0).squeeze(-1)
    idx = dist_idx.idx.squeeze(0).squeeze(-1)
    neighbour_feats = aggregated_templates[:, 3:][idx]
    neighbour_feats = neighbour_feats.mean(dim=1)
    neighbour_feats /= neighbour_feats.norm(dim=-1, keepdim=True)

    obj_points_feats = torch.cat([ref_points, neighbour_feats], dim=-1)

    return obj_points_feats


def depth_image_to_pointcloud_old(depth, scale, K):
    u = torch.arange(0, depth.shape[2])
    v = torch.arange(0, depth.shape[1])

    u, v = torch.meshgrid(u, v, indexing="xy")
    u = u.to(depth.device)
    v = v.to(depth.device)

    # depth metric is mm, depth_scale metric is m
    # K metric is m
    Z = depth * scale / 1000
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    xyz = torch.stack((X, Y, Z), dim=-1)
    return xyz


def depth_image_to_pointcloud(depth, scale, K):
    """
    Convert a depth image to a point cloud in 3D space.

    Args:
        depth (torch.Tensor): Depth image tensor of shape (B, H, W).
        scale (torch.Tensor): Scale factor tensor of shape (B,).
        K (torch.Tensor): Camera intrinsic matrix tensor of shape (B, 3, 3).

    Returns:
        torch.Tensor: Point cloud tensor of shape (B, H, W, 3), where the last dimension represents (X, Y, Z) coordinates.
    """
    assert len(depth.shape) == 3
    assert len(K.shape) == 3
    assert len(scale.shape) == 1
    assert depth.shape[0] == K.shape[0] == scale.shape[0]

    B, H, W = depth.shape
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]

    # Generate pixel coordinates grid
    # !!!!!!!!!!!!!! Much faster than using torch.meshgrid !!!!!!!!!!!!!!
    u = torch.arange(W, device=depth.device).unsqueeze(0).unsqueeze(0)
    v = torch.arange(H, device=depth.device).unsqueeze(1).unsqueeze(0)
    u = u.float().expand(B, H, -1)
    v = v.float().expand(B, -1, W)

    ###### Homoheneous calculation is not memory efficient
    ###### Here I use the gridwise calculation
    """
        z = d / depth_scale
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
    """
    Z = depth * scale / 1000
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = torch.stack((X, Y, Z), dim=-1)
    return xyz


# name this fu
def filter_and_sample_point_cloud(xyzmaps, mask, point_size, filter_bad_depth=True):
    """
    Filters and samples a point cloud based on a given mask and point size.

    Args:
        xyzmaps (torch.Tensor): Input tensor of shape (..., C) representing the XYZ maps.
        mask (torch.Tensor): Input tensor of shape (..., C) representing the mask.
        point_size (int): Number of points to sample from the point cloud.
        featmaps (torch.Tensor, optional): Input tensor of shape (..., C) representing the feature maps.

    Returns:
        torch.Tensor: Filtered and sampled XYZ maps of shape (B, C, point_size).
        torch.Tensor: Filtered and sampled feature maps of shape (B, C, point_size) if featmaps is not None, else None.
    """
    mask = mask > 0.5
    xyzmaps = xyzmaps * mask
    if filter_bad_depth:
        filter = xyzmaps[..., 2] > 0        
    else:
        filter =  ~torch.all(xyzmaps == 0, dim=-1)
    lengths = (filter > 0).sum(dim=tuple(range(1, filter.dim())))
    xyzmaps = packed_to_padded(xyzmaps[filter], lengths, -10)
    sample_ind = sample_farthest_points(
        xyzmaps[:, :, :3],
        lengths=lengths,
        # K=min(point_size, lengths.max().item()), This cause inconsistency when stacking data
        K=point_size,
    )[1]
    xyzmaps = xyzmaps[torch.arange(xyzmaps.shape[0]).unsqueeze(-1), sample_ind]
    return xyzmaps


def center_pointsfeats_to(from_pointsfeats, to_pointsfeats):
    assert (
        len(from_pointsfeats.shape) == 3 and len(to_pointsfeats.shape) == 3
    ), "The points should be 3D"
    obj_centers = from_pointsfeats[:, :, :3].mean(dim=1)
    proposal_xyzs_centers = to_pointsfeats[:, :, :3].mean(dim=1)
    center_T = proposal_xyzs_centers - obj_centers

    transform = transform3d.Translate(center_T)
    centered_pointsfeats = torch.cat(
        [
            transform.transform_points(from_pointsfeats[:, :, :3].to(torch.float32)),
            from_pointsfeats[:, :, 3:],
        ],
        dim=-1,
    ).to(torch.float32)

    return centered_pointsfeats, center_T
