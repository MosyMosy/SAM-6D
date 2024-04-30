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
from pytorch3d.ops import knn_points

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
            "category_id": self.object_ids + 1
            if dataset_name != "lmo"
            else lmo_object_ids[self.object_ids],
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


# ----------------------------------------------------------------------------------
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
    distances = torch.sqrt(torch.sum((points - centroid)**2, axis=1))
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

def average_features_to_refpoints(batch_points, feats, ref_points, threshold=None):
    assert len(batch_points.shape) == 3
    assert batch_points.shape[0] == feats.shape[0]
    assert len(ref_points.shape) == 2
    if threshold is None:
        threshold = mean_distance_nearest_neighbor_torch(ref_points) * 2.5
        threshold = mean_distance_to_centroid(ref_points) * 1e-5
    ref_features = torch.zeros(
        (ref_points.shape[0], feats.shape[-1]),
        dtype=torch.float32,
        device=ref_points.device,
    )
    ref_features_count = torch.zeros(
        (ref_points.shape[0]),
        dtype=torch.float32,
        device=ref_points.device,
    )
    for i in range(len(batch_points)):
        result = knn_points(batch_points[i].unsqueeze(0), ref_points.unsqueeze(0), K=1)
        dists = result.dists.squeeze(0).squeeze(-1)
        idx = result.idx.squeeze(0).squeeze(-1)
        good_idx = (dists < threshold).nonzero().squeeze(-1)
        # np.savetxt(f"tmp/pcl/near{i}.xyz", batch_points[i][good_idx].cpu().numpy())
        filtered_knn_index = idx[good_idx]
        ref_features[filtered_knn_index] += feats[i][good_idx]
        ref_features_count[filtered_knn_index] += 1
        
    ref_features = ref_features / ref_features_count.unsqueeze(-1)
    obj_points_feats = torch.cat([ref_points, ref_features], dim=-1)

    return obj_points_feats

