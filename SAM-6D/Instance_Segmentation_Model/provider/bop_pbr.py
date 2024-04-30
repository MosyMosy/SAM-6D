import sys

# sys.path.insert(0,'/export/livia/home/vision/Myazdanpanah/projects/SAM-6D/SAM-6D/Instance_Segmentation_Model/')


import logging, os
import os.path as osp
from tqdm import tqdm
import time
import numpy as np
import torchvision.transforms as T
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import os.path as osp
import pandas as pd
from utils.inout import load_json, save_json, casting_format_to_save_json
from utils.poses.pose_utils import (
    load_index_level_in_level2,
    get_obj_poses_from_template_level,
    NearestTemplateFinder,
    farthest_sampling,
    combine_R_and_T,
)
from utils.trimesh_utils import depth_image_to_pointcloud
import torch
from utils.bbox_utils import CropResizePad
import pytorch_lightning as pl
from functools import partial
import multiprocessing
from provider.bop import BaseBOP
from pytorch3d.transforms import Rotate, Translate


class BOPTemplatePBR(BaseBOP):
    def __init__(
        self,
        root_dir,
        template_dir,
        obj_ids,
        processing_config,
        level_templates,
        pose_distribution,
        split="train_pbr",
        min_visib_fract=0.8,
        max_num_scenes=10,  # not need to search all scenes since it is slow
        max_num_frames=1000,  # not need to search all frames since it is slow
        **kwargs,
    ):
        self.template_dir = template_dir
        obj_ids = self.get_obj_ids(self.template_dir)
        if obj_ids is None:
            obj_ids = [
                int(obj_id[4:])
                for obj_id in os.listdir(template_dir)
                if osp.isdir(osp.join(template_dir, obj_id))
            ]
            obj_ids = sorted(obj_ids)
            logging.info(f"Found {obj_ids} objects in {self.template_dir}")
        self.obj_ids = obj_ids

        self.level_templates = level_templates
        self.pose_distribution = pose_distribution
        self.load_template_poses(level_templates, pose_distribution)
        self.processing_config = processing_config
        self.root_dir = root_dir
        self.split = split
        self.load_list_scene(split=split)
        logging.info(
            f"Found {len(self.list_scenes)} scene, but using only {max_num_scenes} scene for faster runtime"
        )

        self.list_scenes = self.list_scenes[:max_num_scenes]
        self.max_num_frames = max_num_frames
        self.min_visib_fract = min_visib_fract
        self.rgb_transform = T.Compose(
            [
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        self.proposal_processor = CropResizePad(self.processing_config.image_size)

    def __len__(self):
        return len(self.obj_ids)

    def load_metaData(self, reset_metaData):
        start_time = time.time()
        metaData = {
            "scene_id": [],
            "frame_id": [],
            "rgb_path": [],
            "depth_path": [],
            "camera_K": [],
            "depth_scale": [],
            "visib_fract": [],
            "obj_id": [],
            "idx_obj": [],
            "obj_poses": [],
        }
        logging.info(f"Loading metaData for split {self.split}")
        metaData_path = osp.join(self.root_dir, f"{self.split}_metaData.csv")
        if reset_metaData:
            for scene_path in tqdm(self.list_scenes, desc="Loading metaData"):
                scene_id = scene_path.split("/")[-1]
                if osp.exists(osp.join(scene_path, "rgb")):
                    rgb_paths = sorted(Path(scene_path).glob("rgb/*.[pj][pn][g]"))
                else:
                    rgb_paths = sorted(Path(scene_path).glob("gray/*.tif"))
                depth_paths = sorted(Path(scene_path).glob("depth/*.[pj][pn][g]"))
                # load poses
                scene_gt_info = load_json(osp.join(scene_path, "scene_gt_info.json"))
                scene_gt = load_json(osp.join(scene_path, "scene_gt.json"))
                scene_camera = load_json(osp.join(scene_path, "scene_camera.json"))
                for idx_frame in range(len(rgb_paths)):
                    rgb_path = rgb_paths[idx_frame]
                    depth_path = depth_paths[idx_frame]
                    frame_id = int(str(rgb_path).split("/")[-1].split(".")[0])
                    obj_ids = [int(x["obj_id"]) for x in scene_gt[f"{frame_id}"]]
                    obj_poses = np.array(
                        [
                            combine_R_and_T(x["cam_R_m2c"], x["cam_t_m2c"])
                            for x in scene_gt[f"{frame_id}"]
                        ]
                    )

                    visib_fracts = [
                        float(x["visib_fract"]) for x in scene_gt_info[f"{frame_id}"]
                    ]

                    camera_K = np.array(scene_camera[f"{frame_id}"]["cam_K"]).reshape(
                        3, 3
                    )
                    depth_scale = scene_camera[f"{frame_id}"]["depth_scale"]

                    # add to metaData
                    metaData["visib_fract"].extend(visib_fracts)
                    metaData["obj_id"].extend(obj_ids)
                    metaData["idx_obj"].extend(range(len(obj_ids)))
                    metaData["obj_poses"].extend(obj_poses)

                    metaData["scene_id"].extend([scene_id] * len(obj_ids))
                    metaData["frame_id"].extend([frame_id] * len(obj_ids))
                    metaData["rgb_path"].extend([str(rgb_path)] * len(obj_ids))
                    metaData["depth_path"].extend([str(depth_path)] * len(obj_ids))
                    metaData["camera_K"].extend([camera_K] * len(obj_ids))
                    metaData["depth_scale"].extend([depth_scale] * len(obj_ids))

                    if idx_frame > self.max_num_frames:
                        break
            self.metaData = pd.DataFrame.from_dict(metaData, orient="index")
            self.metaData = self.metaData.transpose()
            self.metaData.to_csv(metaData_path)
        else:
            self.metaData = pd.read_csv(metaData_path)

        # shuffle data
        self.metaData = self.metaData.sample(frac=1, random_state=2021).reset_index(
            drop=True
        )
        finish_time = time.time()
        logging.info(
            f"Finish loading metaData of size {len(self.metaData)} in {finish_time - start_time:.2f} seconds"
        )
        return metaData

    def load_template_poses(self, level_templates, pose_distribution):
        if pose_distribution == "all":
            self.index_templates = load_index_level_in_level2(level_templates, "all")
            self.template_poses = get_obj_poses_from_template_level(
                self.level_templates, self.pose_distribution
            )
        else:
            raise NotImplementedError

    def load_processed_metaData(self, reset_metaData):
        finder = NearestTemplateFinder(
            level_templates=self.level_templates,
            pose_distribution=self.pose_distribution,
            return_inplane=False,
        )
        metaData_path = osp.join(self.root_dir, f"{self.split}_processed_metaData.json")
        if reset_metaData or not osp.exists(metaData_path):
            self.load_metaData(reset_metaData=reset_metaData)
            # keep only objects having visib_fract > self.processing_config.min_visib_fract
            init_size = len(self.metaData)
            idx_keep = np.array(self.metaData["visib_fract"]) > self.min_visib_fract
            self.metaData = self.metaData.iloc[np.arange(len(self.metaData))[idx_keep]]
            self.metaData = self.metaData.reset_index(drop=True)

            selected_index = []
            index_dataframe = np.arange(0, len(self.metaData))
            # for each object, find reference frames by taking top k frames with farthest distance
            for obj_id in tqdm(
                self.obj_ids, desc="Finding nearest rendering close to template poses"
            ):
                selected_index_obj = index_dataframe[self.metaData["obj_id"] == obj_id]
                # subsample a bit if there are too many frames
                selected_index_obj = np.random.choice(selected_index_obj, 5000)
                obj_poses = np.array(
                    self.metaData.iloc[selected_index_obj].obj_poses.tolist()
                )
                # normalize translation to have unit norm
                obj_poses = np.array(obj_poses).reshape(-1, 4, 4)
                distance = np.linalg.norm(obj_poses[:, :3, 3], axis=1, keepdims=True)
                # print(distance[:10], distance.shape)
                obj_poses_normal = obj_poses.copy()
                obj_poses_normal[:, :3, 3] = obj_poses[:, :3, 3] / distance

                idx_keep = finder.search_nearest_query(obj_poses_normal)
                # update metaData
                selected_index.extend(selected_index_obj[idx_keep])
            self.metaData = self.metaData.iloc[selected_index]
            logging.info(
                f"Finish processing metaData from {init_size} to {len(self.metaData)}"
            )
            self.metaData = self.metaData.reset_index(drop=True)
            # self.metaData = casting_format_to_save_json(self.metaData)
            self.metaData.to_csv(metaData_path)
        else:
            self.metaData = pd.read_csv(metaData_path).reset_index(drop=True)

    def __getitem__(self, idx):
        templates, xyzs, masks, boxes, obj_template_poses = [], [], [], [], []
        obj_ids = []
        idx_range = range(
            idx * len(self.template_poses),
            (idx + 1) * len(self.template_poses),
        )
        for i in idx_range:
            rgb_path = self.metaData.iloc[i].rgb_path
            obj_id = self.metaData.iloc[i].obj_id
            obj_ids.append(obj_id)
            idx_obj = self.metaData.iloc[i].idx_obj
            scene_id = self.metaData.iloc[i].scene_id
            frame_id = self.metaData.iloc[i].frame_id
            mask_path = osp.join(
                self.root_dir,
                self.split,
                f"{int(scene_id):06d}",
                "mask_visib",
                f"{frame_id:06d}_{idx_obj:06d}.png",
            )

            depth_path = self.metaData.iloc[i].depth_path
            rgb = Image.open(rgb_path)
            mask = Image.open(mask_path)
            depth = Image.open(depth_path)
            masked_rgb = Image.composite(
                rgb, Image.new("RGB", rgb.size, (0, 0, 0)), mask
            )

            boxes.append(mask.getbbox())
            image = torch.from_numpy(np.array(masked_rgb.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask) / 255).float()
            depth = torch.from_numpy(np.array(depth).astype(float))
            # depth = depth * mask !!!!!!!! We should mask the point clouds after the proposal_processor
            xyz = depth_image_to_pointcloud(
                depth.unsqueeze(0),
                self.metaData.iloc[i].depth_scale,
                torch.from_numpy(self.metaData.iloc[i].camera_K),
            )[0]

            templates.append(image)
            xyzs.append(xyz)
            masks.append(mask.unsqueeze(-1))
            obj_template_poses.append(self.metaData.iloc[i].obj_poses)

        assert (
            len(np.unique(obj_ids)) == 1
        ), f"Only support one object per batch but found {np.unique(obj_ids)}"

        templates = torch.stack(templates).permute(0, 3, 1, 2)
        xyzs = torch.stack(xyzs).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = torch.tensor(np.array(boxes))
        obj_template_poses = torch.tensor(obj_template_poses)
        obj_template_R = obj_template_poses[:, :3, :3]
        obj_template_T = obj_template_poses[:, :3, 3] / 1000 # convert to meters

        templates_croped = self.proposal_processor(images=templates, boxes=boxes)
        xyzs_cropped = self.proposal_processor(images=xyzs, boxes=boxes)
        # a = xyzs_cropped.permute(0,2,3,1)[0]
        # a = a[a[:,:,2] > 0]
        # np.savetxt('tmp/pcl/scaled.xyz', a.numpy())
        masks_cropped = self.proposal_processor(images=masks, boxes=boxes)

        return {
            "templates": self.rgb_transform(templates_croped),
            "xyzs": xyzs_cropped,
            "template_masks": masks_cropped[:, 0, :, :],
            "obj_template_R": obj_template_R,
            "obj_template_T": obj_template_T,
        }



if __name__ == "__main__":
    import sys

    sys.path.insert(
        0,
        "/export/livia/home/vision/Myazdanpanah/projects/SAM-6D/SAM-6D/Instance_Segmentation_Model/",
    )

    logging.basicConfig(level=logging.INFO)
    from omegaconf import DictConfig, OmegaConf
    from torchvision.utils import make_grid, save_image

    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )
    dataset = BOPTemplatePBR(
        root_dir="/export/livia/home/vision/Myazdanpanah/projects/SAM-6D/SAM-6D/Data/BOP/tless",
        template_dir="/export/livia/home/vision/Myazdanpanah/projects/SAM-6D/SAM-6D/Data/BOP/templates_pyrender/tless",
        template_raw_dir="",
        obj_ids=None,
        level_templates=1,
        pose_distribution="all",
        processing_config=processing_config,
    )
    os.makedirs("./tmp", exist_ok=True)
    dataset.load_processed_metaData(reset_metaData=True)
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        sample["templates"] = inv_rgb_transform(sample["templates"])
        save_image(sample["templates"], f"./tmp/tless_{idx}.png", nrow=7)
