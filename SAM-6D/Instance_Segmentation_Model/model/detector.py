import torch
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
from PIL import Image
import logging
import os
import os.path as osp
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from utils.inout import save_json, load_json, save_json_bop23
from model.utils import BatchedData, Detections, convert_npz_to_json
from hydra.utils import instantiate
import time
import glob
from functools import partial
import multiprocessing
import trimesh
from model.loss import MaskedPatch_MatrixSimilarity
from utils.trimesh_utils import (
    depth_image_to_pointcloud_translate_torch,
    depth_image_to_pointcloud,
)
from utils.poses.pose_utils import get_obj_poses_from_template_level
from utils.bbox_utils import xyxy_to_xywh, compute_iou
from pytorch3d.transforms import transform3d
from model.utils import (
    BatchedData,
    Detections,
    convert_npz_to_json,
    feature2pixel,
    packed_to_padded,
    mask_outliers_by_reference,
    filter_and_sample_point_cloud,
    clean_point_average_features_to_refpoints,
)
from pytorch3d.ops import sample_farthest_points, box3d_overlap
from pytorch3d.structures import Pointclouds


class Instance_Segmentation_Model_old(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        self.visible_thred = visible_thred
        self.pointcloud_sample_num = pointcloud_sample_num

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")

    def set_reference_objects(self):
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects ...")

        start_time = time.time()
        self.ref_data = {
            "descriptors": BatchedData(None),
            "appe_descriptors": BatchedData(None),
        }
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        appe_descriptors_path = osp.join(
            self.ref_dataset.template_dir, "descriptors_appe.pth"
        )

        # Loading main descriptors
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            # by Moslem and Ali, we only need the 5th object of the descriptors
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(
                self.device
            )  # [29:]
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(ref_feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        # Loading appearance descriptors
        if self.onboarding_config.rendering_type == "pbr":
            appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")
        if (
            os.path.exists(appe_descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            # by Moslem and Ali, we only need the 5th object of the descriptors
            self.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(
                self.device
            )  # [29:]
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing appearance descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                ref_feats = self.descriptor_model.compute_masked_patch_feature(
                    ref_imgs, ref_masks
                )
                self.ref_data["appe_descriptors"].append(ref_feats)

            self.ref_data["appe_descriptors"].stack()
            self.ref_data["appe_descriptors"] = self.ref_data["appe_descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["appe_descriptors"], appe_descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}, \
            Appearance descriptors shape: {self.ref_data['appe_descriptors'].shape}"
        )

    def set_reference_object_pointcloud(self):
        """
        Loading the pointclouds of reference objects: (N_object, N_pointcloud, 3)
        N_pointcloud: the number of points sampled from the reference object mesh.
        """
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects point cloud ...")

        start_time = time.time()
        pointcloud = BatchedData(None)
        pointcloud_path = osp.join(self.ref_dataset.template_dir, "pointcloud.pth")
        obj_pose_path = f"{self.ref_dataset.template_dir}/template_poses.npy"

        # Loading pointcloud pose
        if (
            os.path.exists(obj_pose_path)
            and not self.onboarding_config.reset_descriptors
        ):
            poses = (
                torch.tensor(np.load(obj_pose_path)).to(self.device).to(torch.float32)
            )  # N_all_template x 4 x 4
        else:
            template_poses = get_obj_poses_from_template_level(
                level=2, pose_distribution="all"
            )
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
            np.save(obj_pose_path, template_poses)

        self.ref_data["poses"] = poses[
            self.ref_dataset.index_templates, :, :
        ]  # N_template x 4 x 4
        if (
            os.path.exists(pointcloud_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["pointcloud"] = torch.load(
                pointcloud_path, map_location="cuda:0"
            ).to(self.device)
        else:
            mesh_path = osp.join(self.ref_dataset.root_dir, "models")
            if not os.path.exists(mesh_path):
                raise Exception("Can not find the mesh path.")
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Generating pointcloud dataset ...",
            ):
                # loading cad
                if self.dataset_name == "lmo":
                    all_pc_idx = [1, 5, 6, 8, 9, 10, 11, 12]
                    pc_id = all_pc_idx[idx]
                else:
                    pc_id = idx + 1
                mesh = trimesh.load_mesh(
                    os.path.join(mesh_path, f"obj_{(pc_id):06d}.ply")
                )
                model_points = (
                    mesh.sample(self.pointcloud_sample_num).astype(np.float32) / 1000.0
                )
                pointcloud.append(torch.tensor(model_points))

            pointcloud.stack()  # N_objects x N_pointcloud x 3
            self.ref_data["pointcloud"] = pointcloud.data.to(self.device)

            # save the precomputed features for future use
            torch.save(self.ref_data["pointcloud"], pointcloud_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Pointcloud shape: {self.ref_data['pointcloud'].shape}"
        )

    def best_template_pose(self, scores, pred_idx_objects):
        _, best_template_idxes = torch.max(scores, dim=-1)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        best_template_idx = torch.gather(
            best_template_idxes, dim=1, index=pred_idx_objects
        )[:, 0]

        return best_template_idx

    def project_template_to_image(self, best_pose, pred_object_idx, batch, proposals):
        """
        Obtain the RT of the best template, then project the reference pointclouds to query image,
        getting the bbox of projected pointcloud from the image.
        """

        pose_R = self.ref_data["poses"][best_pose, 0:3, 0:3]  # N_query x 3 x 3
        select_pc = self.ref_data["pointcloud"][
            pred_object_idx, ...
        ]  # N_query x N_pointcloud x 3
        (N_query, N_pointcloud, _) = select_pc.shape

        # translate object_selected pointcloud by the selected best pose and camera coordinate
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)
        translate = self.Calculate_the_query_translation(
            proposals,
            batch["depth"][0],
            batch["cam_intrinsic"][0],
            batch["depth_scale"],
        )
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1)

        # project the pointcloud to the image
        cam_instrinsic = (
            batch["cam_intrinsic"][0][None, ...].repeat(N_query, 1, 1).to(torch.float32)
        )
        image_homo = torch.bmm(cam_instrinsic, posed_pc.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(
            torch.int
        )  # N_query x N_pointcloud x 2
        (imageH, imageW) = batch["depth"][0].shape
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1)
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1)

        return image_vu

    def Calculate_the_query_translation(
        self, proposal, depth, cam_intrinsic, depth_scale
    ):
        """
        Calculate the translation amount from the origin of the object coordinate system to the camera coordinate system.
        Cut out the depth using the provided mask and calculate the mean as the translation.
        proposal: N_query x imageH x imageW
        depth: imageH x imageW
        """
        (N_query, imageH, imageW) = proposal.squeeze_().shape
        masked_depth = proposal * (depth[None, ...].repeat(N_query, 1, 1))
        translate = depth_image_to_pointcloud_translate_torch(
            masked_depth, depth_scale, cam_intrinsic
        )
        return translate.to(torch.float32)

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def compute_semantic_score(self, proposal_decriptors):
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        semantic_score = score_per_proposal[idx_selected_proposals]

        # compute the best view of template
        flitered_scores = scores[idx_selected_proposals, ...]
        best_template = self.best_template_pose(flitered_scores, pred_idx_objects)

        return idx_selected_proposals, pred_idx_objects, semantic_score, best_template

    def compute_appearance_score(
        self, best_pose, pred_objects_idx, qurey_appe_descriptors
    ):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate(
            (pred_objects_idx[None, :], best_pose[None, :]), dim=0
        )
        ref_appe_descriptors = self.ref_data["appe_descriptors"][
            con_idx[0, ...], con_idx[1, ...], ...
        ]  # N_query x N_patch x N_feature

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(
            qurey_appe_descriptors, ref_appe_descriptors
        )

        return appe_scores, ref_appe_descriptors

    def compute_geometric_score(
        self,
        image_uv,
        proposals,
        appe_descriptors,
        ref_aux_descriptor,
        visible_thred=0.5,
    ):

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(
            appe_descriptors, ref_aux_descriptor, visible_thred
        )

        # IoU calculation
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)

        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            self.set_reference_objects()
            self.set_reference_object_pointcloud()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]).cpu().numpy().transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np)

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # compute semantic descriptors and appearance descriptors for query proposals
        query_decriptors, query_appe_descriptors = self.descriptor_model(
            image_np, detections
        )
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # compute the geometric score
        image_uv = self.project_template_to_image(
            best_template, pred_idx_objects, batch, detections.masks
        )

        geometric_score, visible_ratio = self.compute_geometric_score(
            image_uv,
            detections,
            query_appe_descriptors,
            ref_aux_descriptor,
            visible_thred=self.visible_thred,
        )

        # final score
        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
            pool = multiprocessing.Pool(processes=num_workers)
            convert_npz_to_json_with_idx = partial(
                convert_npz_to_json,
                list_npz_paths=result_paths,
            )
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
            formatted_detections = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection)

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")


class Instance_Segmentation_Model_geo3d(pl.LightningModule):
    def __init__(
        self,
        segmentor_model,
        descriptor_model,
        onboarding_config,
        matching_config,
        post_processing_config,
        log_interval,
        log_dir,
        visible_thred,
        pointcloud_sample_num,
        **kwargs,
    ):
        # define the network
        super().__init__()
        self.segmentor_model = segmentor_model
        self.descriptor_model = descriptor_model

        self.onboarding_config = onboarding_config
        self.matching_config = matching_config
        self.post_processing_config = post_processing_config
        self.log_interval = log_interval
        self.log_dir = log_dir

        self.visible_thred = visible_thred
        self.pointcloud_sample_num = pointcloud_sample_num

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(osp.join(self.log_dir, "predictions"), exist_ok=True)
        self.inv_rgb_transform = T.Compose(
            [
                T.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                ),
            ]
        )
        logging.info(f"Init CNOS done!")

    def set_semantic_features(self):
        logging.info("Initializing semantic_features ...")
        start_time = time.time()

        self.ref_data["descriptors"] = BatchedData(None)
        descriptors_path = osp.join(self.ref_dataset.template_dir, "descriptors.pth")
        if self.onboarding_config.rendering_type == "pbr":
            descriptors_path = descriptors_path.replace(".pth", "_pbr.pth")

        if (
            os.path.exists(descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["descriptors"] = torch.load(descriptors_path).to(self.device)
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                # save_image(self.ref_dataset.inv_rgb_transform(ref_imgs), f"./tmp/selected_templates/tless_{idx}.png", nrow=7)
                feats = self.descriptor_model.compute_features(
                    ref_imgs, token_name="x_norm_clstoken"
                )
                self.ref_data["descriptors"].append(feats)

            self.ref_data["descriptors"].stack()  # N_objects x descriptor_size
            self.ref_data["descriptors"] = self.ref_data["descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["descriptors"], descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Descriptors shape: {self.ref_data['descriptors'].shape}"
        )

    def set_apperance_features(self):
        logging.info("Initializing apperance_features ...")
        start_time = time.time()

        self.ref_data["appe_descriptors"] = BatchedData(None)
        appe_descriptors_path = osp.join(
            self.ref_dataset.template_dir, "descriptors_appe.pth"
        )
        if self.onboarding_config.rendering_type == "pbr":
            appe_descriptors_path = appe_descriptors_path.replace(".pth", "_pbr.pth")

        if (
            os.path.exists(appe_descriptors_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["appe_descriptors"] = torch.load(appe_descriptors_path).to(
                self.device
            )
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing appearance descriptors ...",
            ):
                ref_imgs = self.ref_dataset[idx]["templates"].to(self.device)
                ref_masks = self.ref_dataset[idx]["template_masks"].to(self.device)
                feats = self.descriptor_model.compute_masked_patch_feature(
                    ref_imgs, ref_masks
                )
                self.ref_data["appe_descriptors"].append(feats)

            self.ref_data["appe_descriptors"].stack()
            self.ref_data["appe_descriptors"] = self.ref_data["appe_descriptors"].data

            # save the precomputed features for future use
            torch.save(self.ref_data["appe_descriptors"], appe_descriptors_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, apperance_features shape: {self.ref_data['appe_descriptors'].shape}"
        )

    def set_obj_points_features(self):
        logging.info("Initializing obj_points_features ...")
        start_time = time.time()

        self.ref_data["template_points_feats"] = BatchedData(None)
        self.ref_data["obj_points_feats"] = BatchedData(None)
        self.ref_data["obj_template_R"] = BatchedData(None)
        self.ref_data["obj_template_T"] = BatchedData(None)
        self.ref_data["obj_template_mask"] = BatchedData(None)
        template_points_feats_path = osp.join(
            self.ref_dataset.template_dir, "template_points_feats.pth"
        )
        obj_points_feats_path = osp.join(
            self.ref_dataset.template_dir, "obj_points_feats.pth"
        )
        obj_template_R_path = osp.join(
            self.ref_dataset.template_dir, "obj_template_R.pth"
        )
        obj_template_T_path = osp.join(
            self.ref_dataset.template_dir, "obj_template_T.pth"
        )
        obj_template_mask_path = osp.join(
            self.ref_dataset.template_dir, "obj_template_mask.pth"
        )
        if self.onboarding_config.rendering_type == "pbr":
            template_points_feats_path = template_points_feats_path.replace(
                ".pth", "_pbr.pth"
            )
            obj_points_feats_path = obj_points_feats_path.replace(".pth", "_pbr.pth")
            obj_template_R_path = obj_template_R_path.replace(".pth", "_pbr.pth")
            obj_template_T_path = obj_template_T_path.replace(".pth", "_pbr.pth")
            obj_template_mask_path = obj_template_mask_path.replace(".pth", "_pbr.pth")

        if (
            os.path.exists(template_points_feats_path)
            and os.path.exists(obj_points_feats_path)
            and os.path.exists(obj_template_R_path)
            and os.path.exists(obj_template_T_path)
            and os.path.exists(obj_template_mask_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["template_points_feats"] = torch.load(
                template_points_feats_path
            ).to(self.device)
            self.ref_data["obj_points_feats"] = torch.load(obj_points_feats_path).to(
                self.device
            )
            self.ref_data["obj_template_R"] = torch.load(obj_template_R_path).to(
                self.device
            )
            self.ref_data["obj_template_T"] = torch.load(obj_template_T_path).to(
                self.device
            )
            self.ref_data["obj_template_mask"] = torch.load(obj_template_mask_path).to(
                self.device
            )
        else:
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Computing template_points_feats ...",
            ):
                template_info = self.ref_dataset[idx]
                xyzs = template_info["xyzs"].to(self.device)
                feats = self.ref_data["appe_descriptors"][idx]
                ref_masks = template_info["template_masks"].to(self.device)
                uncropped_masks = template_info["template_masks_uncropped"].to(
                    self.device
                )
                target_size = template_info["templates"].size()
                templates_R = template_info["obj_template_R"].to(self.device)
                templates_T = template_info["obj_template_T"].to(self.device)
                ref_pointcloud = self.ref_data["pointcloud"][idx].to(torch.float32)

                B, C, H, W = xyzs.shape
                feats = feature2pixel(feats, target_size)
                xyzs = torch.cat([xyzs, feats], dim=1)
                xyzs = xyzs.permute(0, 2, 3, 1).view(
                    B, H * W, -1
                )  # B , H x W , (3 + C)

                # feats = feats.permute(0, 2, 3, 1)  # B x H x W x C
                wrong_feature_mask = (
                    torch.norm(xyzs[:, :, 3:], dim=-1, keepdim=True).squeeze(-1) != 0
                )
                # transform the template points to the object coordinate system
                transform = (
                    transform3d.Translate(templates_T).inverse().rotate(templates_R)
                )
                xyzs = torch.cat(
                    [
                        transform.transform_points(xyzs[:, :, :3].to(torch.float32)),
                        xyzs[:, :, 3:],
                    ],
                    dim=-1,
                ).to(torch.float32)

                # As we have zero features in the masked apperance features, we need to filter them out
                # These regions are in the border of the mask from masked patchs.
                wrong_depth_mask = ref_masks.view(B, H * W) == 1

                # cleaning the templates points by the reference pointcloud
                outlier_points_mask = mask_outliers_by_reference(
                    ref_pointcloud.unsqueeze(0).repeat(xyzs.shape[0], 1, 1), xyzs
                )
                final_mask = wrong_feature_mask & wrong_depth_mask & outlier_points_mask
                xyzs = filter_and_sample_point_cloud(
                    xyzs,
                    final_mask.unsqueeze(-1),
                    self.onboarding_config.template_point_size,
                    filter_bad_depth=False,
                )

                self.ref_data["template_points_feats"].append(xyzs)

                obj_points_feats = clean_point_average_features_to_refpoints(
                    xyzs,
                    ref_pointcloud,
                    self.onboarding_config.template_obj_point_size,
                )
                self.ref_data["obj_points_feats"].append(
                    obj_points_feats.to(torch.float32)
                )

                self.ref_data["obj_template_R"].append(templates_R)
                self.ref_data["obj_template_T"].append(templates_T)
                self.ref_data["obj_template_mask"].append(uncropped_masks)
                # for i in range(42):
                # np.savetxt(f'tmp/pcl/new{i}.xyz', xyzs[i].cpu().numpy())

            self.ref_data[
                "template_points_feats"
            ].stack()  # N_objects x descriptor_size
            # center the template's pointclouds
            self.ref_data["template_points_feats"] = self.ref_data[
                "template_points_feats"
            ].data

            self.ref_data["obj_points_feats"].stack()
            self.ref_data["obj_points_feats"] = self.ref_data["obj_points_feats"].data

            self.ref_data["obj_template_R"].stack()
            self.ref_data["obj_template_R"] = self.ref_data["obj_template_R"].data

            self.ref_data["obj_template_T"].stack()
            self.ref_data["obj_template_T"] = self.ref_data["obj_template_T"].data

            self.ref_data["obj_template_mask"].stack()
            self.ref_data["obj_template_mask"] = self.ref_data["obj_template_mask"].data

            # save the precomputed features for future use
            torch.save(
                self.ref_data["template_points_feats"], template_points_feats_path
            )
            torch.save(self.ref_data["obj_points_feats"], obj_points_feats_path)

            torch.save(self.ref_data["obj_template_R"], obj_template_R_path)
            torch.save(self.ref_data["obj_template_T"], obj_template_T_path)
            torch.save(self.ref_data["obj_template_mask"], obj_template_mask_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, obj_points_features shape: {self.ref_data['template_points_feats'].shape}"
        )

    def set_reference_object_pointcloud(self):
        """
        Loading the pointclouds of reference objects: (N_object, N_pointcloud, 3)
        N_pointcloud: the number of points sampled from the reference object mesh.

        Poses: (42, 4, 4)
        Pointcloud: (30, 2048, 3)
        """
        os.makedirs(
            osp.join(self.log_dir, f"predictions/{self.dataset_name}"), exist_ok=True
        )
        logging.info("Initializing reference objects point cloud ...")

        start_time = time.time()
        pointcloud = BatchedData(None)
        pointcloud_path = osp.join(self.ref_dataset.template_dir, "pointcloud.pth")
        obj_pose_path = osp.join(self.ref_dataset.template_dir, "template_poses.npy")

        # Loading pointcloud pose
        if (
            os.path.exists(obj_pose_path)
            and not self.onboarding_config.reset_descriptors
        ):
            # 642*4*4 All the templates of the dataset
            poses = (
                torch.tensor(np.load(obj_pose_path)).to(self.device).to(torch.float32)
            )  # N_all_template x 4 x 4
        else:
            template_poses = get_obj_poses_from_template_level(
                level=2, pose_distribution="all"
            )
            template_poses[:, :3, 3] *= 0.4
            poses = torch.tensor(template_poses).to(self.device).to(torch.float32)
            np.save(obj_pose_path, template_poses)
        # use just 42 templates from 642 templates. distributed around the object.
        self.ref_data["poses"] = poses[
            self.ref_dataset.index_templates, :, :
        ]  # N_template x 4 x 4

        if (
            os.path.exists(pointcloud_path)
            and not self.onboarding_config.reset_descriptors
        ):
            self.ref_data["pointcloud"] = torch.load(
                pointcloud_path, map_location="cuda:0"
            ).to(self.device)
        else:
            mesh_path = osp.join(self.ref_dataset.root_dir, "models")
            if not os.path.exists(mesh_path):
                raise Exception("Can not find the mesh path.")
            for idx in tqdm(
                range(len(self.ref_dataset)),
                desc="Generating pointcloud dataset ...",
            ):
                # loading cad
                if self.dataset_name == "lmo":
                    all_pc_idx = [1, 5, 6, 8, 9, 10, 11, 12]
                    pc_id = all_pc_idx[idx]
                else:
                    pc_id = idx + 1
                mesh = trimesh.load_mesh(
                    os.path.join(mesh_path, f"obj_{(pc_id):06d}.ply")
                )
                model_points = (
                    mesh.sample(self.pointcloud_sample_num).astype(np.float32) / 1000.0
                )
                pointcloud.append(torch.tensor(model_points))

            pointcloud.stack()  # N_objects x N_pointcloud x 3
            self.ref_data["pointcloud"] = pointcloud.data.to(self.device)

            # save the precomputed features for future use
            torch.save(self.ref_data["pointcloud"], pointcloud_path)

        end_time = time.time()
        logging.info(
            f"Runtime: {end_time-start_time:.02f}s, Pointcloud shape: {self.ref_data['pointcloud'].shape}"
        )

    def best_template_pose(self, scores, pred_idx_objects):
        _, best_template_idxes = torch.max(scores, dim=-1)
        N_query, N_object = best_template_idxes.shape[0], best_template_idxes.shape[1]
        pred_idx_objects = pred_idx_objects[:, None].repeat(1, N_object)

        assert N_query == pred_idx_objects.shape[0], "Prediction num != Query num"

        best_template_idx = torch.gather(
            best_template_idxes, dim=1, index=pred_idx_objects
        )[:, 0]

        return best_template_idx

    def project_template_to_image(self, best_pose, pred_object_idx, batch, proposals):
        """
        Obtain the RT of the best template, then project the reference pointclouds to query image,
        getting the bbox of projected pointcloud from the image.
        """

        # pose_R = self.ref_data["poses"][best_pose, 0:3, 0:3]  # N_query x 3 x 3
        # based on the real pose of the best template
        pose_R = self.ref_data["obj_template_R"][pred_object_idx, best_pose].to(torch.float32)  # N_query x 3 x 3
        select_pc = self.ref_data["pointcloud"][
            pred_object_idx, ...
        ]  # N_query x N_pointcloud x 3
        (N_query, N_pointcloud, _) = select_pc.shape

        # translate object_selected pointcloud by the selected best pose and camera coordinate
        posed_pc = torch.matmul(pose_R, select_pc.permute(0, 2, 1)).permute(0, 2, 1)
        translate = self.Calculate_the_query_translation(
            proposals,
            batch["depth"][0],
            batch["cam_intrinsic"][0],
            batch["depth_scale"],
        )
        posed_pc = posed_pc + translate[:, None, :].repeat(1, N_pointcloud, 1)

        # project the pointcloud to the image
        cam_instrinsic = (
            batch["cam_intrinsic"][0][None, ...].repeat(N_query, 1, 1).to(torch.float32)
        )
        image_homo = torch.bmm(cam_instrinsic, posed_pc.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        image_vu = (image_homo / image_homo[:, :, -1][:, :, None])[:, :, 0:2].to(
            torch.int
        )  # N_query x N_pointcloud x 2
        (imageH, imageW) = batch["depth"][0].shape
        image_vu[:, :, 0].clamp_(min=0, max=imageW - 1)
        image_vu[:, :, 1].clamp_(min=0, max=imageH - 1)

        return image_vu

    def Calculate_the_query_translation(
        self, proposal, depth, cam_intrinsic, depth_scale
    ):
        """
        Calculate the translation amount from the origin of the object coordinate system to the camera coordinate system.
        Cut out the depth using the provided mask and calculate the mean as the translation.
        proposal: N_query x imageH x imageW
        depth: imageH x imageW
        """
        (N_query, imageH, imageW) = proposal.squeeze_().shape
        masked_depth = proposal * (depth[None, ...].repeat(N_query, 1, 1))
        translate = depth_image_to_pointcloud_translate_torch(
            masked_depth, depth_scale, cam_intrinsic
        )
        return translate.to(torch.float32)

    def move_to_device(self):
        self.descriptor_model.model = self.descriptor_model.model.to(self.device)
        self.descriptor_model.model.device = self.device
        # if there is predictor in the model, move it to device
        if hasattr(self.segmentor_model, "predictor"):
            self.segmentor_model.predictor.model = (
                self.segmentor_model.predictor.model.to(self.device)
            )
        else:
            self.segmentor_model.model.setup_model(device=self.device, verbose=True)
        logging.info(f"Moving models to {self.device} done!")

    def compute_semantic_score(self, proposal_decriptors):
        # compute matching scores for each proposals
        scores = self.matching_config.metric(
            proposal_decriptors, self.ref_data["descriptors"]
        )  # N_proposals x N_objects x N_templates
        if self.matching_config.aggregation_function == "mean":
            score_per_proposal_and_object = (
                torch.sum(scores, dim=-1) / scores.shape[-1]
            )  # N_proposals x N_objects
        elif self.matching_config.aggregation_function == "median":
            score_per_proposal_and_object = torch.median(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "max":
            score_per_proposal_and_object = torch.max(scores, dim=-1)[0]
        elif self.matching_config.aggregation_function == "avg_5":
            score_per_proposal_and_object = torch.topk(scores, k=5, dim=-1)[0]
            score_per_proposal_and_object = torch.mean(
                score_per_proposal_and_object, dim=-1
            )
        else:
            raise NotImplementedError

        # assign each proposal to the object with highest scores
        score_per_proposal, assigned_idx_object = torch.max(
            score_per_proposal_and_object, dim=-1
        )  # N_query

        idx_selected_proposals = torch.arange(
            len(score_per_proposal), device=score_per_proposal.device
        )[score_per_proposal > self.matching_config.confidence_thresh]
        pred_idx_objects = assigned_idx_object[idx_selected_proposals]
        semantic_score = score_per_proposal[idx_selected_proposals]

        # compute the best view of template
        flitered_scores = scores[idx_selected_proposals, ...]
        best_template = self.best_template_pose(flitered_scores, pred_idx_objects)

        return idx_selected_proposals, pred_idx_objects, semantic_score, best_template

    def compute_appearance_score(
        self, best_pose, pred_objects_idx, qurey_appe_descriptors
    ):
        """
        Based on the best template, calculate appearance similarity indicated by appearance score
        """
        con_idx = torch.concatenate(
            (pred_objects_idx[None, :], best_pose[None, :]), dim=0
        )
        ref_appe_descriptors = self.ref_data["appe_descriptors"][
            con_idx[0, ...], con_idx[1, ...], ...
        ]  # N_query x N_patch x N_feature

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        appe_scores = aux_metric.compute_straight(
            qurey_appe_descriptors, ref_appe_descriptors
        )

        return appe_scores, ref_appe_descriptors

    def get_corners(self, points):
        min_coords, _ = torch.min(points, dim=1)
        max_coords, _ = torch.max(points, dim=1)

        # Generate the 8 corners of the bounding box for each point cloud in the batch
        corners = torch.stack(
            [
                torch.stack([min_coords[:, i] for i in range(3)], dim=1),
                torch.stack(
                    [
                        min_coords[:, i] if i != 0 else max_coords[:, i]
                        for i in range(3)
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        max_coords[:, i] if i != 2 else min_coords[:, i]
                        for i in range(3)
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        min_coords[:, i] if i != 1 else max_coords[:, i]
                        for i in range(3)
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        min_coords[:, i] if i != 2 else max_coords[:, i]
                        for i in range(3)
                    ],
                    dim=1,
                ),
                torch.stack(
                    [
                        max_coords[:, i] if i != 1 else min_coords[:, i]
                        for i in range(3)
                    ],
                    dim=1,
                ),
                torch.stack([max_coords[:, i] for i in range(3)], dim=1),
                torch.stack(
                    [
                        max_coords[:, i] if i != 0 else min_coords[:, i]
                        for i in range(3)
                    ],
                    dim=1,
                ),
            ]
        ).permute(1, 0, 2)
        return corners.to(torch.float32)

    def batch_box_volume(self, corners):
        # Ensure 'corners' is a tensor of shape (B, 8, 3) where B is the batch size
        # Compute the three edge vectors from the first corner
        edge1 = corners[:, 1] - corners[:, 0]  # Vector from corner 0 to 1
        edge2 = corners[:, 3] - corners[:, 0]  # Vector from corner 0 to 3
        edge3 = corners[:, 4] - corners[:, 0]  # Vector from corner 0 to 4

        # Calculate the cross product of edge1 and edge2
        cross_product = torch.cross(
            edge1, edge2, dim=1
        )  # Ensure correct dimension for cross product

        # Calculate the dot product to find the volume
        volume = torch.abs(torch.einsum("bi,bi->b", cross_product, edge3))

        return volume

    def compute_geometric_score(
        self,
        image_uv,
        proposals,
        appe_descriptors,
        ref_aux_descriptor,
        visible_thred=0.5,
    ):

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(
            appe_descriptors, ref_aux_descriptor, visible_thred
        )

        # IoU calculation
        y1x1 = torch.min(image_uv, dim=1).values
        y2x2 = torch.max(image_uv, dim=1).values
        xyxy = torch.concatenate((y1x1, y2x2), dim=-1)

        iou = compute_iou(xyxy, proposals.boxes)

        return iou, visible_ratio

    def compute_3D_geometric_score(
        self,
        batch,
        best_template_idx,
        pred_objects_idx,
        proposals,
        appe_descriptors,
        ref_aux_descriptor,
        visible_thred=0.5,
    ):
        box_corner_vertices = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=torch.float32,
        ).to(self.device)
        templates_points = self.ref_data["template_points_feats"][
            pred_objects_idx, best_template_idx, :, :3
        ]

        aux_metric = MaskedPatch_MatrixSimilarity(metric="cosine", chunk_size=64)
        visible_ratio = aux_metric.compute_visible_ratio(
            appe_descriptors, ref_aux_descriptor, visible_thred
        )

        depth = batch["depth"][0]
        cam_intrinsic = batch["cam_intrinsic"][0]
        depth_scale = batch["depth_scale"][0]
        mask = proposals.masks
        mask = mask > 0.5
        masked_depth = depth[None, :] * mask
        xyzs = depth_image_to_pointcloud(masked_depth, depth_scale, cam_intrinsic)
        filter = xyzs[:, :, :, 2] > 0
        lengths = (filter > 0).sum(dim=[1, 2])
        xyzs = packed_to_padded(xyzs[filter], lengths)
        # padded FPS
        sample_ind = sample_farthest_points(
            xyzs,
            lengths=lengths,
            K=self.onboarding_config.template_point_size,
        )[1]
        xyzs = xyzs[torch.arange(xyzs.shape[0]).unsqueeze(-1), sample_ind]

        templates_centers = templates_points.mean(dim=1)
        xyzs_centers = xyzs.mean(dim=1)
        templates_points -= (templates_centers - xyzs_centers)[:, None, :]
        templates_corners = self.get_corners(templates_points)
        xyzs_corners = self.get_corners(xyzs)
        bad_corners = lengths == 0
        depth_thresh = 1e-6
        bad_corners = bad_corners * self.batch_box_volume(xyzs_corners) < depth_thresh
        xyzs_corners[bad_corners] = box_corner_vertices
        intersection_vol, iou_3d = box3d_overlap(
            templates_corners, xyzs_corners, eps=depth_thresh
        )
        # IoU calculation
        # y1x1 = torch.min(image_uv, dim=1).values
        # y2x2 = torch.max(image_uv, dim=1).values
        # xyxy = torch.concatenate((y1x1, y2x2), dim=-1)
        # iou = compute_iou(xyxy, proposals.boxes)
        iou_3d = torch.diag(iou_3d)
        iou_3d[bad_corners] = 0.0
        return iou_3d, visible_ratio

    def test_step(self, batch, idx):
        if idx == 0:
            os.makedirs(
                osp.join(
                    self.log_dir,
                    f"predictions/{self.dataset_name}/{self.name_prediction_file}",
                ),
                exist_ok=True,
            )
            os.makedirs(
                osp.join(self.log_dir, f"predictions/{self.dataset_name}"),
                exist_ok=True,
            )
            self.ref_data = {}
            self.set_semantic_features()
            self.set_apperance_features()
            self.set_reference_object_pointcloud()
            self.set_obj_points_features()
            self.move_to_device()
        assert batch["image"].shape[0] == 1, "Batch size must be 1"

        image_np = (
            self.inv_rgb_transform(batch["image"][0]).cpu().numpy().transpose(1, 2, 0)
        )
        image_np = np.uint8(image_np.clip(0, 1) * 255)

        # run propoals
        proposal_stage_start_time = time.time()
        proposals = self.segmentor_model.generate_masks(image_np)

        # init detections with masks and boxes
        detections = Detections(proposals)
        detections.remove_very_small_detections(
            config=self.post_processing_config.mask_post_processing
        )

        # compute semantic descriptors and appearance descriptors for query proposals
        query_decriptors, query_appe_descriptors = self.descriptor_model(
            image_np, detections
        )
        proposal_stage_end_time = time.time()

        # matching descriptors
        matching_stage_start_time = time.time()
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = self.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor = self.compute_appearance_score(
            best_template, pred_idx_objects, query_appe_descriptors
        )

        # compute the geometric score
        # image_uv = self.project_template_to_image(
        #     best_template, pred_idx_objects, batch, detections.masks
        # )
        detections.masks = detections.masks[:, 0]
        
        if self.onboarding_config.geometric_score == "3D":
            geometric_score, visible_ratio = self.compute_3D_geometric_score(
                batch,
                best_template,
                pred_idx_objects,
                detections,
                query_appe_descriptors,
                ref_aux_descriptor,
                visible_thred=self.visible_thred,
            )
        elif self.onboarding_config.geometric_score == "projection":
            image_uv = self.project_template_to_image(
                best_template, pred_idx_objects, batch, detections.masks
            )
            geometric_score, visible_ratio = self.compute_geometric_score(
                image_uv,
                detections,
                query_appe_descriptors,
                ref_aux_descriptor,
                visible_thred=self.visible_thred,
            )
        else:
            raise NotImplementedError


        # final score
        final_score = (
            semantic_score + appe_scores + geometric_score * visible_ratio
        ) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", pred_idx_objects)
        detections.apply_nms_per_object_id(
            nms_thresh=self.post_processing_config.nms_thresh
        )
        matching_stage_end_time = time.time()

        runtime = (
            proposal_stage_end_time
            - proposal_stage_start_time
            + matching_stage_end_time
            - matching_stage_start_time
        )
        detections.to_numpy()

        scene_id = batch["scene_id"][0]
        frame_id = batch["frame_id"][0]
        file_path = osp.join(
            self.log_dir,
            f"predictions/{self.dataset_name}/{self.name_prediction_file}/scene{scene_id}_frame{frame_id}",
        )

        # save detections to file
        results = detections.save_to_file(
            scene_id=int(scene_id),
            frame_id=int(frame_id),
            runtime=runtime,
            file_path=file_path,
            dataset_name=self.dataset_name,
            return_results=True,
        )
        # save runtime to file
        np.savez(
            file_path + "_runtime",
            proposal_stage=proposal_stage_end_time - proposal_stage_start_time,
            matching_stage=matching_stage_end_time - matching_stage_start_time,
        )
        return 0

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:  # only rank 0 process
            # can use self.all_gather to gather results from all processes
            # but it is simpler just load the results from files so no file is missing
            result_paths = sorted(
                glob.glob(
                    osp.join(
                        self.log_dir,
                        f"predictions/{self.dataset_name}/{self.name_prediction_file}/*.npz",
                    )
                )
            )
            result_paths = sorted(
                [path for path in result_paths if "runtime" not in path]
            )
            num_workers = 10
            logging.info(f"Converting npz to json requires {num_workers} workers ...")
            pool = multiprocessing.Pool(processes=num_workers)
            convert_npz_to_json_with_idx = partial(
                convert_npz_to_json,
                list_npz_paths=result_paths,
            )
            detections = list(
                tqdm(
                    pool.imap_unordered(
                        convert_npz_to_json_with_idx, range(len(result_paths))
                    ),
                    total=len(result_paths),
                    desc="Converting npz to json",
                )
            )
            formatted_detections = []
            for detection in tqdm(detections, desc="Loading results ..."):
                formatted_detections.extend(detection)

            detections_path = f"{self.log_dir}/{self.name_prediction_file}.json"
            save_json_bop23(detections_path, formatted_detections)
            logging.info(f"Saved predictions to {detections_path}")
