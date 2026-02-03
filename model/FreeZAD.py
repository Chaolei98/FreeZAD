import torch
from torch import nn
import torch.nn.functional as F  # NOTE: use nn.functional (conv1d lives here)
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import importlib
import numpy as np
from skimage import measure
import copy

tokenize = open_clip.get_tokenizer("coca_ViT-L-14")


class FreeZAD(nn.Module):
    def __init__(
        self,
        stride: int,
        kernel_size: int,
        normalize: bool,
        dataset: str,
        visualize: bool,
        remove_background: bool,
        split: int,
        setting: int,
        video_path: str,
    ):
        super(FreeZAD, self).__init__()

        # Load COCA model (OpenCLIP)
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14",
            pretrained="mscoco_finetuned_laion2B-s13B-b90k",
        )
        self.model = self.model.float()
        print("Loaded COCA model")

        # Hyper-parameters / options
        self.stride = stride
        self.kernel_size = kernel_size
        self.normalize = normalize
        self.dataset = dataset
        self.visualize = visualize
        self.remove_background = remove_background
        self.topk = 3
        self.m = 0.7
        self.split = split
        self.setting = setting
        self.video_path = video_path

        # Dataset-specific paths / metadata
        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
            self.video_dir = os.path.join(self.video_path, "Thumos14/videos/")
            fps_file_path = "data/thumos_annotations/thumos_fps.json"
            with open(fps_file_path, "r") as f:
                self.all_video_fps = json.load(f)

        elif self.dataset == "anet":
            dict_test_name = (
                f"t2_dict_test_{split}"
                if self.setting == 50
                else f"t1_dict_test_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/activitynet_annotations/anet_anno_action.json"
            self.video_dir = os.path.join(self.video_path)

        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        # Class name mapping
        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}

        # Text embeddings for all classes
        self.text_features = self.get_text_features()

        # Load annotations
        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

    def get_text_features(self):
        # Build prompts and encode them with COCA text encoder
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)  # split camelCase
            prompts.append("a video of action " + c)

        text = [tokenize(p) for p in prompts]
        text = torch.stack(text).squeeze()
        text = text.to(next(self.model.parameters()).device)

        text_features = self.model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_score(self, model, x, y):
        # Normalize and compute softmax similarity scores over classes
        x = x / x.norm(dim=-1, keepdim=True)
        scores = (model.logit_scale.exp() * x @ y.T).softmax(dim=-1)
        pred = scores.argmax(dim=-1)
        return pred, scores

    def infer_pseudo_labels(self, image_features):
        # Use video-level average feature to pick a top-1 pseudo label
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)

        self.text_features = self.text_features.to(image_features.device)
        _, scores_avg = self.compute_score(
            self.model,
            image_features_avg.unsqueeze(0),
            self.text_features,
        )
        _, indexes = torch.topk(scores_avg, self.topk)
        return indexes[0][0]

    def moving_average(self, data, window_size):
        # 1D moving average smoothing with edge padding
        padding_size = window_size
        padded_data = torch.cat(
            [
                torch.ones(padding_size, device=data.device) * data[0],
                data,
                torch.ones(padding_size, device=data.device) * data[-1],
            ]
        )
        kernel = (torch.ones(window_size, device=data.device) / window_size)
        smoothed = F.conv1d(padded_data.view(1, 1, -1), kernel.view(1, 1, -1))
        smoothed = smoothed.view(-1)[padding_size // 2 + 1 : -padding_size // 2]
        return smoothed

    def select_segments(self, similarity):
        # Thresholding rule depends on dataset
        if self.dataset == "thumos":
            mask = similarity > similarity.mean()
        elif self.dataset == "anet":
            mask = similarity > self.m
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        selected = torch.nonzero(mask).squeeze()
        segments = []

        if selected.numel() and selected.dim() > 0:
            interval_start = selected[0]
            for i in range(1, len(selected)):
                # Merge indices that are within stride
                if selected[i] <= selected[i - 1] + self.stride:
                    continue
                interval_end = selected[i - 1]
                if interval_start != interval_end:
                    segments.append([interval_start.item(), interval_end.item()])
                interval_start = selected[i]

            # Add the last interval
            if interval_start != selected[-1]:
                segments.append([interval_start.item(), selected[-1].item()])

        return segments

    def get_video_fps(self, video_name):
        # Fallback FPS reading for ActivityNet
        video_extensions = [".mp4", ".mkv", ".webm"]
        fps = None
        for ext in video_extensions:
            video_path = os.path.join(self.video_dir, video_name + ext)
            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                break
        return fps

    def compute_frequency_energy(self, proposals):
        # Compute frequency-domain energy as an "actionness" score for each proposal
        actionness_scores = []
        for proposal in proposals:
            proposal_np = proposal.cpu().numpy()
            freq_features = np.fft.fft(proposal_np, axis=0)
            energy = np.sum(np.abs(freq_features) ** 2, axis=0)
            energy_mean = np.mean(energy).item()
            actionness_scores.append(torch.tensor(energy_mean).sigmoid())
        return actionness_scores

    def logarithmic_weights(self, length, c=1.0):
        # Larger weight for positions closer to the segment boundary
        indices = np.arange(1, length + 1)
        weights = 1 / np.log(indices + c)
        weights /= weights.sum()
        return weights

    def mask_to_oic_score(
        self,
        mask,
        metric,
        weight_inner=1,
        weight_outter=-0.2,
        weight_max=1,
    ):
        """
        Compute OIC-style scores for each connected component in a binary mask.

        Args:
            mask: (t,) binary tensor
            metric: (t,) score tensor (e.g., similarity over time)

        Returns:
            out_detections: list of scalar scores, one per detected segment
        """
        if mask.device != metric.device:
            raise ValueError("mask and metric should be on the same device")

        mask_np = mask.cpu().numpy()
        metric_np = metric.cpu().numpy()

        out_detections = []
        detection_map = measure.label(mask_np, background=0)
        detection_num = detection_map.max()
        t = len(mask_np)

        for detection_id in range(1, detection_num + 1):
            start = np.where(detection_map == detection_id)[0].min()
            end = np.where(detection_map == detection_id)[0].max() + 1

            length = end - start
            inner_area = metric_np[detection_map == detection_id]

            left_start = max(int(start - length * 0.25), 0)
            right_end = min(int(end + length * 0.25), t + 1)

            outter_area_left = metric_np[left_start:start]
            outter_area_right = metric_np[end:right_end]

            if outter_area_left.shape[0] == 0 and outter_area_right.shape[0] == 0:
                detection_score = inner_area.mean() * weight_inner + inner_area.max() * weight_max
            else:
                weight_left = self.logarithmic_weights(outter_area_left.shape[0])[::-1]
                weight_right = self.logarithmic_weights(outter_area_right.shape[0])

                outter_left = outter_area_left * weight_left if outter_area_left.shape[0] else 0.0
                outter_right = outter_area_right * weight_right if outter_area_right.shape[0] else 0.0

                detection_score = (
                    inner_area.mean() * weight_inner
                    + (outter_left.sum() if isinstance(outter_left, np.ndarray) else outter_left) * weight_outter
                    + (outter_right.sum() if isinstance(outter_right, np.ndarray) else outter_right) * weight_outter
                    + inner_area.max() * weight_max
                )

            out_detections.append(detection_score)

        return out_detections

    def forward(self, x):
        idx, video_name, image_features = x
        video_name = video_name[0]

        # Apply visual projection (match COCA visual embedding space)
        with torch.no_grad():
            image_features = image_features @ self.model.visual.proj

        image_features = image_features.squeeze(0)

        # (1) Video-level pseudo-labeling
        indexes = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]
        pseudolabel_feature = self.text_features[indexes].squeeze()
        pseudolabel_feature = pseudolabel_feature / pseudolabel_feature.norm(dim=-1, keepdim=True)

        # Optional background removal using the average embedding
        if self.remove_background:
            image_features = image_features - self.background_embedding

        # (2) Self-supervised refinement via similarity curve
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = self.model.logit_scale.exp() * (pseudolabel_feature @ image_features_norm.T)

        # Keep an unsmoothed copy for OIC scoring
        similarity_hard = similarity.detach().clone().squeeze()

        if self.dataset == "thumos":
            similarity = self.moving_average(similarity, self.kernel_size).squeeze()

        if self.normalize:
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())

        # FPS source differs by dataset
        if self.dataset == "thumos":
            fps = self.all_video_fps["database"][video_name]["fps"]
        elif self.dataset == "anet":
            fps = self.get_video_fps(video_name)
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        # Ground-truth segments (frame indices)
        segments_gt = [anno["segment"] for anno in self.annotations[video_name]["annotations"]]
        segments_gt = [[int(float(seg[0]) * fps), int(float(seg[1]) * fps)] for seg in segments_gt]
        label_gt = [anno["label"] for anno in self.annotations[video_name]["annotations"]]
        unique_labels = set(label_gt)

        # Proposals from thresholded similarity
        segments = self.select_segments(similarity)

        pred_mask = torch.zeros(image_features.shape[0], device=image_features.device)
        gt_mask = torch.zeros(image_features.shape[0], device=image_features.device)

        # (3) Actionness (frequency energy) for each proposal
        actionness = None
        if len(segments) >= 1:
            image_features_per_segment = [image_features[seg[0] : seg[1]] for seg in segments]
            actionness = self.compute_frequency_energy(image_features_per_segment)

        if segments:
            # Segment-level pooled features for classification
            seg_feats = [torch.mean(image_features[seg[0] : seg[1]], dim=0) for seg in segments]
            seg_feats = torch.stack(seg_feats)

            pred, scores = self.compute_score(
                self.model,
                seg_feats,
                self.text_features.to(seg_feats.device),
            )

            for seg in segments:
                pred_mask[seg[0] : seg[1]] = 1
            for anno in segments_gt:
                gt_mask[anno[0] : anno[1]] = 1

            # OIC scoring over detected regions
            OIC_score = self.mask_to_oic_score(pred_mask, similarity_hard)

            output = [
                {
                    "label": indexes.item(),
                    # "score": scores[i],
                    "segment": segments[i],
                    "score": OIC_score[i] * (actionness[i].item() if actionness is not None else 1.0),
                }
                for i in range(pred.shape[0])
            ]
        else:
            output = [{"label": -1, "score": 0, "segment": []}]

        if self.visualize:
            self.plot_visualize(video_name, similarity, indexes, segments_gt, segments, unique_labels)
            return (video_name, output, pred_mask, gt_mask, unique_labels, plt)
        else:
            return (video_name, output, pred_mask, gt_mask, unique_labels, None)
