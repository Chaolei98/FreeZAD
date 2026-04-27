import torch
from torch import nn
from torch.functional import F
from src.models.components.loss import ByolLoss
import open_clip
import matplotlib.pyplot as plt
import json
import os
import cv2
import re
import copy
import importlib
from skimage import measure
import numpy as np
import math
import random
import time

tokenize = open_clip.get_tokenizer("coca_ViT-L-14")

class AdaZAD(nn.Module):
    def __init__(
        self,
        p: float,
        stride: int,
        randper: int,
        kernel_size: int,
        n: int,
        normalize: bool,
        dataset: str,
        visualize: bool,
        text_projection: bool,
        text_encoder: bool,
        image_projection: bool,
        logit_scale: bool,
        remove_background: bool,
        ltype: str,
        steps: int,
        refine_with_captions: bool,
        split: int,
        setting: int,
        video_path: str,
    ):
        super(AdaZAD, self).__init__()

        self.stride = stride
        self.randper = randper
        self.p = p
        self.n = n
        self.normalize = normalize
        self.text_projection = text_projection
        self.text_encoder = text_encoder
        self.image_projection = image_projection
        self.logit_scale = logit_scale
        self.remove_background = remove_background
        self.ltype = ltype
        self.steps = steps
        self.refine_with_captions = refine_with_captions
        self.split = split
        self.setting = setting
        self.dataset = dataset
        self.visualize = visualize
        self.kernel_size = kernel_size
        self.video_path = video_path
        self.topk = 3
        self.m = 0.7

        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
        )
        self.model = self.model.float()
        print(f"Loaded COCA model")

        if self.dataset == "thumos":
            dict_test_name = (
                f"t2_dict_test_thumos_{split}"
                if self.setting == 50
                else f"t1_dict_test_thumos_{split}" if self.setting == 75 else None
            )
            self.annotations_path = "./data/thumos_annotations/thumos_anno_action.json"
            self.video_dir = os.path.join(self.video_path)
        elif self.dataset == "anet":
            dict_test_name = (
                f"t2_dict_test_{split}"
                if self.setting == 50
                else f"t1_dict_test_{split}" if self.setting == 75 else None
            )
            self.annotations_path = (
                "./data/activitynet_annotations/anet_anno_action.json"
            )
            self.video_dir = os.path.join(self.video_path)
        else:
            raise ValueError(f"Not implemented dataset: {self.dataset}")

        self.dict_test = getattr(
            importlib.import_module("config.zero_shot"), dict_test_name, None
        )
        self.cls_names = self.dict_test
        self.num_classes = len(self.cls_names)
        self.inverted_cls = {v: k for k, v in self.cls_names.items()}
        self.text_features = self.get_text_features(self.model)

        with open(self.annotations_path, "r") as f:
            self.annotations = json.load(f)

        if self.ltype == "BCE":
            self.tta_loss = torch.nn.BCEWithLogitsLoss()
        elif "BYOL" in self.ltype:
            self.tta_loss = ByolLoss()
        else:
            raise ValueError(f"Not implemented loss type: {self.ltype}")

    def get_text_features(self, model):
        prompts = []
        for c in self.cls_names:
            c = re.sub(r"([a-z])([A-Z])", r"\1 \2", c)
            prompts.append("a video of action" + " " + c)

        text = [tokenize(p) for p in prompts]
        text = torch.stack(text)
        text = text.squeeze()
        text = text.to(next(model.parameters()).device)
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_score(self, x, y):
        x = x / x.norm(dim=-1, keepdim=True)
        scores = (self.model.logit_scale.exp() * x @ y.T).softmax(dim=-1)
        pred = scores.argmax(dim=-1)
        return pred, scores

    def select_segments(self, similarity):
        
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
                if selected[i] <= selected[i - 1] + self.stride:
                    continue
                else:
                    interval_end = selected[i - 1]
                    if interval_start != interval_end:
                        segments.append([interval_start.item(), interval_end.item()])
                    interval_start = selected[i]

            if interval_start != selected[-1]:
                segments.append([interval_start.item(), selected[-1].item()])

        return segments

    def get_video_fps(self, video_name):
        video_extensions = [".mp4", ".mkv", ".webm"]
        for ext in video_extensions:
            video_path = os.path.join(self.video_dir, video_name + ext)

            if os.path.exists(video_path):
                fps = cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FPS)
                break
        return fps

    def infer_pseudo_labels(self, image_features):
        image_features_avg = image_features.mean(dim=0)
        self.background_embedding = image_features_avg.unsqueeze(0)
        self.text_features = self.text_features.to(image_features.device)
        _, scores_avg = self.compute_score(
            image_features_avg.unsqueeze(0),
            self.text_features,
        )
        _, indexes = torch.topk(scores_avg, self.topk)
        return indexes[0][0], scores_avg

    def moving_average(self, data, window_size):
        padding_size = window_size
        padded_data = torch.cat(
            [
                torch.ones(padding_size).to(data.device) * data[0],
                data,
                torch.ones(padding_size).to(data.device) * data[-1],
            ]
        )
        kernel = (torch.ones(window_size) / window_size).to(data.device)
        smoothed_data = F.conv1d(padded_data.view(1, 1, -1), kernel.view(1, 1, -1))
        # Crop the smoothed data and remove the padded part to keep the same length as the original data
        smoothed_data = smoothed_data.view(-1)[
            padding_size // 2 + 1 : -padding_size // 2
        ]
        return smoothed_data

    def get_segments_gt(self, video_name, fps):
        segments_gt = [
            anno["segment"]
            for anno in self.annotations[video_name]["annotations"]
            if anno["label"] in self.cls_names
        ]
        segments_gt = [
            [int(float(seg[0]) * fps), int(float(seg[1]) * fps)] for seg in segments_gt
        ]
        label_gt = [
            anno["label"]
            for anno in self.annotations[video_name]["annotations"]
            if anno["label"] in self.cls_names
        ]
        unique_labels = set(label_gt)
        return segments_gt, unique_labels

    def get_indices(self, signal):

        if (100 * self.n) >= signal.shape[1]:
            nindices = torch.arange(signal.shape[1]).to("cuda")
        else:
            nindices = torch.topk(-signal, (100 * self.n) % signal.shape[1])[1]

        nindices = nindices.squeeze().sort()[0]

        if nindices.shape[0] < self.n:
            nindices = nindices.repeat_interleave(self.n // nindices.shape[0] + 1)
            nindices = nindices[: self.n]
        
        # Select elements at positions such as 0, 133, 266, and 399; the elements are temporal indices
        nindices = nindices[:: (len(nindices) - 1) // (self.n - 1)][: self.n]

        # Randomly sample around positions such as 0, 133, 266, and 399
        nindices = torch.clamp(
            nindices
            + torch.randint(-self.randper, self.randper, (self.n,)).to(signal.device),
            0,
            signal.shape[1] - 1,
        )

        return nindices

    def get_positive_frame(self, similarity):
        t = similarity.shape[0]
        max_index = torch.argmax(similarity).item()

        max_start_index = max(0, max_index - 15)
        max_end_index = min(t, max_index + 16)


        max_candidate_indices = list(range(max_start_index, max_end_index))

        pindices = random.sample(max_candidate_indices, self.n)

        return pindices

    def logarithmic_weights(self, length, c=1.0):
        indices = np.arange(1, length + 1)
        weights = 1 / np.log(indices + c)
        weights /= weights.sum()
        return weights

    def mask_to_oic_score(self, mask, metric, weight_inner=1, weight_outter=-0.2, weight_max=1):
        '''
            mask: shape with (t)
            metric: shape with (t)

        return:
            out_detections: List with length detection_num
        '''
        # Ensure that mask and metric are on the same device
        if mask.device != metric.device:
            raise ValueError("mask and metric should be on the same device")
        device = metric.device
        mask_np = mask.cpu().numpy()
        metric_np = metric.cpu().numpy()

        out_detections = []
        detection_map = measure.label(mask_np, background=0)
        detection_num = detection_map.max()
        t = len(mask_np)

        for detection_id in range(1, detection_num + 1):

            start = np.where(detection_map == detection_id)[0].min()
            end = np.where(detection_map == detection_id)[0].max() + 1  # Add 1 because the end index is exclusive

            length = end - start

            inner_area = metric_np[detection_map == detection_id]

            left_start = max(int(start - length * 0.25), 0)
            right_end = min(int(end + length * 0.25), t+1)

            outter_area_left = metric_np[left_start:start]
            outter_area_right = metric_np[end:right_end]

            outter_area = np.concatenate((outter_area_left, outter_area_right),
                                        axis=0)

            if outter_area.shape[0] == 0:
                detection_score = inner_area.mean() * weight_inner + inner_area.max() * weight_max
            else:

                weight_left = self.logarithmic_weights(outter_area_left.shape[0])[::-1]
                weight_right = self.logarithmic_weights(outter_area_right.shape[0])
                outter_left = outter_area_left * weight_left
                outter_right = outter_area_right * weight_right

                detection_score = (inner_area.mean() * weight_inner +
                                outter_left.sum() * weight_outter +
                                outter_right.sum() * weight_outter +
                                + inner_area.max() * weight_max)
                
            out_detections.append(detection_score)

        # Convert the results back to the original device

        return out_detections

    def compute_tta_embedding(self, class_label, device):
        class_label = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_label)
        class_label = "a video of action" + " " + class_label
        text = tokenize(class_label).to(device)
        tta_emb = self.model.encode_text(text)
        tta_emb = tta_emb / tta_emb.norm(dim=-1, keepdim=True)
        return tta_emb
    
    def compute_frequency_energy(self, proposals):
        actionness_scores = []
        for proposal in proposals:
            proposal = proposal.cpu().numpy()
            freq_features = np.fft.fft(proposal, axis=0)
            energy = np.sum(np.abs(freq_features)**2, axis=0)
            energy_mean = np.mean(energy).item()
            actionness_scores.append(torch.tensor(energy_mean).sigmoid())
        return actionness_scores

    def forward(self, x, optimizer):
        idx, video_name, image_features_pre = x
        image_features_pre = copy.deepcopy(image_features_pre)  # (1, t, d)
        video_name = video_name[0]
        fps = self.get_video_fps(video_name)

        if not self.image_projection:
            image_features = image_features_pre
            image_features = image_features.squeeze(0)
        else:
            image_features_pre.requires_grad = True
            with torch.no_grad():
                image_features = image_features_pre @ self.model.visual.proj    # (1,t,d) @ (d,d) -> (1,t,d)
                image_features = image_features.squeeze(0)  # (t, d)
        
        # video-level pseudo-labelling
        indexes, _ = self.infer_pseudo_labels(image_features)
        class_label = self.inverted_cls[indexes.item()]

        segments_gt, unique_labels = self.get_segments_gt(video_name, fps)

        # TTA training steps
        for _ in range(self.steps):
            if self.image_projection:
                image_features = (image_features_pre @ self.model.visual.proj).squeeze(
                    0
                )
            
            # self-supervised prediction refinement
            tta_emb = self.compute_tta_embedding(class_label, image_features.device)    # (1,d)
            
            features = image_features - self.background_embedding if self.remove_background else image_features
            similarity = self.model.logit_scale.exp() * tta_emb @ features.T    # (1,t)

            if self.dataset == "thumos":
                similarity = self.moving_average(
                    similarity.squeeze(), self.kernel_size
                ).unsqueeze(0)  # (1,t)
            
            # ----- ② Positive and negative sample selection -----
            pindices = self.get_positive_frame(similarity.squeeze())
            nindices = self.get_indices(similarity)

            image_features_p, image_features_n = image_features[pindices], image_features[nindices] # (4,d)
            image_features_p = image_features_p / image_features_p.norm(dim=-1, keepdim=True)
            image_features_n = image_features_n / image_features_n.norm(dim=-1, keepdim=True)
            similarity_p = (self.model.logit_scale.exp() * tta_emb @ image_features_p.T) # (1,4)
            similarity_n = (self.model.logit_scale.exp() * tta_emb @ image_features_n.T) # (1,4)
            similarity = torch.cat([similarity_p.squeeze(), similarity_n.squeeze()], dim=0)   # (8)
            gt = torch.cat(
                [
                    torch.ones(similarity_p.shape[1]),
                    torch.zeros(similarity_n.shape[1]),
                ],
                dim=0,
            ).to(similarity.device) # (8)
            
            
            if self.ltype in ["BYOL", "BCE"]:
                tta_loss = self.tta_loss(similarity, gt)
            elif self.ltype == "BYOLfeat":
                tta_loss = self.tta_loss(similarity, gt) + self.tta_loss(
                    image_features_p,   # (4,d)
                    tta_emb.repeat_interleave(image_features_p.shape[0], dim=0),   # (4,d)
                )
            else:
                raise ValueError(f"Not implemented loss type: {self.ltype}")

            tta_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            tta_emb = self.compute_tta_embedding(class_label, image_features.device)    # (1, d)
            
            if self.remove_background:
                image_features = image_features - self.background_embedding
            
            image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = self.model.logit_scale.exp() * tta_emb @ image_features_norm.T # (1,t)

            # similarity_hard: the similarity used for OIC scoring does not require moving average or normalization
            similarity_hard = copy.deepcopy(similarity)
            similarity_hard = similarity_hard.squeeze()
            
            if self.dataset == "thumos":
                similarity = self.moving_average(similarity.squeeze(), self.kernel_size)    # (t)
            if self.normalize:
                similarity = (similarity - similarity.min()) / (
                    similarity.max() - similarity.min()
                )
            similarity = similarity.squeeze()   # (t)
            segments = self.select_segments(similarity) # Generate proposals
            pred_mask = torch.zeros(image_features.shape[0]).to(image_features.device)  # (t,)
            gt_mask = torch.zeros(image_features.shape[0]).to(image_features.device)  # (t,) 
            
            #---- ③ actionness -----
            if len(segments) >= 1:
                image_features_per_segment = []
                for i, seg in enumerate(segments):
                    image_features_per_segment.append(image_features[seg[0] : seg[1]])

                actionness = self.compute_frequency_energy(image_features_per_segment)

            if segments:
                image_features = [
                    torch.mean(image_features[seg[0] : seg[1]], dim=0)
                    for seg in segments
                ]
                text_features = self.get_text_features(self.model)
                image_features = torch.stack(image_features)
                pred, scores = self.compute_score(
                    image_features,
                    text_features.to(image_features.device),
                )
                for seg in segments:
                    pred_mask[seg[0] : seg[1]] = 1
                for anno in segments_gt:
                    gt_mask[anno[0] : anno[1]] = 1

                # ----- ① OIC -----
                OIC_score = self.mask_to_oic_score(pred_mask, similarity_hard)    # (n)
                output = [
                    {
                        "label": indexes.item(),    # video-level pseudo-label
                        # "score": scores[i],         # (k,)
                        "segment": segments[i],     # [s_frame,e_frame]
                        "score": OIC_score[i] * actionness[i].item()
                    }
                    for i in range((len(segments)))
                ]
            else:
                output = [
                    {
                        "label": -1,
                        "score": 0,
                        "segment": [],
                        "OIC": 0,
                    }
                ]
       
        sim_plot = None
        return (
            video_name,
            output,
            pred_mask,
            gt_mask,
            unique_labels,
            sim_plot,
        )
