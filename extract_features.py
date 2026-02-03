import os
import cv2
import torch
from src.data.components.utils import map_reduce, transform, load_json
import pandas as pd
import json
from PIL import Image
import open_clip
import numpy as np
import time
from transformers import CLIPTokenizer, CLIPModel
import torch


video_info_path = "./data/activitynet_annotations/video_info_new.csv"
video_anno_path = "./data/thumos_annotations/thumos_anno_action.json"

video_path_file = "./dataset/thumos_video/val_test"
# video_path_file = "./dataset/anet_video/train_val"

def process_frames(allframes):
    allframes = [
        transform()(Image.fromarray((frame).astype("uint8")).convert("RGB"))
        for frame in allframes
    ]
    return allframes

def get_video_info(dataset):
    """
    Return:
        video_infos: a dict
        {
            "video_name": {"duration_second": XX, "subset": XX},
            ...
        }
    """
    video_infos = {}
    if dataset == "anet":
        dataset_info = pd.DataFrame(pd.read_csv(video_info_path)).values[:]
        for info in dataset_info:
            video_infos[info[0]] = {"duration": info[2], "subset": info[5]}
    elif dataset == "thumos":
        dataset_info = json.load(open(video_anno_path))
        for info in dataset_info.keys():
            video_infos[info] = {
                "duration": dataset_info[info]["duration_second"],
                "subset": info.split("_")[1],
            }
    else:
        raise NotImplementedError("Dataset not implemented")
    return video_infos

def loadVideo(idx):

    video_extensions = [".mp4", ".mkv", ".webm"]
    video = None
    for ext in video_extensions:
        video_path = os.path.join(video_path_file, f"{idx}{ext}")
        if os.path.exists(video_path):
            video = cv2.VideoCapture(video_path)
            break

    if video is None or not video.isOpened():
        # raise Exception(
        #     f"Video is not opened! {os.path.join(video_path, idx)}"
        # )
        return None, None
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read non-empty frames
    allframes = [
        frame
        for rval, frame in (video.read() for _ in range(num_frames))
        if frame is not None
    ]

    # Preprocess each frame
    allframes = map_reduce(process_frames, num_workers=8, reduce="sum")(
        allframes
    )
    video_data = torch.stack(allframes, dim=0)

    return video_data, fps

def get_image_features(images):
    chunk_size = 100  # Process 100 frames per batch
    t = images.shape[0]
    image_features = []
    
    with torch.no_grad():  # Disable autograd to reduce memory usage

        for i in range(t // chunk_size):
            # Encode image features for the current batch
            current_images = images[i * chunk_size : (i + 1) * chunk_size].cuda()
            # features = encode_function(current_images, normalize=True)
            features = encode_function(current_images)
            image_features.append(features)
            
            # Release GPU memory after processing this batch
            del features
            del current_images
            torch.cuda.empty_cache()

        if t % chunk_size != 0:
            # Process the remaining frames
            current_images = images[t - t % chunk_size :].cuda()
            # features = encode_function(current_images, normalize=True)
            features = encode_function(current_images)
            image_features.append(features)
            
            # Release GPU memory after processing the last batch
            del features
            del current_images
            torch.cuda.empty_cache()

    return image_features

if __name__ == "__main__":
    # Output directory
    save_dir = "./dataset/thumos_features"
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)


    #===== CoCa =====
    model, _, _ = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14", pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    # Replace the projection matrix with an identity matrix
    with torch.no_grad():
        eye_matrix = torch.eye(model.visual.proj.shape[0], model.visual.proj.shape[1]).cuda()
        model.visual.proj.copy_(eye_matrix)
    model = model.float().cuda()
    encode_function = model.encode_image
    #===================

    #===== CLIP-16 =====
    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").float()
    # Replace the visual_projection layer with an identity mapping
    # model.visual_projection = torch.nn.Identity()
    # model = model.float().cuda()
    # encode_function = model.get_image_features
    #===================

    video_infos = get_video_info("thumos")
    video_names = video_infos.keys()
    video_names = list(video_names)[::-1]
    
    for step, name in enumerate(video_names):

        # Build output file path
        print("----------------------------------------")
        save_path = os.path.join(save_dir, f"{name}.npy")

        # Skip if the feature file already exists
        if os.path.exists(save_path):
            print(f"File {name} already exists. Skipping...")
            continue

        # Record start time for this video
        start_time = time.time()
        # ① video -> (T,3,H,W)
        print(f"Now loading {name} video to images")
        images, fps = loadVideo(name)
        if images is None:
            print(f"File {name} has no images!!!")
            continue
        print(name, "temporal num:", images.shape[0])

        # ② (T,3,H,W) -> (T,D)
        images = images.squeeze(0)

        image_features = get_image_features(images)
        image_features = torch.cat(image_features, dim=0)

        # Convert torch Tensor to numpy array
        image_features_np = image_features.cpu().numpy()

        # Save to .npy file
        np.save(save_path, image_features_np)

        # Manually release CUDA memory
        del image_features
        torch.cuda.empty_cache() 
        
        # Record end time for this video
        end_time = time.time()

        # Print elapsed time
        elapsed_time = end_time - start_time
        print(f"Finished step {step+1}, Time taken: {elapsed_time:.2f} seconds")
