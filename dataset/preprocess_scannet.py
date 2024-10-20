import os
import sys
root_path = os.path.abspath(__file__)
root_path = '/'.join(root_path.split('/')[:-3])
sys.path.append(root_path)

import cv2
import glob
import json
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from multiprocessing import cpu_count, Pool


def get_keyframe_indices(filenames, window_size):
    """
        select non-blurry images within a moving window
    """
    scores = []
    cores = cpu_count() - 1
    # print("Using", cores, "cores")
    with Pool(cores) as pool:
        for result in pool.map(compute_blur_score_opencv, filenames) :
            scores.append(result)
    keyframes = [i + np.argmin(scores[i:i + window_size]) for i in range(0, len(scores), window_size)]
    return keyframes, scores


def compute_blur_score_opencv(filename):
    """
    Estimate the amount of blur an image has with the variance of the Laplacian.
    Normalize by pixel number to offset the effect of image size on pixel gradients & variance
    https://github.com/deepfakes/faceswap/blob/ac40b0f52f5a745aa058f92339302065177dd28b/tools/sort/sort.py#L626
    """
    image = cv2.imread(str(filename))
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_map = cv2.Laplacian(image, cv2.CV_32F)
    score = np.var(blur_map) / np.sqrt(image.shape[0] * image.shape[1])
    return 1.0 - score


def subsample_scannet(src_folder, rate):
    """
    sample every nth frame from scannet
    """
    all_frames = sorted(list(x.stem for x in (src_folder / 'pose').iterdir()), key=lambda y: int(y) if y.isnumeric() else y)
    total_sampled = int(len(all_frames) * rate)
    sampled_frames = [all_frames[i * (len(all_frames) // total_sampled)] for i in range(total_sampled)]
    unsampled_frames = [x for x in all_frames if x not in sampled_frames]
    for frame in sampled_frames:
        if 'inf' in Path(src_folder / "pose" / f"{frame}.txt").read_text():
            unsampled_frames.append(frame)
    folders = ["color", "depth", "instance", "pose", "semantics"]
    exts = ['jpg', 'png', 'png', 'txt', 'png']
    for folder, ext in tqdm(zip(folders, exts), desc='sampling'):
        assert (src_folder / folder).exists(), src_folder
        for frame in unsampled_frames:
            if (src_folder / folder / f'{frame}.{ext}').exists():
                os.remove(str(src_folder / folder / f'{frame}.{ext}'))
            else:
                print(str(src_folder / folder / f'{frame}.{ext}'), "already exists!")


def subsample_scannet_blur_window(src_folder, min_frames):
    """
    sample non blurry frames from scannet
    """
    if os.path.exists(src_folder / f"sampled_frames.json"):
        print("sampled_frames.json already exists, skipping")
        return
    scene_name = src_folder.name
    all_frames = sorted(list(x.stem for x in (src_folder / f'pose').iterdir()), key=lambda y: int(y) if y.isnumeric() else y)
    valid_frames = []
    for frame in all_frames:
        if 'inf' not in Path(src_folder / f"pose" / f"{frame}.txt").read_text():
            valid_frames.append(frame)
    print("Found", len(all_frames), "frames, ", len(valid_frames), "are valid")
    valid_frame_paths = [Path(src_folder / f"color" / f"{frame}.jpg") for frame in valid_frames]
    window_size = max(3, len(valid_frames) // min_frames)
    frame_indices, _ = get_keyframe_indices(valid_frame_paths, window_size)
    print("Using a window size of", window_size, "got", len(frame_indices), "frames")
    sampled_frames = [valid_frames[i] for i in frame_indices]
    # save as json
    json.dump(sampled_frames, open(src_folder / f"sampled_frames.json", 'w'), indent=4)


def resize_files(img_paths, resize_depth=False):
    for img_path in img_paths:
        img = Image.open(img_path) # 1296x968
        if not os.path.exists(img_path.replace('color', 'color_512512')):
            img1 = img.resize([512, 512], Image.Resampling.LANCZOS)
            img1.save(img_path.replace('color', 'color_512512'))
        if not os.path.exists(img_path.replace('color', 'color_480640')):
            img2 = img.resize([640, 480], Image.Resampling.LANCZOS)
            img2.save(img_path.replace('color', 'color_480640'))
        if resize_depth and not os.path.exists(img_path.replace('color', 'depth_512512')):
            p_depth = img_path.replace('color', 'depth').replace('.jpg', '.png')
            depth = Image.open(p_depth)
            depth1 = depth.resize([512, 512], Image.Resampling.NEAREST)
            depth1.save(p_depth.replace('depth', 'depth_512512'))


def process_one_scene(scene_path):
    dest = scene_path
    print('#' * 80)
    scene_name = path.split('/')[-1]
    print(f'subsampling from {scene_name}...')
    subsample_scannet_blur_window(dest, min_frames=400)

    print('resizing images...')
    os.makedirs(f'{path}/color_512512', exist_ok=True)
    os.makedirs(f'{path}/color_480640', exist_ok=True)
    os.makedirs(f'{path}/depth_512512', exist_ok=True)

    img_ids = json.load(open(f'{path}/sampled_frames.json', 'r'))
    img_paths = [f'{path}/color/{img_id}.jpg' for img_id in img_ids]
    resize_files(img_paths, resize_depth=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='scannet preprocessing')
    parser.add_argument('--data_root', required=False, default='/home/yuliu/Dataset/scannet', help='file path')
    args = parser.parse_args()

    scene_paths = sorted(glob.glob(f'{args.data_root}/*'))
    for path in scene_paths:
        scene_root = Path(path)
        process_one_scene(scene_root)
        
    test_root = args.data_root.replace('scannet', 'scannet_test')
    scene_paths = sorted(glob.glob(f'{test_root}/*'))
    for path in scene_paths:
        scene_name = path.split('/')[-1]
        print(f'resizing images for {scene_name}...')
        os.makedirs(f'{path}/color_512512', exist_ok=True)
        os.makedirs(f'{path}/color_480640', exist_ok=True)
        os.makedirs(f'{path}/depth_512512', exist_ok=True)

        img_paths = glob.glob(f'{path}/color/*.jpg')
        resize_files(img_paths, resize_depth=True)
        
