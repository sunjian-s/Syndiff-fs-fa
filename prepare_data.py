import argparse
import glob
import os
import random

import cv2
import numpy as np


def normalize(img):
    """Map uint8 RGB image to [-1, 1]."""
    return (img.astype(np.float32) / 127.5) - 1.0


def filter_image_files(file_list):
    img_suffixes = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    return [f for f in file_list if f.lower().endswith(img_suffixes)]


def get_file_pairs(fundus_dir, ffa_dir):
    if not os.path.exists(fundus_dir) or not os.path.exists(ffa_dir):
        print(f'Path missing:\n  {fundus_dir}\n  {ffa_dir}')
        return [], []

    files_fundus = sorted(filter_image_files(glob.glob(os.path.join(fundus_dir, '*'))))
    files_ffa = sorted(filter_image_files(glob.glob(os.path.join(ffa_dir, '*'))))

    if len(files_fundus) != len(files_ffa):
        print('Warning: fundus/ffa counts do not match, truncating to min length.')
        min_len = min(len(files_fundus), len(files_ffa))
        files_fundus = files_fundus[:min_len]
        files_ffa = files_ffa[:min_len]

    return files_fundus, files_ffa


def load_pair(fundus_path, ffa_path):
    img_fundus = cv2.imread(fundus_path)
    if img_fundus is None:
        return None, None
    img_fundus = cv2.cvtColor(img_fundus, cv2.COLOR_BGR2RGB)

    img_ffa = cv2.imread(ffa_path, cv2.IMREAD_GRAYSCALE)
    if img_ffa is None:
        return None, None
    img_ffa = cv2.cvtColor(img_ffa, cv2.COLOR_GRAY2RGB)
    return img_fundus, img_ffa


def safe_random_top_left(h, w, crop_size):
    if h <= crop_size or w <= crop_size:
        return 0, 0
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    return top, left


def crop_with_resize(img, top, left, crop_size, out_size):
    patch = img[top:top + crop_size, left:left + crop_size]
    if patch.shape[0] != crop_size or patch.shape[1] != crop_size:
        patch = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
        return patch
    if crop_size != out_size:
        patch = cv2.resize(patch, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return patch


def compute_lesion_score(ffa_rgb_patch, bright_percentile=97.0):
    gray = cv2.cvtColor(ffa_rgb_patch, cv2.COLOR_RGB2GRAY)
    thr = np.percentile(gray, bright_percentile)
    bright_mask = gray >= thr
    # Score encourages both bright lesion density and contrast.
    bright_ratio = float(bright_mask.mean())
    contrast = float(gray[bright_mask].mean() - gray.mean()) if bright_mask.any() else 0.0
    return bright_ratio + (contrast / 255.0)


def sample_patch_pair(img_fundus, img_ffa, strategy, out_size, large_crop_size, lesion_trials, lesion_percentile):
    h, w = img_fundus.shape[:2]

    if strategy == 'random256':
        crop_size = out_size
        top, left = safe_random_top_left(h, w, crop_size)
        return (
            crop_with_resize(img_fundus, top, left, crop_size, out_size),
            crop_with_resize(img_ffa, top, left, crop_size, out_size),
            False,
        )

    if strategy == 'resize512':
        crop_size = min(large_crop_size, h, w)
        top, left = safe_random_top_left(h, w, crop_size)
        return (
            crop_with_resize(img_fundus, top, left, crop_size, out_size),
            crop_with_resize(img_ffa, top, left, crop_size, out_size),
            False,
        )

    if strategy == 'lesion256':
        crop_size = out_size
        best_score = -1.0
        best_pair = None
        for _ in range(max(lesion_trials, 1)):
            top, left = safe_random_top_left(h, w, crop_size)
            fundus_patch = crop_with_resize(img_fundus, top, left, crop_size, out_size)
            ffa_patch = crop_with_resize(img_ffa, top, left, crop_size, out_size)
            score = compute_lesion_score(ffa_patch, bright_percentile=lesion_percentile)
            if score > best_score:
                best_score = score
                best_pair = (fundus_patch, ffa_patch)
        return best_pair[0], best_pair[1], True

    raise ValueError(f'Unsupported strategy: {strategy}')


def apply_random_flip(patch_fundus, patch_ffa):
    if random.random() > 0.5:
        patch_fundus = cv2.flip(patch_fundus, 1)
        patch_ffa = cv2.flip(patch_ffa, 1)
    if random.random() > 0.5:
        patch_fundus = cv2.flip(patch_fundus, 0)
        patch_ffa = cv2.flip(patch_ffa, 0)
    return patch_fundus, patch_ffa


def process_train_data(fundus_files, ffa_files, args):
    print(f'Building train set with strategy={args.train_strategy}, pairs={len(fundus_files)}')
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path) in enumerate(zip(fundus_files, ffa_files)):
        img_fundus, img_ffa = load_pair(f_path, a_path)
        if img_fundus is None or img_ffa is None:
            continue

        for _ in range(args.patches_per_img):
            patch_fundus, patch_ffa, _ = sample_patch_pair(
                img_fundus,
                img_ffa,
                strategy=args.train_strategy,
                out_size=args.image_size,
                large_crop_size=args.large_crop_size,
                lesion_trials=args.lesion_trials,
                lesion_percentile=args.lesion_percentile,
            )
            patch_fundus, patch_ffa = apply_random_flip(patch_fundus, patch_ffa)
            data_fundus_list.append(normalize(patch_fundus))
            data_ffa_list.append(normalize(patch_ffa))

        if (idx + 1) % 10 == 0:
            print(f'  processed {idx + 1} / {len(fundus_files)} train pairs')

    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)
    print(f'  train done. Fundus: {arr_fundus.shape}, FFA: {arr_ffa.shape}')
    return arr_fundus, arr_ffa


def process_eval_data(fundus_files, ffa_files, image_size):
    print(f'Building eval set with deterministic 5-crop, pairs={len(fundus_files)}')
    data_fundus_list = []
    data_ffa_list = []

    for idx, (f_path, a_path) in enumerate(zip(fundus_files, ffa_files)):
        img_fundus, img_ffa = load_pair(f_path, a_path)
        if img_fundus is None or img_ffa is None:
            continue

        h, w = img_fundus.shape[:2]
        if h <= image_size or w <= image_size:
            patch_fundus = cv2.resize(img_fundus, (image_size, image_size), interpolation=cv2.INTER_AREA)
            patch_ffa = cv2.resize(img_ffa, (image_size, image_size), interpolation=cv2.INTER_AREA)
            data_fundus_list.append(normalize(patch_fundus))
            data_ffa_list.append(normalize(patch_ffa))
            continue

        crops = [
            (0, 0),
            (0, w - image_size),
            (h - image_size, 0),
            (h - image_size, w - image_size),
            ((h - image_size) // 2, (w - image_size) // 2),
        ]

        for top, left in crops:
            patch_fundus = img_fundus[top:top + image_size, left:left + image_size]
            patch_ffa = img_ffa[top:top + image_size, left:left + image_size]
            data_fundus_list.append(normalize(patch_fundus))
            data_ffa_list.append(normalize(patch_ffa))

        if (idx + 1) % 10 == 0:
            print(f'  processed {idx + 1} / {len(fundus_files)} eval pairs')

    arr_fundus = np.array(data_fundus_list).transpose(0, 3, 1, 2)
    arr_ffa = np.array(data_ffa_list).transpose(0, 3, 1, 2)
    print(f'  eval done. Fundus: {arr_fundus.shape}, FFA: {arr_ffa.shape}')
    return arr_fundus, arr_ffa


def save_dataset_pair(output_dir, prefix1, prefix2, data1, data2):
    import scipy.io
    scipy.io.savemat(os.path.join(output_dir, prefix1), {'data': data1})
    scipy.io.savemat(os.path.join(output_dir, prefix2), {'data': data2})


def build_parser():
    parser = argparse.ArgumentParser(description='Build SynDiff paired mat datasets with configurable patch strategies.')
    parser.add_argument('--train_fundus_dir', type=str, default='./raw_images/fundus')
    parser.add_argument('--train_ffa_dir', type=str, default='./raw_images/ffa')
    parser.add_argument('--test_fundus_dir', type=str, default='./test_set/fundus')
    parser.add_argument('--test_ffa_dir', type=str, default='./test_set/ffa')
    parser.add_argument('--output_dir', type=str, default='./SynDiff_dataset')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--patches_per_img', type=int, default=10)
    parser.add_argument('--train_strategy', type=str, default='random256',
                        choices=['random256', 'resize512', 'lesion256'])
    parser.add_argument('--large_crop_size', type=int, default=512,
                        help='crop size used by resize512 strategy before resizing to image_size')
    parser.add_argument('--lesion_trials', type=int, default=12,
                        help='number of random candidates evaluated per patch in lesion256 strategy')
    parser.add_argument('--lesion_percentile', type=float, default=97.0,
                        help='bright percentile used to score FFA lesion candidates')
    parser.add_argument('--seed', type=int, default=42)
    return parser


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print('\n--- Building train dataset ---')
    train_fundus, train_ffa = get_file_pairs(args.train_fundus_dir, args.train_ffa_dir)
    if train_fundus:
        train_c1, train_c2 = process_train_data(train_fundus, train_ffa, args)
        save_dataset_pair(
            args.output_dir,
            'data_train_contrast1.mat',
            'data_train_contrast2.mat',
            train_c1,
            train_c2,
        )
    else:
        print('Skip train set: no valid train pairs found.')

    print('\n--- Building test/val dataset ---')
    test_fundus, test_ffa = get_file_pairs(args.test_fundus_dir, args.test_ffa_dir)
    if test_fundus:
        test_c1, test_c2 = process_eval_data(test_fundus, test_ffa, args.image_size)
        save_dataset_pair(
            args.output_dir,
            'data_test_contrast1.mat',
            'data_test_contrast2.mat',
            test_c1,
            test_c2,
        )
        save_dataset_pair(
            args.output_dir,
            'data_val_contrast1.mat',
            'data_val_contrast2.mat',
            test_c1,
            test_c2,
        )
    else:
        print('Skip test/val set: no valid eval pairs found.')

    print(f'\nDone. Dataset saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
