import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import os
from utils import get_data_loaders
from model.model import Model
from WEvade import WEvade_W, WEvade_W_binary_search_r

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd


MEAN, STD = 0.5, 0.5


def transform_image(image, device):
    image = image.to(device)
    return torch.clamp(image, min=-1, max=1)


def normalize_img_np(img_np):
    return img_np * STD + MEAN


def plot_image(image_arr, save_name):
    figure, ax = plt.subplots(ncols=1, nrows=1)
    ax.imshow(image_arr)
    plt.savefig(save_name)
    plt.close(figure)


def main(args):
    device = torch.device("cuda")
    vis_root_dir = os.path.join(
        ".", "visualization"
    )
    os.makedirs(vis_root_dir, exist_ok=True)
    # Load model.
    model = Model(args.image_size, args.watermark_length, device)
    checkpoint = torch.load(args.checkpoint)
    model.encoder.load_state_dict(checkpoint['enc-model'])
    model.decoder.load_state_dict(checkpoint['dec-model'])
    model.encoder.eval()
    model.decoder.eval()

    # Load dataset.
    data_loader = get_data_loaders(args.image_size, args.dataset_folder)
    dataiter = iter(data_loader)

    # === Read a image ===
    for _ in range(2):
        next(dataiter)
    images_tensor, labels_tensor = next(dataiter)
    img_np = np.transpose(images_tensor.cpu().numpy()[0, :, :, :], [1, 2, 0])

    # === Visualize Figure ===
    save_name = os.path.join(vis_root_dir, "image_orig.png")
    plot_image(normalize_img_np(img_np), save_name)

    # ==== Perform a svd decomp ====
    u1_list, s1_list, v1_list = [], [], []
    for axis in range(3):
        array = img_np[:, :, axis]
        U, s1, V = svd(array)
        # print("Check SVD reconstrcut: ")
        # recon_arr = U @ np.diag(s1) @ V
        # recon_arr = np.sum(np.abs(recon_arr - array))
        # print("   Error: {}".format(recon_arr))
        u1_list.append(U)
        s1_list.append(s1)
        v1_list.append(V)

    # Visualize singular values
    figure, ax = plt.subplots(ncols=1, nrows=3)
    for idx in range(3):
        s = s1_list[idx]
        ax[idx].bar(list(range(len(s))), s)
    save_name = os.path.join(vis_root_dir, "svd_orig.png")
    plt.savefig(save_name)
    plt.close(figure)

    # === Load Watermark ===
    gt_watermark_np = np.load('./watermark/watermark_coco.npy')
    groundtruth_watermark = torch.from_numpy(gt_watermark_np).to(device)
    
    # === process image and apply watermarking ===
    images_tensor = images_tensor.to(device)

    # input range [-1, 1]
    images_tensor = transform_image(images_tensor, device)
    watermarked_image = model.encoder(images_tensor, groundtruth_watermark)
    watermarked_image = transform_image(watermarked_image, device=device)  # output range [-1, 1] enforcement using clipping

    # === rescale into [0, 1] for plotting ===
    watermarked_image_np = (watermarked_image.detach().cpu().numpy()[0, :, :, :] + 1) / 2.
    watermarked_img_np = np.transpose(
        watermarked_image_np,
        [1, 2, 0]
    )

    save_name = os.path.join(vis_root_dir, "image_watermarked.png")
    plot_image(watermarked_img_np, save_name)

    # ==== Perform a svd decomp ====
    u2_list, s2_list, v2_list = [], [], []
    for axis in range(3):
        array = watermarked_img_np[:, :, axis]
        U, s2, V = svd(array)
        u2_list.append(U)
        s2_list.append(s2)
        v2_list.append(V)

    # Visualize singular values
    figure, ax = plt.subplots(ncols=1, nrows=3)
    for idx in range(3):
        s = s2_list[idx]
        ax[idx].bar(list(range(len(s))), s)
    save_name = os.path.join(vis_root_dir, "svd_watermarked.png")
    plt.savefig(save_name)
    plt.close(figure)

    # Visualize diff singular values
    figure, ax = plt.subplots(ncols=1, nrows=3)
    for idx in range(3):
        s_diff = s1_list[idx] - s2_list[idx]
        ax[idx].bar(list(range(len(s_diff))), s_diff)
    save_name = os.path.join(vis_root_dir, "svd_diff.png")
    plt.savefig(save_name)
    plt.close(figure)


    # === Visualize Diff Image ===
    img_diff = ((watermarked_img_np - img_np) + 2) / 4
    save_name = os.path.join(vis_root_dir, "image_diff.png")
    plot_image(img_diff, save_name)

    # ==== Try an SVD decomp ===
    recon_img_list = []
    for idx in range(3):
        U = u2_list[idx]
        V = v2_list[idx]
        s = s2_list[idx]
        # === Manipulat the singular values ===
        end = 20
        energy_ratio = np.sum(s[0:end]) / np.sum(s)
        print("Energy Ratio: ", energy_ratio)
        factor = 0.9
        s_top = s[0:end]
        s = s * (1 + (1-factor)*energy_ratio/(1-energy_ratio))
        s[0:end] = s_top * factor

        recon_arr = U @ np.diag(s) @ V
        recon_img_list.append(recon_arr)
    recon_img = np.stack(recon_img_list, 2)
    print("Recognize image shape: ", recon_img.shape)
    print()

    figure, ax = plt.subplots(ncols=1, nrows=1)
    ax.imshow(recon_img)
    save_name = os.path.join(vis_root_dir, "image_recon.png")
    plt.savefig(save_name)
    plt.close(figure)

    # === Decode Watermark ===
    watermark_decoded_raw = model.decoder(watermarked_image)
    watermark_decoded = watermark_decoded_raw.round().clip(0, 1).to(dtype=torch.long)
    watermark_decoded_np = watermark_decoded.detach().cpu().numpy()
    print("Orig Watermark: ", gt_watermark_np)
    print("Decoded       : ", watermark_decoded_np)
    # print("  Raw         : ", watermark_decoded_raw.detach().cpu().numpy())

    # === Decode a watermark from the "reconstructed" image ===
    recon_img = np.transpose(recon_img, [2, 0, 1])
    recon_img = recon_img[np.newaxis, :, :, :]
    recon_img_tensor = torch.from_numpy(recon_img).to(device, dtype=torch.float)
    recon_img_tensor = (recon_img_tensor - MEAN) / STD
    watermark_recon_raw = model.decoder(recon_img_tensor)
    watermark_recon = watermark_recon_raw.round().clip(0, 1).to(dtype=torch.long)
    watermarK_recon_np = watermark_recon.detach().cpu().numpy()
    print("Recon         : ", watermarK_recon_np)

    # === Decode a watermark from "natural" image ===
    natural_watermark_raw = model.decoder(images_tensor)
    natural_watermark = natural_watermark_raw.round().clip(0, 1).to(dtype=torch.long)
    natural_watermark_np = natural_watermark.detach().cpu().numpy()
    print("Natural       : ", natural_watermark_np)
    # print("  Raw         : ", natural_watermark_raw.detach().cpu().numpy())

    print("Bit-wise accuracy: ")
    print("Decoded: ", np.mean(watermark_decoded_np == gt_watermark_np) * 100)
    print("Recon  : ", np.mean(watermarK_recon_np == gt_watermark_np) * 100)
    print("Natural: ", np.mean(natural_watermark_np == gt_watermark_np) * 100)


if __name__ == "__main__":
    print("This script is to play around the watermark system and have fun.")
    parser = argparse.ArgumentParser(description='WEvade-W Arguments.')
    parser.add_argument('--checkpoint', default='./ckpt/coco.pth', type=str, help='Model checkpoint file.')
    parser.add_argument('--dataset-folder', default='./dataset/coco/val', type=str, help='Dataset folder path.')
    parser.add_argument('--image-size', default=128, type=int, help='Size of the images (height and width).')
    parser.add_argument('--watermark-length', default=30, type=int, help='Number of bits in a watermark.')
    parser.add_argument('--tau', default=0.8, type=float, help='Detection threshold of the detector.')
    parser.add_argument('--iteration', default=5000, type=int, help='Max iteration in WEvdae-W.')
    parser.add_argument('--epsilon', default=0.01, type=float, help='Epsilon used in WEvdae-W.')
    parser.add_argument('--alpha', default=0.1, type=float, help='Learning rate used in WEvade-W.')
    parser.add_argument('--rb', default=2, type=float, help='Upper bound of perturbation.')
    parser.add_argument('--WEvade-type', default='WEvade-W-II', type=str, help='Using WEvade-W-I/II.')
    parser.add_argument('--detector-type', default='double-tailed', type=str, help='Using double-tailed/single-tailed detctor.')
    # In our algorithm, we use binary-search to obtain perturbation upper bound. But in the experiment, we find
    # binary-search actually has no significant effect on the perturbation results. And we reduce time cost if not
    # using binary-search.
    parser.add_argument('--binary-search', default=False, type=bool, help='Whether use binary-search to find perturbation.')

    args = parser.parse_args()
    main(args)
    print("Completed.")