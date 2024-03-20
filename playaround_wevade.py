import torch
import torch.nn as nn
import argparse
import time
from tqdm import tqdm
import os
from utils import get_data_loaders, transform_image, AverageMeter
from model.model import Model
from WEvade import WEvade_W, WEvade_W_binary_search_r

import numpy as np
import matplotlib.pyplot as plt


MEAN, STD = 0.5, 0.5


def transform_image(image, device):
    image = image.to(device)
    return torch.clamp(image, min=-1, max=1)


def normalize_img_np(img_np):
    return img_np * STD + MEAN


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
    for _ in range(5):
        next(dataiter)
    images_tensor, labels_tensor = next(dataiter)
    img_np = np.transpose(images_tensor.cpu().numpy()[0, :, :, :], [1, 2, 0])

    figure = plt.figure()
    plt.imshow(normalize_img_np(img_np))
    save_name = os.path.join(vis_root_dir, "image_orig.png")
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

    figure = plt.figure()
    plt.imshow((watermarked_img_np))
    save_name = os.path.join(vis_root_dir, "image_watermarked.png")
    plt.savefig(save_name)
    plt.close(figure)

    img_diff = ((watermarked_img_np - img_np) + 2) / 4
    figure = plt.figure()
    plt.imshow(img_diff)
    save_name = os.path.join(vis_root_dir, "image_diff.png")
    plt.savefig(save_name)
    plt.close(figure)

    # === Decode Watermark ===
    watermark_decoded = model.decoder(watermarked_image).round().clip(0, 1).to(dtype=torch.long)
    watermark_decoded_np = watermark_decoded.detach().cpu().numpy()
    print("Orig Watermark: ", gt_watermark_np)
    print("Decoded       : ", watermark_decoded_np)

    # === Decode a watermark from "natural" image ===
    natural_watermark = model.decoder(images_tensor).round().clip(0, 1).to(dtype=torch.long)
    natural_watermark_np = natural_watermark.detach().cpu().numpy()
    print("Natural       : ", natural_watermark_np)

    print("Bit-wise accuracy: ")
    print("Decoded: ", np.mean(watermark_decoded_np == gt_watermark_np) * 100)
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