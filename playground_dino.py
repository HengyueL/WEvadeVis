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
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


MEAN, STD = 0.5, 0.5


def normalize_img_np(img_np):
    return img_np * STD + MEAN

def transform_image(image, device):
    image = image.to(device)
    return torch.clamp(image, min=-1, max=1)


def plot_image(image_arr, save_name):
    figure, ax = plt.subplots(ncols=1, nrows=1)
    ax.imshow(image_arr)
    plt.savefig(save_name)
    plt.close(figure)


def main(args):
    device = torch.device("cuda")
    vis_root_dir = os.path.join(
        ".", "visualization_dino"
    )
    os.makedirs(vis_root_dir, exist_ok=True)

    dino_backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    # Load encoder-decoder model
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
    for _ in range(10):
        next(dataiter)
        images_tensor, labels_tensor = next(dataiter)

        # === Visualize Figure ===
        img_np = np.transpose(images_tensor.cpu().numpy()[0, :, :, :], [1, 2, 0])
        save_name = os.path.join(vis_root_dir, "image_orig.png")
        plot_image(normalize_img_np(img_np), save_name)

        # === Load Watermark ===
        gt_watermark_np = np.load('./watermark/watermark_coco.npy')
        groundtruth_watermark = torch.from_numpy(gt_watermark_np).to(device)

        images_tensor = images_tensor.to(device)
        images_tensor = transform_image(images_tensor, device)
        watermarked_image = model.encoder(images_tensor, groundtruth_watermark)
        watermarked_image = transform_image(watermarked_image, device=device)

        # === Vis watermarked image ===
        watermarked_image_np = (watermarked_image.detach().cpu().numpy()[0, :, :, :] + 1) / 2.
        watermarked_img_np = np.transpose(
            watermarked_image_np,
            [1, 2, 0]
        )
        save_name = os.path.join(vis_root_dir, "image_watermarked.png")
        plot_image(watermarked_img_np, save_name)

        # Calculate DINO encoding
        upsampler = torch.nn.Upsample(size=224, mode='bilinear')
        orig_encoding = dino_backbone(upsampler(images_tensor)).view(1, -1)
        watermarked_encoding = dino_backbone(upsampler(watermarked_image)).view(1, -1)

        # Benchmarked by a noisy iamge
        noise = 0.01 * torch.randn_like(images_tensor) + images_tensor
        # == Visualize Noise Image ==
        noise_np = (noise.clone().cpu().numpy()[0, :, :, :] + 1) / 2.
        noisy_img_np = np.transpose(
            noise_np,
            [1, 2, 0]
        )
        save_name = os.path.join(vis_root_dir, "image_noisy.png")
        plot_image(noisy_img_np, save_name)

        noise_encoding = dino_backbone(upsampler(noise)).view(1, -1)
        l2_dist = torch.linalg.norm(
            orig_encoding - watermarked_encoding,
            ord=2,
            dim=1
        )
        l2_noise_dist = torch.linalg.norm(
            noise_encoding - orig_encoding,
            ord=2,
            dim=1
        )
        print("Dino Encoding L2 Dist: {} - Noise {}".format(l2_dist.item(), l2_noise_dist.item()))

        # Check Gaussian PSNR
        psrn_noisy = compare_psnr(noisy_img_np, normalize_img_np(img_np), data_range=1)
        print("Noisy Image PSNR: ", psrn_noisy)

        # Decode watermarks
        watermark_decoded_raw = model.decoder(watermarked_image)
        watermark_decoded = watermark_decoded_raw.round().clip(0, 1).to(dtype=torch.long)
        watermark_decoded = watermark_decoded.detach().cpu().numpy()
        watermark_decoded_raw = watermark_decoded_raw.detach().cpu().numpy()

        watermark_noise_raw = model.decoder(noise)
        watermark_noise = watermark_noise_raw.round().clip(0, 1).to(dtype=torch.long)
        watermark_noise = watermark_noise.detach().cpu().numpy()
        watermark_noise_raw = watermark_noise_raw.detach().cpu().numpy()

        print("Display Watermarks")
        print("GT      : ", gt_watermark_np)
        print("Decoded : ", watermark_decoded)
        print("NoisyImg: ", watermark_noise)

        # print("Raw prediction values: ")
        # print("GT      : ", gt_watermark_np)
        # print("Decoded : ", watermark_decoded_raw)
        # print("NoisyImg: ", watermark_noise_raw)

        print("Bit-wise accuracy: ")
        print("Decoded: ", np.mean(watermark_decoded == gt_watermark_np) * 100)
        print("Noisy  : ", np.mean(watermark_noise == gt_watermark_np) * 100)


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