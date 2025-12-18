#!/usr/bin/env python3
"""
Resize cryo-EM maps to 1 Angstrom voxel spacing.

This is a standalone implementation extracted from DiffModeler to remove
the dependency on the DiffModeler submodule.
"""

import argparse
import os
from pathlib import Path

import mrcfile
import numpy as np
import torch
import torch.nn.functional as F


def resize_map_to_1a(input_mrc, output_mrc, use_gpu=False):
    """
    Resample a cryo-EM map to 1 Angstrom voxel spacing.

    Parameters
    ----------
    input_mrc : str or Path
        Input MRC/MAP file path
    output_mrc : str or Path
        Output MRC/MAP file path
    use_gpu : bool
        Whether to use GPU for resampling (default: False)
    """
    with torch.no_grad():
        # Use modern autocast API
        autocast_context = torch.amp.autocast('cuda', enabled=use_gpu) if use_gpu else torch.amp.autocast('cpu', enabled=False)

        with autocast_context:
            with mrcfile.open(input_mrc, permissive=True) as orig_map:
                orig_voxel_size = np.array([
                    orig_map.voxel_size.x,
                    orig_map.voxel_size.y,
                    orig_map.voxel_size.z
                ])

                # Check if already 1 Angstrom
                if np.allclose(orig_voxel_size, 1.0, atol=0.01):
                    print(f"No resizing needed for {input_mrc}. Creating symlink to {output_mrc}.")
                    if os.path.exists(output_mrc):
                        os.remove(output_mrc)
                    os.symlink(os.path.abspath(input_mrc), output_mrc)
                    return

                # Load and prepare data
                orig_data = torch.from_numpy(
                    orig_map.data.astype(np.float32).copy()
                ).unsqueeze(0).unsqueeze(0)

                if use_gpu:
                    orig_data = orig_data.cuda()

                print(f"[{input_mrc}] Original Shape (ZYX): {orig_data.shape[2:]}")
                print(
                    f"[{input_mrc}] Original Voxel Size (XYZ): "
                    f"{orig_map.voxel_size.x:.4f}, {orig_map.voxel_size.y:.4f}, {orig_map.voxel_size.z:.4f}"
                )

                # Calculate new grid size
                new_grid_size = np.array(orig_data.shape[2:]) * np.array([
                    orig_map.voxel_size.z,
                    orig_map.voxel_size.y,
                    orig_map.voxel_size.x
                ])
                print(f"[{input_mrc}] New Grid Size (ZYX): {new_grid_size}")
                new_grid_size = np.floor(new_grid_size).astype(np.int32)
                print(f"[{input_mrc}] New Grid Size (int, ZYX): {new_grid_size}")

                # Create sampling grid
                # Handle different PyTorch versions
                device = "cuda" if use_gpu else "cpu"
                kwargs = {"indexing": "ij"} if hasattr(torch, '__version__') and (
                    int(torch.__version__.split(".")[0]) >= 2 or
                    int(torch.__version__.split(".")[1]) >= 10
                ) else {}

                z = (
                    torch.arange(0, new_grid_size[0], device=device) /
                    orig_voxel_size[2] / (orig_data.shape[2] - 1) * 2 - 1
                )
                y = (
                    torch.arange(0, new_grid_size[1], device=device) /
                    orig_voxel_size[1] / (orig_data.shape[3] - 1) * 2 - 1
                )
                x = (
                    torch.arange(0, new_grid_size[2], device=device) /
                    orig_voxel_size[0] / (orig_data.shape[4] - 1) * 2 - 1
                )

                new_grid = torch.stack(
                    torch.meshgrid(x, y, z, **kwargs),
                    dim=-1,
                ).unsqueeze(0)

                # Resample using grid_sample
                new_data = F.grid_sample(
                    orig_data, new_grid,
                    mode="bilinear",
                    align_corners=True
                ).cpu().numpy()[0, 0]

                new_voxel_size = np.array((1.0, 1.0, 1.0))

                # Transpose to correct orientation
                new_data = new_data.transpose((2, 1, 0))

                print(f"[{input_mrc}] New Shape (ZYX): {new_data.shape}")
                print(
                    f"[{input_mrc}] New Voxel Size (XYZ): "
                    f"{new_voxel_size[0]:.4f}, {new_voxel_size[1]:.4f}, {new_voxel_size[2]:.4f}"
                )

                # Write output MRC file
                with mrcfile.new(output_mrc, data=new_data.astype(np.float32), overwrite=True) as mrc:
                    vox_sizes = mrc.voxel_size
                    vox_sizes.flags.writeable = True
                    vox_sizes.x = new_voxel_size[0]
                    vox_sizes.y = new_voxel_size[1]
                    vox_sizes.z = new_voxel_size[2]
                    mrc.voxel_size = vox_sizes
                    mrc.update_header_from_data()

                    # Preserve origin information
                    mrc.header.nxstart = 0
                    mrc.header.nystart = 0
                    mrc.header.nzstart = 0
                    mrc.header.origin.x = (
                        orig_map.header.origin.x +
                        orig_map.header.nxstart * orig_voxel_size[0]
                    )
                    mrc.header.origin.y = (
                        orig_map.header.origin.y +
                        orig_map.header.nystart * orig_voxel_size[1]
                    )
                    mrc.header.origin.z = (
                        orig_map.header.origin.z +
                        orig_map.header.nzstart * orig_voxel_size[2]
                    )

                    # Preserve axis order
                    mrc.header.mapc = orig_map.header.mapc
                    mrc.header.mapr = orig_map.header.mapr
                    mrc.header.maps = orig_map.header.maps

                    mrc.update_header_stats()
                    mrc.flush()


def resize_map(input_map_path, output_map_path):
    """
    Resize a cryo-EM map to 1 Angstrom voxel spacing.

    Tries GPU first, falls back to CPU if GPU fails.

    Parameters
    ----------
    input_map_path : str or Path
        Input MRC/MAP file path
    output_map_path : str or Path
        Output MRC/MAP file path

    Returns
    -------
    str
        Path to output map file
    """
    try:
        resize_map_to_1a(input_map_path, output_map_path, use_gpu=True)
    except Exception as e:
        print(f"GPU resampling failed: {e}")
        print("Falling back to CPU...")
        resize_map_to_1a(input_map_path, output_map_path, use_gpu=False)

    return output_map_path


# Alias for backward compatibility with DiffModeler
Resize_Map = resize_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resize cryo-EM map to 1 Angstrom voxel spacing"
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input MRC/MAP file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output MRC/MAP file"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode (default: try GPU first)"
    )

    args = parser.parse_args()

    if args.cpu:
        resize_map_to_1a(args.input, args.output, use_gpu=False)
    else:
        resize_map(args.input, args.output)

    print(f"\nDone! Resized map saved to: {args.output}")
