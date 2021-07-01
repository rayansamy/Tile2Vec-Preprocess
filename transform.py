import argparse, sys
import os
from utils.tools import *


if __name__ == "__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--input_folder', help='Input Folder ', type=str)
    parser.add_argument('--output_folder', help='Folder to store tiles', type=str)
    parser.add_argument('--tiles_original_sizes', help='Tiles original sizes ( before being reshaped ) ', type=int, default=256)
    args=parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder
    size = args.tiles_original_sizes

    transform(input_folder, output_folder, size)