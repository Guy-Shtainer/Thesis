import yt
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import unyt as u
import argparse
import numpy as np
import pickle


def main(directory_path):
    yt.enable_parallelism()
    # print(directory_path)
    ds = yt.load(directory_path)
    hc = HaloCatalog(data_ds=ds, finder_method="hop",output_dir = './')
    hc.create()

    with open('hc.pkl', 'wb') as file:
        pickle.dump(hc, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example script with command-line arguments")
    
    # Define positional arguments
    parser.add_argument("directory_path", type=str, help="Path to snapshot")
    
    # Define optional arguments (flags)
    # parser.add_argument("--flag", action="store_true", help="A boolean flag")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    print(args)

    # Call the main function with the parsed arguments
    main(args.directory_path)
