import yt
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import unyt as u
import argparse
import numpy as np
import pickle


def main(directory_path):
    yt.enable_parallelism()
    print(directory_path)
    ds = yt.load(directory_path)
    ad = ds.all_data()

    # particles types naming
    gas = "PartType0"
    star = "PartType4"
    
    # Define the box size (20 kpc)
    box_size = 20  # kpc
    
    # Find the densest region and its center within the box size
    density_max = ad[gas, 'density'].max()  # Find the maximum density
    max_density_position = ad[gas, 'density'].argmax()  # Find the position of the maximum density
    
    # Get the coordinates of the densest region's center, i.e the center of the galaxy
    densest_region_center = np.array(ad[gas, 'Coordinates'][max_density_position])
    
    # Define the box with the center at the densest region and the specified size
    region = ds.box(
        densest_region_center - 0.5 * box_size,
        densest_region_center + 0.5 * box_size
    )

   

    search_args = dict(use_gas=False, use_particles=True, particle_type=star) # making sure to use only gas particles (I think) and mass-weighted
    L_disk = region.quantities.angular_momentum_vector(**search_args)
    L_disk_norm = L_disk/np.linalg.norm(L_disk)
    print(L_disk)
    print(L_disk_norm)

    with open('result.pkl', 'wb') as file:
        pickle.dump(L_disk_norm, file)

    
    return L_disk_norm

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
