from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os
import glob
import re
import logging
import shutil

ADS_DL_LINK = "https://dl.fbaipublicfiles.com/opencatalystproject/data/per_adsorbate_is2res/"

def uncompress_data(compressed_dir):
    import scripts.uncompress as uncompress
    parser = uncompress.get_parser()
    args, _ = parser.parse_known_args()
    args.ipdir = compressed_dir
    args.opdir = compressed_dir + "_uncompressed"
    uncompress.main(args)
    return args.opdir

def download_ads(root_dir, ads_num):
    download_link = ADS_DL_LINK + str(ads_num) + ".tar"
    os.system(f"wget {download_link} -P {root_dir}")
    filename = os.path.join(root_dir, os.path.basename(download_link))
    logging.info("Extracting contents...")
    os.system(f"tar -xvf {filename} -C {root_dir}")
    os.remove(filename)

def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

def write_ads_to_lmdb(root_dir, ads_num):
    with open(os.path.join(root_dir, str(ads_num), "system.txt")) as sysfile:
        system_data = sysfile.readlines()
    ref_energies = {}
    for line in system_data:
        id, energy = line.split(",")
        id = int(re.findall(r'random([\d]*)', id)[0])
        ref_energies[int(id)] = float(energy)
    
    lmdb_path = os.path.join(root_dir, str(ads_num) + ".lmdb")
    db = lmdb.open(
        lmdb_path,
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )

    a2g = AtomsToGraphs(
        max_neigh=50,
        radius=6,
        r_energy=True,    # False for test data
        r_forces=True,
        r_distances=False,
        r_fixed=True,
    )

    traj_folder = os.path.join(root_dir, str(ads_num), str(ads_num)+"_uncompressed")
    ads_files = glob.glob(os.path.join(traj_folder, "*.extxyz"))
    for file in tqdm.tqdm(ads_files):
        system_id = int(re.findall(r'random([\d]*)', file)[0])
        ref_energy = ref_energies[system_id]

        data_objects = read_trajectory_extract_features(a2g, file)
        initial_struc = data_objects[0]
        relaxed_struc = data_objects[1]
        
        initial_struc.y_init = initial_struc.y - ref_energy # subtract off reference energy, if applicable
        del initial_struc.y
        initial_struc.y_relaxed = relaxed_struc.y - ref_energy # subtract off reference energy, if applicable
        initial_struc.pos_relaxed = relaxed_struc.pos

        initial_struc.sid = system_id
        initial_struc.ads_num=ads_num

        if initial_struc.edge_index.shape[1] == 0:
            print("no neighbors", system_id)
            continue

        txn = db.begin(write=True)
        txn.put(f"{system_id}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
        txn.commit()
        db.sync()
    db.close()


def process_adsorbates(root_dir, N_ADS=82):
    os.makedirs(root_dir, exist_ok=True)
    for i in range(N_ADS):
        ads_dir = os.path.join(root_dir, str(i))
        if not os.path.exists(ads_dir):
            download_ads(root_dir, i)

        compressed_dir = os.path.join(ads_dir, str(i))
        uncompressed_dir = os.path.join(ads_dir, str(i) + "_uncompressed")
        if os.path.exists(compressed_dir) and not os.path.exists(uncompressed_dir):
            _ = uncompress_data(compressed_dir)
            shutil.rmtree(compressed_dir)

        write_ads_to_lmdb(ads_dir, i)
        shutil.rmtree(ads_dir)


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, default="adsorbate-data")

args = parser.parse_args()

process_adsorbates(args.root_dir)

