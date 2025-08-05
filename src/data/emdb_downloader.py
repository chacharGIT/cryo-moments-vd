from math import e
import numpy as np
import os
import requests
import gzip
from aspire.volume import Volume
import mrcfile
import urllib.parse
from config.config import settings
import tempfile

def search_emdb_asymmetric_ids(resolution_cutoff, max_results, page=1):
    """
    Search EMDB for single particle EM maps with resolution cutoff,
    filter for asymmetric entries (space group == 1),
    and return only their EMDB IDs.
    """
    query_str = (
        f'resolution:[* TO {resolution_cutoff}] AND '
        f'structure_determination_method:"singleParticle"'
    )
    encoded_query = urllib.parse.quote(query_str)

    fields = "emdb_id,interpretation,map"

    url = (
        f"https://www.ebi.ac.uk/emdb/api/search/{encoded_query}"
        f"?rows={max_results}&page={page}&fl={fields}"
    )

    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    emdb_ids = []
    for entry in data:
        try:
            add_maps = entry.get("interpretation", {}).get("additional_map_list", {}).get("additional_map", [])
            if add_maps:
                space_group = int(add_maps[0].get("symmetry", {}).get("space_group", 0))
            else:
                space_group = int(entry.get("map", {}).get("symmetry", {}).get("space_group", 0))
        except Exception:
            space_group = None
        
        if space_group == 1 and "emdb_id" in entry:
            emdb_ids.append(entry["emdb_id"])

    return emdb_ids

def download_emdb_map(emdb_id, download_dir):
    """
    Download the primary map file (.map.gz) for a given EMDB ID using the EMDB API.
    Returns the local file path to the downloaded .map.gz file.
    """
    emdb_id_str = str(emdb_id).upper().replace("EMD-", "").zfill(4)
    api_url = f"https://www.ebi.ac.uk/emdb/api/entry/map/{emdb_id_str}"
    
    # Get map file metadata
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()

    map_filename = data['map']['file']
    download_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id_str}/map/{map_filename}"
    
    # Ensure download_dir exists
    os.makedirs(download_dir, exist_ok=True)
    local_path = os.path.join(download_dir, map_filename)

    # Check if file already exists
    if os.path.exists(local_path):
        print(f"File {local_path} already exists. Skipping download.")
        return local_path

    # Download the actual map file
    print(f"Downloading {emdb_id} map to {local_path}...")
    map_resp = requests.get(download_url, stream=True)
    map_resp.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in map_resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path

def load_aspire_volume(filepath, downsample_size=None):
    """
    Load a .map or .map.gz volume file into an ASPIRE Volume object. Optionally downsample.
    
    Parameters:
        filepath (str): Path to the .map or .map.gz file.
        downsample_size (int, optional): If set, downsample the volume to this size.

    Returns:
        Volume: ASPIRE Volume object containing the 3D data.
    """
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rb') as f_in, tempfile.NamedTemporaryFile(suffix='.map', delete=False) as f_out:
            f_out.write(f_in.read())
            temp_path = f_out.name
            cleanup = True
    else:
        temp_path = filepath
        cleanup = False

    with mrcfile.open(temp_path, permissive=True) as mrc:
        data = mrc.data.astype(np.float32)
        volume = Volume(data)

    if downsample_size is not None:
        volume = volume.downsample(downsample_size)

    if cleanup:
        os.remove(temp_path)

    return volume

if __name__ == "__main__":
    emdb_ids = search_emdb_asymmetric_ids(settings.data_generation.emdb.resolution_cutoff,
                                           settings.data_generation.emdb.max_results)

    for idx, emdb_id in enumerate(emdb_ids):
        print(f"[{idx+1}/{len(emdb_ids)}] Downloading map for {emdb_id}...")
        try:
            map_path = download_emdb_map(emdb_id, settings.data_generation.emdb.download_folder)
            print(f"Downloaded to {map_path}. Loading as aspire Volume...")
            vol = load_aspire_volume(map_path, downsample_size=settings.data_generation.downsample_size)
            print(f"Loaded Volume: shape={vol.shape}, dtype={vol.dtype}")
        except Exception as e:
            print(f"Failed to process {emdb_id}: {e}")