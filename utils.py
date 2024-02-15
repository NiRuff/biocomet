import os
import requests
import tqdm
import pandas as pd
import shutil
import gzip


def download_and_load_dataframe(url, local_filename):

    if not os.path.exists(local_filename):
        # Download the file with progress bar
        response = requests.get(url, stream=True)
        total_size_in_bytes= int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        # Check if download was successful
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")

    # Extract the file
    with gzip.open(local_filename, 'rb') as f_in:
        with open(local_filename[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Read the file into a pandas DataFrame
    dataframe = pd.read_csv(local_filename[:-3], sep='\t')
    return dataframe