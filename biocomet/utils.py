import os
import requests
import tqdm
import pandas as pd
import shutil
import gzip
import json

def updateGeneSymbolsHuman(geneSymbols):
    """
    This function updates gene symbols in a list based on a reference HGNC mapping file.

    Args:
      geneSymbols: A list of gene symbols to be updated.

    Returns:
      A list of updated gene symbols.
    """

    # Path to the tsv file (assuming it's in the same directory)
    dict_path = os.path.join(os.path.dirname(__file__), 'data', 'HGNC_03252024_gene_symbol_mapping.json')

    

    # Read mapping data as DataFrame
    try:
        gene_hgnc_df = pd.read_csv(dict_path, sep='\t', index_col=0)
    except FileNotFoundError:
        print(f"Error: HGNC mapping file not found at {dict_path}")
        return geneSymbols  # Return original list if file not found

    # Update gene symbols in the list
    updated_symbols = []
    for gene in geneSymbols:
        # Check if gene exists in the approved symbol index
        if gene in gene_hgnc_df.index:
            approved_symbol = gene_hgnc_df.loc[gene].name
        else:
            # Updated logic for checking previous and alias symbols
            prev_symbols = gene_hgnc_df['Previous symbols'].apply(lambda x: x.split(', ') if pd.notna(x) else [])
            alias_symbols = gene_hgnc_df['Alias symbols'].apply(lambda x: x.split(', ') if pd.notna(x) else [])

            is_prev_symbol = prev_symbols.apply(lambda x: gene in x)
            is_alias_symbol = alias_symbols.apply(lambda x: gene in x)

            if is_prev_symbol.any() or is_alias_symbol.any():
                approved_symbol = gene_hgnc_df[is_prev_symbol | is_alias_symbol].index[0]  # Ensure single symbol is selected

            else:
                approved_symbol = gene

        updated_symbols.append(approved_symbol)

    return updated_symbols


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