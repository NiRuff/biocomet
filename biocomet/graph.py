import networkx as nx
import pandas as pd
import numpy as np
import requests
import igraph as ig
from .utils import download_and_load_dataframe
from .community_detection import apply_leiden, apply_louvain
from .functional_annotation import checkFuncSignificance, checkFuncSignificanceFullNetwork
from .visualization import plot_nv, plotPPI, plotWordclouds, plotWordCloudsPPI, visualize_KEGG, plotRegNetworks
from .centrality import calcCentrality
import os
import urllib.request
from tqdm import tqdm


class PPIGraph:
    def __init__(self, gene_list, reg_list=None, organism='9606', min_score=400, no_text=False, physical=False, auto_load=False, local_data=False, p_adj_cutoff = 0.05, min_comm_size=3):
        self.gene_list = pd.Series(gene_list)
        self.reg_list = reg_list
        self.organism = organism
        self.min_score = min_score
        self.no_text = no_text
        self.p_adj_cutoff = p_adj_cutoff
        self.physical = physical
        self.local_data = local_data
        self.network = None         # Placeholder for the network
        self.partition = None       # Placeholder for the partition
        self.func_annotation = None # Placeholder for the functional annotation
        self.plot_dir = '.'
        self.gene_reg_dict = None
        self.min_comm_size = min_comm_size
        self.func_annotation_full_network = None
        self.auto_load = auto_load
        self.partition_centrality = None
        self.full_network_centrality = None
        self.build_network()        # Build the network upon initialization


    def set_p_adj_cutoff(self, p_adj_cutoff):
        self.p_adj_cutoff = p_adj_cutoff

    def set_min_comm_size(self, min_comm_size):
        self.set_min_comm_size = min_comm_size

    def load_protein_info(self):

        # Construct the URL for the desired organism's protein info
        organism_code = str(self.organism)  # Convert organism code to string
        file_name = f"{organism_code}.protein.info.v12.0.txt.gz"
        data_url = f"https://stringdb-downloads.org/download/protein.info.v12.0/{file_name}"

        # Define the expected path where the file will be saved
        expected_path = f"./data/{file_name}"

        # Check if the file already exists at the expected path
        if not os.path.exists(expected_path):
            # If not, create the directory (if it doesn't exist) and download the file
            os.makedirs("./data", exist_ok=True)
            print(f"Downloading {file_name} from {data_url}...")

            # Download with progress bar
            with tqdm(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:
                urllib.request.urlretrieve(data_url, filename=expected_path, reporthook=self.download_progress(t))

            print("Download complete.")
        else:
            print(f"{file_name} already exists at {expected_path}.")

        # Load the downloaded file into a pandas DataFrame
        df = pd.read_csv(expected_path, compression='gzip', header=0, sep='\t')
        return df

    def download_progress(self, t):
        def update_to(b=1, bsize=1, tsize=None):
            if tsize is not None:
                t.total = tsize
            t.update(b * bsize - t.n)

        return update_to

    def build_network(self):

        # Convert organism to STRING DB identifier
        if str(self.organism).lower() in (['homo sapiens', 'hs', 'human', '9606']):
            self.organism = 9606
        elif str(self.organism).lower() in (['mus musculus', 'mm', 'mouse', '10090']):
            self.organism = 10090
        else:
            print(
                "Organisms should be 'human' or 'mouse' or the string identifier of your organisms.\nIf the code fails, make sure to use the correct identifier or the suggested strings")

        string_api_url = "https://string-db.org/api"
        output_format = "tsv"
        method = "network"
        organism_code = str(self.organism)  # Convert organism code to string
        file_name = f"{organism_code}.protein.links.detailed.v12.0.txt.gz"
        data_url = f"https://stringdb-downloads.org/download/protein.links.detailed.v12.0/{file_name}"

        if (len(self.gene_list) >= 1000) and (not self.local_data):
            if self.auto_load:
                expected_path = f"./data/{file_name}"
                # Check if data exists at the expected path
                if os.path.exists(expected_path):
                    print(f"Loading data from {expected_path}")
                    self.local_data = expected_path
                else:
                    # Create directory if it doesn't exist
                    os.makedirs("./data", exist_ok=True)
                    # Download the data with a progress bar
                    print(f"Downloading data from {data_url} to {expected_path}...")

                    with tqdm(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:
                        urllib.request.urlretrieve(data_url, filename=expected_path,
                                                   reporthook=self.download_progress(t))

                    print("Download complete.")
                    self.local_data = expected_path
                    print(f"Data is now available at {self.local_data}")
            else:
                raise ValueError(
                    "Your identifier list is too big. For lists with a minimum of 1000 identifiers, the stringDB protein network data "
                    "(full network, including subscores per channel) needs to be downloaded. Provide its path via 'local_data' or set "
                    "auto_load=True when instantiating the network to load it automatically.")

        else:
            self.gene_list = getPreferredNames(self.gene_list, organism=self.organism)

        if not self.local_data: # is automatically True if list is too long for string API
            # Construct the request URL
            request_url = f"{string_api_url}/{output_format}/{method}"

            ## Set parameters
            params = {
                "identifiers": "%0d".join(self.gene_list),  # your protein
                "species": self.organism,  # species NCBI identifier
                "caller_identity": "biocomet",  # your app name
                "required_score": self.min_score,  # required score
                "network_type": 'physical' if self.physical else 'functional'  # network type
            }

            # create df for network creation
            interactions = []

            ## Call STRING
            response = requests.post(request_url, params=params)

            if response.status_code != 200:
                raise ConnectionError(f"Warning: The request was unsuccessful. Status code: {response.status_code}")

            if 'error' in response.text.strip().split("\n")[0]:
                raise Warning('Gene list input might not have been mapped correctly. String-db response:')
                print(response.text.strip().split("\n"))

            for line in response.text.strip().split("\n"):
                l = line.strip().split("\t")
                interactions.append(l)

            # manage string output
            interactions = pd.DataFrame(interactions)
            interactions.columns = interactions.iloc[0]
            interactions.drop(0, inplace=True)

        else:
            interactions = pd.read_csv(self.local_data, sep=" ")

            pref_name_df = self.load_protein_info()

            # Create a dictionary from the pref_name_df
            protein_name_dict = pd.Series(pref_name_df.preferred_name.values,
                                          index=pref_name_df['#string_protein_id']).to_dict()

            # map names
            self.gene_list = self.gene_list.replace(protein_name_dict)

            # Map the dictionary to protein1 and protein2 columns to create new columns
            interactions['preferredName_A'] = interactions['protein1'].map(protein_name_dict)
            interactions['preferredName_B'] = interactions['protein2'].map(protein_name_dict)

            # Filter interactions to include only those where both proteins are in the input gene list
            interactions = interactions[
                interactions['preferredName_A'].isin(self.gene_list) & interactions['preferredName_B'].isin(self.gene_list)]

            # apply min_score
            interactions = interactions[interactions['combined_score'] >= self.min_score]

            # Rename columns
            interactions.rename(columns={
                'protein1': 'stringId_A',
                'protein2': 'stringId_B',
                'combined_score': 'score',
                'neighborhood': 'nscore',
                'fusion': 'fscore',
                'cooccurence': 'pscore',
                'coexpression': 'ascore',
                'experimental': 'escore',
                'database': 'dscore',
                'textmining': 'tscore'
            }, inplace=True)

            # Create ncbiTaxonId column and assign value of organism
            interactions['ncbiTaxonId'] = self.organism

            # Reorder columns
            column_order = ['stringId_A', 'stringId_B', 'preferredName_A', 'preferredName_B',
                            'ncbiTaxonId', 'score', 'nscore', 'fscore', 'pscore',
                            'ascore', 'escore', 'dscore', 'tscore']
            interactions = interactions[column_order]

            # scaling necessary as local file has values (0,1000) instead of (0,1) as the API file does
            columns_to_scale = ["nscore", "fscore", "pscore", "ascore", "escore", "dscore"]

            # Assuming 'interactions' is your DataFrame
            for col in columns_to_scale:
                interactions[col] = pd.to_numeric(interactions[col]) / 1000

        if self.no_text:
            print("#######")
            print("You chose no_text-mode. The community creation will not take stringDB textmining into account.")
            print("Arc plots and Circos plots will be created with the scores after textiming exclusion.")
            print("However, be aware that the PPI network images can not be created automatically while excluding the textmining scores,"
                  " therefore it is best here to create the stringDB network images again using stringDB directly.")
            print("#######")

            # create combined score without textmining
            interactions["score"] = 1 - (
                    (1.0 - pd.to_numeric(interactions["nscore"])) *
                    (1.0 - pd.to_numeric(interactions["fscore"])) *
                    (1.0 - pd.to_numeric(interactions["pscore"])) *
                    (1.0 - pd.to_numeric(interactions["ascore"])) *
                    (1.0 - pd.to_numeric(interactions["escore"])) *
                    (1.0 - pd.to_numeric(interactions["dscore"])))

            interactions["score"] = interactions["score"] / interactions["score"].max()

            if self.min_score > 1:
                min_score = self.min_score/1000
            else:
                min_score = self.min_score

            # now remove zero scores
            if self.no_text == "strict":
                interactions = interactions[interactions["score"] >= min_score]
            else:
                interactions = interactions[interactions["score"] > 0]

        # Create a new graph
        self.network = nx.Graph(name='Protein Interaction Graph')

        if self.reg_list is not None:
            # Verify gene list and regulation list compatibility
            if len(self.gene_list) != len(self.reg_list):
                raise ValueError("Gene list and regulation list must match in length and contain no duplicates.")

            elif len(set(self.gene_list)) != len(self.gene_list):
                print('Duplicates found in gene list.')
                gene_reg_dict = dict()
                for gene, reg in zip(self.gene_list, self.reg_list):
                    if gene in gene_reg_dict.keys():
                        if reg != gene_reg_dict[gene]: # otherwise nothing needs to be done
                            raise ValueError(gene + " found at least twice, however, with varying regulation values associated: " + " ".join([reg, gene_reg_dict[gene]]))
                    else:
                        gene_reg_dict[gene] = reg

                # Deduplicate
                self.gene_list = gene_reg_dict.keys()
                self.reg_list = gene_reg_dict.values()
                print('Gene and Regulation list successfully deduplicated.')

            # Create dictionary mapping gene_list to reg_list
            self.gene_reg_dict = dict(zip(self.gene_list, self.reg_list))

            # Update interactions DataFrame with regulation data
            interactions['regulation_a'] = interactions['preferredName_A'].map(self.gene_reg_dict)
            interactions['regulation_b'] = interactions['preferredName_B'].map(self.gene_reg_dict)

            for _, interaction in interactions.iterrows():
                a = interaction["preferredName_A"]
                b = interaction["preferredName_B"]
                weight = float(interaction["score"])
                reg_a = interaction["regulation_a"]
                reg_b = interaction["regulation_b"]

                # Add nodes with regulation property if they don't exist already
                if a not in self.network:
                    self.network.add_node(a, regulation=reg_a)
                if b not in self.network:
                    self.network.add_node(b, regulation=reg_b)

                # Add weighted edge between nodes
                self.network.add_edge(a, b, weight=weight)
        else:
            for _, interaction in interactions.iterrows():
                a = interaction["preferredName_A"]
                b = interaction["preferredName_B"]
                weight = float(interaction["score"])
                self.network.add_edge(a, b, weight=weight)

    def calc_centrality(self, full_network=False):
        if full_network:
            self.full_network_centrality = calcCentrality(G = self.network, partition=None, full_network=True)
        else:
            if self.partition == None:
                print('Community detection necessary first. Starting community detection now with default parameters.')
                self.community_detection()
            self.partition_centrality = calcCentrality(G=self.network, partition=self.partition, full_network=False)

    def write_centrality_to_csv(self, full_network=False, file_name = None,):

        if full_network:
            if self.full_network_centrality == None:
                print('Node Centrality Calculation necessary first. Starting centrality calculations.')
                self.calc_centrality(full_network=True)
            partition_centrality = self.full_network_centrality

            if file_name == None:
                file_name = 'full_network_centrality.csv'
        else:
            if self.partition_centrality == None:
                print('Node Centrality Calculation necessary first. Starting centrality calculations.')
                self.calc_centrality(full_network=False)
            partition_centrality = self.partition_centrality

            if file_name == None:
                file_name = 'community_network_centrality.csv'

        # Flatten the partition_centrality dictionary to a format suitable for DataFrame
        data = []
        for comm, genes in partition_centrality.items():
            for gene, centrality_measures in genes.items():
                row = {'gene': gene, 'community': comm, **centrality_measures}
                data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('gene', inplace=True)
        print('Saving centrality data to ' + file_name)
        # Write to CSV
        df.to_csv(file_name)


    def community_detection(self, iterations=10, algorithm='leiden', seed=None):
        if algorithm == 'leiden':
            self.partition = apply_leiden(to_igraph(self.network), iterations=iterations, seed=seed)
        elif algorithm == 'louvain':
            self.partition = apply_louvain(self.network, iterations=iterations, seed=seed)

    def get_functional_annotation(self, full_network=False, categories = 'default'):



        if full_network:
            if len(self.gene_list) > 1000:
                raise ValueError("Gene list too long. Either the community is too large with more than 1000 genes"
                                 " or you are conducting a full_network analysis on a large network. Functional Enrichment "
                                 " can not be calculated currently for such large lists of genes. For the latter case, "
                                 " consider setting full_network=False to check the individual communities for their "
                                 " functional enrichment.")

            if (categories.lower() in ['default', 'pathways', 'all', 'no_pmid', 'no_go']) or (type(categories) == list):
                self.func_annotation_full_network = checkFuncSignificanceFullNetwork(self, categories=categories)
            else:
                raise AttributeError("categories argument not set properly. Specify list of databases or "
                                     "one of the following strings 'all'/'default'/'pathways'")
        else:
            if self.partition is None:
                print('Community detection necessary first. Starting community detection now with default parameters.')
                self.community_detection()
            if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
                self.func_annotation = checkFuncSignificance(self, sig_only=True,
                                                             categories=categories, min_comm_size=self.min_comm_size)
            else:
                raise AttributeError("categories argument not set properly. Specify list of databases or "
                                     "one of the following strings 'all'/'default'/'pathways'")

    def set_plot_dir(self, plot_dir):
        self.plot_dir = plot_dir

    def plot_arc(self, show=True, background='transparent', legend=True):

        if self.partition == None:
            print('Community detection necessary first. Starting community detection now with default parameters.')
            self.community_detection()

        if self.func_annotation == None:
            print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
            self.get_functional_annotation()

        plot_nv(self.network, self.partition, self.min_comm_size, self.plot_dir, legend=legend, kind='ArcPlots', show=show, background=background)

    def plot_circos(self, show=True, background='transparent', legend=True):

        if self.partition == None:
            print('Community detection necessary first. Starting community detection now with default parameters.')
            self.community_detection()

        if self.func_annotation == None:
            print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
            self.get_functional_annotation()

        plot_nv(self.network, self.partition, self.min_comm_size, self.plot_dir, legend=legend, kind='CircosPlots', show=show, background=background)

    def plot_PPI(self, full_network=False, show=True, background='transparent'):
        if full_network:
            plotPPI(self, full_network=True, show=show, background=background)
        else:
            if self.partition == None:
                print('Community detection necessary first. Starting community detection now with default parameters.')
                self.community_detection()
            plotPPI(self, full_network=False, show=show, background=background)

    def plot_Wordclouds(self, full_network=False, categories='default', show=True, background='transparent', weightedSetCover=False):
        if full_network:
            if self.func_annotation_full_network == None:
                print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
                self.get_functional_annotation(full_network=True, categories=categories)

            if (categories.lower() in ['default', 'pathways', 'all', 'no_pmid', 'no_go']) or (type(categories) == list):
                plotWordclouds(self.func_annotation_full_network,
                               categories=categories,
                               plot_dir=self.plot_dir, show=show, background=background, weightedSetCover=weightedSetCover)
            else:
                raise AttributeError("categories argument not set properly. Specify list of databases or "
                                     "one of the following strings 'all'/'default'/'pathways'")

        else:        # not full network
            if self.partition == None:
                print('Community detection necessary first. Starting community detection now with default parameters.')
                self.community_detection()

            if self.func_annotation == None:
                print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
                self.get_functional_annotation(categories=categories)

            if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
                plotWordclouds(self.func_annotation,
                               categories=categories,
                               plot_dir=self.plot_dir, show=show, background=background, weightedSetCover=weightedSetCover)
            else:
                raise AttributeError("categories argument not set properly. Specify list of databases or "
                                     "one of the following strings 'all'/'default'/'pathways'")

    def plot_Wordclouds_PPI(self, full_network=False, categories='default', show=True, background='transparent', weightedSetCover=False):

        if full_network:
            if self.func_annotation_full_network == None:
                print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
                self.get_functional_annotation(full_network=True, categories=categories)

            if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
                plotWordCloudsPPI(self, full_network=full_network, categories=categories, show=show, background=background,weightedSetCover=weightedSetCover)
            else:
                raise AttributeError("categories argument not set properly. Specify list of databases or "
                                     "one of the following strings 'all'/'default'/'pathways'")

        else:
            if self.partition == None:
                print('Community detection necessary first. Starting community detection now with default parameters.')
                self.community_detection()

            if self.func_annotation == None:
                print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
                self.get_functional_annotation()

            if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
                plotWordCloudsPPI(self, categories=categories, show=show, background=background,weightedSetCover=weightedSetCover)
            else:
                raise AttributeError("categories argument not set properly. Specify list of databases or "
                                     "one of the following strings 'all'/'default'/'pathways'")

    def plot_KEGG(self, pathway='all', community='all', show=True, transparency=.5, background='transparent'):

        if self.partition == None:
            print('Community detection necessary first. Starting community detection now with default parameters.')
            self.community_detection()
        if self.func_annotation == None:
            print('Functional annotation necessary first. Starting functional annotation now with default parameters.')
            self.get_functional_annotation()

        if self.reg_list is not None:
            # Verify gene list and regulation list compatibility
            if len(self.gene_list) != len(self.reg_list) or len(set(self.gene_list)) != len(self.gene_list):
                raise ValueError("Gene list and regulation list must match in length and contain no duplicates.")

            if self.gene_reg_dict is None:
                # Create dictionary mapping gene_list to reg_list
                self.gene_reg_dict = dict(zip(self.gene_list, self.reg_list))

        # first check if specific pathway chosen
        if pathway != 'all':
            if community != 'all':  # specific community and pathway
                df = self.func_annotation[community]  # just specific community's df
                df_kegg = df[(df['category'] == 'KEGG') & (df['term'] == pathway)]
                if not df_kegg.empty():
                    pathway_genes_dict = zip(df_kegg['term'], df_kegg['inputGenes'])
                    for pathway_id, genes in pathway_genes_dict:
                        gene_reg_dict = {k:v for k,v in self.gene_reg_dict.items() if k in genes}
                        visualize_KEGG(pathway_id=pathway_id, gene_reg_dict=gene_reg_dict, organism=self.organism,
                                       plot_dir=self.plot_dir, transparency=transparency, community=community, show=show, background=background)
                else:
                    print(pathway + ' not found in sig. results of community ' + community)

            else:  # specific pathway in all communities
                for comm, df in self.func_annotation.items():
                    df_kegg = df[(df['category'] == 'KEGG') & (df['term'] == pathway)]
                    if not df_kegg.empty():
                        pathway_genes_dict = zip(df_kegg['term'], df_kegg['inputGenes'])
                        for pathway_id, genes in pathway_genes_dict:
                            gene_reg_dict = {k:v for k,v in self.gene_reg_dict.items() if k in genes}
                            visualize_KEGG(pathway_id=pathway_id, gene_reg_dict=gene_reg_dict, organism=self.organism,
                                           plot_dir=self.plot_dir, transparency=transparency, community=comm, show=show, background=background)
        else:
            if community != 'all':  # implement all pathways of given community
                df = self.func_annotation[community]  # just specific community's df
                df_kegg = df[df['category'] == 'KEGG']
                pathway_genes_dict = zip(df_kegg['term'], df_kegg['inputGenes'])
                for pathway_id, genes in pathway_genes_dict:
                    gene_reg_dict = {k: v for k, v in self.gene_reg_dict.items() if k in genes}
                    visualize_KEGG(pathway_id=pathway_id, gene_reg_dict=gene_reg_dict, organism=self.organism,
                                   plot_dir=self.plot_dir, transparency=transparency, community=community, show=show, background=background)
            else: # all pathways all communities
                for comm, df in self.func_annotation.items():
                    df_kegg = df[df['category'] == 'KEGG']
                    pathway_genes_dict = zip(df_kegg['term'], df_kegg['inputGenes'])
                    for pathway_id, genes in pathway_genes_dict:
                        gene_reg_dict = {k:v for k,v in self.gene_reg_dict.items() if k in genes}
                        visualize_KEGG(pathway_id=pathway_id, gene_reg_dict=gene_reg_dict, organism=self.organism,
                                       plot_dir=self.plot_dir, transparency=transparency, community=comm, show=show, background=background)

    def plot_reg_networks(self, full_network=False, community='all', centrality_measure='w_degree', show=True, background='transparent'):
        if full_network:
            if centrality_measure and self.full_network_centrality == None:
                    print('Node Centrality Calculation necessary first. Starting centrality calculations.')
                    self.calc_centrality(full_network=True)
            plotRegNetworks(self.network, self.partition, self.full_network_centrality, self.plot_dir, full_network=True, centrality_measure=centrality_measure, community=community,
                            show=show, background=background, min_comm_size=self.min_comm_size)
        else:
            if self.partition is None:
                print('Community detection necessary first. Starting community detection now with default parameters.')
                self.community_detection()
            if centrality_measure and self.partition_centrality == None:
                    print('Node Centrality Calculation necessary first. Starting centrality calculations.')
                    self.calc_centrality(full_network=False)
            plotRegNetworks(self.network, self.partition, self.partition_centrality, self.plot_dir, full_network=False, centrality_measure=centrality_measure, community=community,
                            show=show, background=background, min_comm_size=self.min_comm_size)


def to_igraph(network):

    g = ig.Graph(directed=network.is_directed())
    g.add_vertices(list(network.nodes()))

    # Prepare edges and weights for igraph
    edges = [(g.vs.find(name=u).index, g.vs.find(name=v).index) for u, v in network.edges()]
    weights = [attr['weight'] for u, v, attr in network.edges(data=True)]

    # Add edges and weights to igraph
    g.add_edges(edges)
    g.es['weight'] = weights
    return g

def query_stringdb(gene_ids, organism):
    base_url = "https://string-db.org/api"
    output_format = "json"
    method = "get_string_ids"
    params = {
        "identifiers": "\r".join(gene_ids),  # Join the list of gene IDs by new lines
        "species": organism,  # Human by default, change as needed
        "limit": 1,  # Limit to 1 result per query for simplicity
        "echo_query": 1,  # Echo back the input query
    }

    response = requests.post(f"{base_url}/{output_format}/{method}", data=params)
    if response.status_code != 200:
        raise ValueError("Error querying STRING database")

    return pd.DataFrame(response.json())

def getPreferredNames(gene_ids, organism=9606):
    return query_stringdb(gene_ids, organism=organism)["preferredName"].values