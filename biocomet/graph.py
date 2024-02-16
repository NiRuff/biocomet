import networkx as nx
import pandas as pd
import numpy as np
import requests
import igraph as ig
from .utils import download_and_load_dataframe
from .community_detection import apply_leiden, apply_louvain
from .functional_annotation import checkFuncSignificance
from .visualization import plot_nv, plotPPI, plotWordclouds, plotWordCloudsPPI, visualize_KEGG, plotRegNetworks



class PPIGraph:
    def __init__(self, gene_list, reg_list=None, organism='9606', min_score=400, no_text=False, physical=False, local_data=False):
        self.gene_list = getPreferredNames(gene_list, organism=organism)
        self.reg_list = reg_list
        self.organism = organism
        self.min_score = min_score
        self.no_text = no_text
        self.physical = physical
        self.local_data = local_data
        self.network = None         # Placeholder for the network
        self.build_network()        # Build the network upon initialization
        self.partition = None       # Placeholder for the partition
        self.func_annotation = None # Placeholder for the functional annotation
        self.plot_dir = '.'
        self.gene_reg_dict = None



    def build_network(self):
        string_api_url = "https://string-db.org/api"
        output_format = "tsv"
        method = "network"

        if (len(self.gene_list) >= 1000) and (not self.local_data):
            raise ValueError(
                "Your identifier list is too big. For lists with min. 1000 identifiers, please download the protein network data (full network, incl. subscores per channel) from stringDB and provide the path via 'local_data'")

        if self.physical:
            network_type = "physical"
        else:
            network_type = "functional"
        # Convert organism to STRING DB identifier
        if str(self.organism).lower() in(['homo sapiens', 'hs', 'human', '9606']):
            self.organism = 9606
        elif str(self.organism).lower() in(['mus musculus', 'mm', 'mouse', '10090']):
            self.organism = 10090
        else:
            print(
                "Organisms should be 'human' or 'mouse' or the string identifier of your organisms.\nIf the code fails, make sure to use the correct identifier or the suggested strings")

        if not self.local_data:
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
            if self.organism == 9606:
                url = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
                local_filename = "9606.protein.info.v12.0.txt.gz"
            elif self.organism == 10090:
                url = "https://stringdb-downloads.org/download/protein.info.v12.0/10090.protein.info.v12.0.txt.gz"
                local_filename = "10090.protein.info.v12.0.txt.gz"
            pref_name_df = download_and_load_dataframe(url, local_filename)

            # Create a dictionary from the pref_name_df
            protein_name_dict = pd.Series(pref_name_df.preferred_name.values,
                                          index=pref_name_df['#string_protein_id']).to_dict()

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

            # scaling necessary as local file has values (0,1000) insterad of (0,1) as the API file does
            columns_to_scale = ["nscore", "fscore", "pscore", "ascore", "escore", "dscore"]

            # Assuming 'interactions' is your DataFrame
            for col in columns_to_scale:
                interactions[col] = pd.to_numeric(interactions[col]) / 1000

        if self.no_text:
            print("#######")
            print("You chose no_text-mode. The community creation will not take stringDB textmining into account.")
            print("Arc plots and Circos plots will be created with the scores after textiming exclusion.")
            print(
                "However, be aware that the PPI network images can not be created automatically while excluding the textmining scores, therefor it is best here to create the stringDB network images again using stringDB directly.")
            print("#######")

            # create combined score without textmining
            interactions["score"] = 1 - (
                    (1.0 - pd.to_numeric(interactions["nscore"])) *
                    (1.0 - pd.to_numeric(interactions["fscore"])) *
                    (1.0 - pd.to_numeric(interactions["pscore"])) *
                    (1.0 - pd.to_numeric(interactions["ascore"])) *
                    (1.0 - pd.to_numeric(interactions["escore"])) *
                    (1.0 - pd.to_numeric(interactions["dscore"])))

            interactions["score"] = interactions["score"] / interactions["score"].max() * 1000

            # now remove zero scores
            if self.no_text == "strict":
                interactions = interactions[interactions["score"] >= self.min_score]
            else:
                interactions = interactions[interactions["score"] > 0]

        # Create a new graph
        self.network = nx.Graph(name='Protein Interaction Graph')

        if self.reg_list is not None:
            # Verify gene list and regulation list compatibility
            if len(self.gene_list) != len(self.reg_list) or len(set(self.gene_list)) != len(self.gene_list):
                raise ValueError("Gene list and regulation list must match in length and contain no duplicates.")

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


    def community_detection(self, iterations=100, algorithm='leiden'):
        if algorithm == 'leiden':
            self.partition = apply_leiden(to_igraph(self.network), iterations=iterations)
        elif algorithm == 'louvain':
            self.partition = apply_louvain(self.network, iterations=iterations)

    def get_functional_annotation(self, categories = 'default'):
        if self.partition is None:
            print('Community detection necessary first. Starting community detection now with default parameters.')
            self.community_detection()
        if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
            self.func_annotation = checkFuncSignificance(self, sig_only=True,
                                                         categories='default', minCommSize=2)
        else:
            raise AttributeError("categories argument not set properly. Specify list of databases or "
                                 "one of the following strings 'all'/'default'/'pathways'")

    def set_plot_dir(self, plot_dir):
        self.plot_dir = plot_dir

    def plot_arc(self):

        if self.partition is None:
            raise ValueError("Partition not conducted yet. Please run PPIGraph.community_detection() first.")

        elif self.func_annotation is None:
            print("No functional annotation done yet. Plot contains all partitions now.")

        plot_nv(self.network, self.partition, self.plot_dir, legend=True, kind='ArcPlots')

    def plot_circos(self):

        if self.partition is None:
            raise ValueError("Partition not conducted yet. Please run PPIGraph.community_detection() first.")

        elif self.func_annotation is None:
            print("No functional annotation done yet. Plot contains all partitions now.")

        plot_nv(self.network, self.partition, self.plot_dir, legend=True, kind='CircosPlots')

    def plot_PPI(self):
        plotPPI(self)

    def plot_Wordclouds(self, categories='default'):
        if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
            plotWordclouds(self.func_annotation,
                           categories=categories,
                           plot_dir=self.plot_dir)
        else:
            raise AttributeError("categories argument not set properly. Specify list of databases or "
                                 "one of the following strings 'all'/'default'/'pathways'")

    def plot_Wordclouds_PPI(self, categories='default'):
        if (categories.lower() in ['default','pathways','all', 'no_pmid', 'no_go']) or (type(categories) == list):
            plotWordCloudsPPI(self, categories=categories)
        else:
            raise AttributeError("categories argument not set properly. Specify list of databases or "
                                 "one of the following strings 'all'/'default'/'pathways'")

    def plotKEGG(self, pathway='all', community='all', show=True, transparency=.5):

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
                                       plot_dir=self.plot_dir, transparency=transparency, community=community, show=show)
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
                                           plot_dir=self.plot_dir, transparency=transparency, community=comm, show=show)
        else:
            if community != 'all':  # implement all pathways of given community
                df = self.func_annotation[community]  # just specific community's df
                df_kegg = df[df['category'] == 'KEGG']
                pathway_genes_dict = zip(df_kegg['term'], df_kegg['inputGenes'])
                for pathway_id, genes in pathway_genes_dict:
                    gene_reg_dict = {k: v for k, v in self.gene_reg_dict.items() if k in genes}
                    visualize_KEGG(pathway_id=pathway_id, gene_reg_dict=gene_reg_dict, organism=self.organism,
                                   plot_dir=self.plot_dir, transparency=transparency, community=community, show=show)
            else: # all pathways all communities
                for comm, df in self.func_annotation.items():
                    df_kegg = df[df['category'] == 'KEGG']
                    pathway_genes_dict = zip(df_kegg['term'], df_kegg['inputGenes'])
                    for pathway_id, genes in pathway_genes_dict:
                        gene_reg_dict = {k:v for k,v in self.gene_reg_dict.items() if k in genes}
                        visualize_KEGG(pathway_id=pathway_id, gene_reg_dict=gene_reg_dict, organism=self.organism,
                                       plot_dir=self.plot_dir, transparency=transparency, community=comm, show=show)

    def plot_reg_networks(self, community='all', show=True):
        if self.partition is None:
            print('Community detection necessary first. Starting community detection now with default parameters.')
            self.community_detection()
        plotRegNetworks(self.network, self.partition, self.plot_dir, community=community, show=show)


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