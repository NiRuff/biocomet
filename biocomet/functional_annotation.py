import requests
import pandas as pd
import json
import networkx as nx
import stringdb


def checkFuncSignificanceFullNetwork(PPIGraph, gene_background=None, categories='default'):
    if type(categories) != str:
        pass
    elif categories.lower() == 'pathways':
        categories = ["KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'default':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'no_pmid':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro",]
    elif categories.lower() == 'no_go':
        categories = ["KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro",]

    string_api_url = "https://version-12.string-db.org/api"
    output_format = "json"
    method = "enrichment"

    funcAnnots = dict()

    # Handle gene_background
    if gene_background is not None:
        if isinstance(gene_background, pd.DataFrame):
            gene_background = gene_background.iloc[:, 0]  # Take the first column
        elif not isinstance(gene_background, pd.Series):
            gene_background = pd.Series(gene_background)

    # Convert gene_background to STRING IDs
    if gene_background is not None:
        background_string_ids = stringdb.get_string_ids(gene_background.dropna().values, species=PPIGraph.organism)
        prefName_string_dict = dict(zip(background_string_ids["preferredName"], background_string_ids["stringId"]))
    else:
        prefName_string_dict = {}

    # Convert PPIGraph.gene_list to STRING IDs
    protein_list_string_ids = stringdb.get_string_ids(PPIGraph.gene_list, species=PPIGraph.organism)
    protein_list = protein_list_string_ids["stringId"].tolist()

    # Check if all genes in PPIGraph.gene_list are in the background
    if gene_background is not None:
        not_in_background = set(protein_list) - set(prefName_string_dict.values())
        if not_in_background:
            print(f"Warning: The following genes from PPIGraph.gene_list were not in the provided background and have been added: {not_in_background}")
            for gene in not_in_background:
                prefName_string_dict[gene] = gene

    # Prepare the request parameters
    params = {
        "identifiers": "%0d".join(protein_list),
        "species": PPIGraph.organism,
        "caller_identity": "biocomet"
    }

    # Add background if provided
    if gene_background is not None:
        params["background_string_identifiers"] = "%0d".join(prefName_string_dict.values())

    # Construct URL
    request_url = "/".join([string_api_url, output_format, method])

    # Call STRING
    response = requests.post(request_url, data=params)

    # Read and parse the results
    data = json.loads(response.text)
    funcAnnots['Full Network'] = pd.DataFrame(data)

    return funcAnnots


def checkFuncSignificance(PPIGraph, gene_background=None, sig_only=True, categories='default', min_comm_size=3):
    if type(categories) != str:
        pass
    elif categories.lower() == 'pathways':
        categories = ["KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'default':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'no_pmid':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro",]
    elif categories.lower() == 'no_go':
        categories = ["KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro",]

    communities = {comm: [] for comm in set(PPIGraph.partition.values())}
    for gene, comm_num in PPIGraph.partition.items():
        communities[comm_num].append(gene)

    string_api_url = "https://version-12.string-db.org/api"
    output_format = "json"
    method = "enrichment"

    sigCommNumbers = set()
    funcAnnots = dict()

    # Handle gene_background
    if gene_background is not None:
        if isinstance(gene_background, pd.DataFrame):
            gene_background = gene_background.iloc[:, 0]  # Take the first column
        elif not isinstance(gene_background, pd.Series):
            gene_background = pd.Series(gene_background)

    # Convert gene_background to STRING IDs
    if gene_background is not None:
        background_string_ids = stringdb.get_string_ids(gene_background.dropna().values, species=PPIGraph.organism)
        prefName_string_dict = dict(zip(background_string_ids["preferredName"], background_string_ids["stringId"]))
    else:
        prefName_string_dict = {}

    for comm, genes in communities.items():
        if len(genes) > 1000:
            raise ValueError(f"Too many proteins in community {comm} for performing functional enrichment. "
                             "Consider using a different community detection approach "
                             "or being more conservative when selecting genes for the analysis")

        # Convert community genes to STRING IDs
        protein_list_string_ids = stringdb.get_string_ids(genes, species=PPIGraph.organism)
        protein_list = protein_list_string_ids["stringId"].tolist()

        # Check if all genes in the community are in the background
        if gene_background is not None:
            not_in_background = set(protein_list) - set(prefName_string_dict.values())
            if not_in_background:
                print(f"Warning: The following genes from community {comm} were not in the provided background and have been added: {not_in_background}")
                for gene in not_in_background:
                    prefName_string_dict[gene] = gene

        # Prepare the request parameters
        params = {
            "identifiers": "%0d".join(protein_list),
            "species": PPIGraph.organism,
            "caller_identity": "biocomet"
        }

        # Add background if provided
        if gene_background is not None:
            params["background_string_identifiers"] = "%0d".join(prefName_string_dict.values())

        # Construct URL
        request_url = "/".join([string_api_url, output_format, method])

        # Call STRING
        response = requests.post(request_url, data=params)

        # Read and parse the results
        data = json.loads(response.text)
        anyProcessSig = False

        for row in data:
            fdr = float(row["fdr"])
            category = row["category"]

            if categories != 'all':  # filter for categories
                if (not sig_only) or (category in categories and fdr < PPIGraph.p_adj_cutoff):
                    anyProcessSig = True
                    sigCommNumbers.add(comm)
            else:  # do not filter
                if (not sig_only) or (fdr < PPIGraph.p_adj_cutoff):
                    anyProcessSig = True
                    sigCommNumbers.add(comm)

        if anyProcessSig:
            funcAnnots[comm] = pd.DataFrame(data)

    # Create dict for sig partition and attributes to add for nodes
    sigPartition = dict()
    attrs = dict()
    node_set = set(PPIGraph.network.nodes)

    smallComms = []
    # Add comm num to nodes. Assign -1 if node is not part of any significant comm
    for comm_num in sigCommNumbers:
        # Only keep communities with length > min_comm_size
        if len(communities[comm_num]) >= min_comm_size:
            for gene in communities[comm_num]:
                attrs[gene] = {"community": comm_num}
                node_set.remove(gene)
                sigPartition[gene] = comm_num
        else:
            for gene in communities[comm_num]:
                sigPartition[gene] = -1
                attrs[gene] = {"community": -1}
            smallComms.append(comm_num)

    for gene in node_set:
        sigPartition[gene] = -1
        attrs[gene] = {"community": -1}

    for num in smallComms:
        sigCommNumbers.remove(num)
        del funcAnnots[num]

    # Add attributes to each node of graph
    nx.set_node_attributes(PPIGraph.network, attrs)

    return funcAnnots