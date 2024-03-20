import requests
import pandas as pd
import json
import networkx as nx



def checkFuncSignificanceFullNetwork(PPIGraph, categories='default'):

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

    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "enrichment"

    funcAnnots = dict()

    protein_list = PPIGraph.gene_list

    ## Construct URL
    request_url = "/".join([string_api_url, output_format, method])

    # 9606 for human, 10090 for mouse
    params = {
        "identifiers": "%0d".join(protein_list),  # your protein
        "species": PPIGraph.organism,  # species NCBI identifier
        "caller_identity": "comet"  # your app name
    }
    ## Call STRING
    response = requests.post(request_url, params=params)

    ## Read and parse the results
    data = json.loads(response.text)
    funcAnnots['Full Network'] = pd.DataFrame(data)

    return funcAnnots


def checkFuncSignificance(PPIGraph, sig_only=True,
                          categories='default',
                          min_comm_size=3):
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

    # get lists of all communites
    #communities = [[] for x in range(len(set(PPIGraph.partition.values())))]
    communities = dict()
    for comm in PPIGraph.partition.values():
        communities[comm] = []

    for gene, comm_num in PPIGraph.partition.items():
        communities[comm_num].append(gene)

    string_api_url = "https://string-db.org/api"
    output_format = "json"
    method = "enrichment"

    sigCommNumbers = set()
    funcAnnots = dict()

#    for i in range(len(communities)):
    for comm, genes in communities.items():

        protein_list = genes

        if len(protein_list) > 1000:
            raise ValueError("Too many proteins in community {} for performing functional enrichment. \n"
                             "Consider using a different community detection approach "
                             "or being more conservative when selecting genes for the analysis".format(comm))

        ## Construct URL
        request_url = "/".join([string_api_url, output_format, method])

        # 9606 for human, 10090 for mouse
        params = {
            "identifiers": "%0d".join(protein_list),  # your protein
            "species": PPIGraph.organism,  # species NCBI identifier
            "caller_identity": "comet"  # your app name
        }
        ## Call STRING
        response = requests.post(request_url, params=params)

        ## Read and parse the results
        data = json.loads(response.text)
        anyProcessSig = False

        for row in data:
            fdr = float(row["fdr"])
            category = row["category"]

            if categories != 'all': # filter for categories
                if (not sig_only) or (category in categories and fdr < PPIGraph.p_adj_cutoff):
                    anyProcessSig = True
                    sigCommNumbers.add(comm)

            else: # do not filter
                if (not sig_only) or (fdr < PPIGraph.p_adj_cutoff):
                    anyProcessSig = True
                    sigCommNumbers.add(comm)

        if anyProcessSig:
            funcAnnots[comm] = pd.DataFrame(data)

    # create dict for sig partition and attributes to add for nodes
    sigPartition = dict()
    attrs = dict()
    node_set = set()
    for n in PPIGraph.network.nodes:
        node_set.add(n)

    smallComms = []
    # add comm num to nodes. Assign -1 if node is not part of any significant comm
    for comm_num in sigCommNumbers:
        # only keep communities with length > min_comm_size
        if len(communities[comm_num]) >= min_comm_size:
            for gene in communities[comm_num]:
                attrs.update({gene: {"community": comm_num}})
                node_set.remove(gene)
                sigPartition[gene] = comm_num
        else:
            for gene in communities[comm_num]:
                sigPartition[gene] = -1
                attrs.update({gene: {"community": -1}})
            smallComms.append(comm_num)

    for gene in node_set:
        sigPartition[gene] = -1
        attrs.update({gene: {"community": -1}})

    for num in smallComms:
        sigCommNumbers.remove(num)
        del funcAnnots[num]

    # add attributes to each node of graph
    nx.set_node_attributes(PPIGraph.network, attrs)

    return funcAnnots