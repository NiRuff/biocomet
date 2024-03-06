import networkx as nx

def calc_w_closeness_centrality(G, weight):
    # Extract weights and find min and max
    weights = [d[weight] for _, _, d in G.edges(data=True)]
    min_weight = min(weights)
    max_weight = max(weights)

    # Check for the case where all weights are the same
    if max_weight == min_weight:
        # Handle this case appropriately.
        # For example, you could assign a default distance or skip scaling.
        # Here, I set all distances to a default value (like 1)
        for _, _, d in G.edges(data=True):
            d['distance'] = 1
    else:
        # Scale inversely: higher weight gets lower distance
        for u, v, d in G.edges(data=True):
            d['distance'] = 1 - (d[weight] - min_weight) / (max_weight - min_weight)

    return nx.closeness_centrality(G, distance='distance')



def calcCentrality(G, partition, full_network=False):

    if full_network: # set only one community, 'Full Network'
        communities = {'Full Network':[n for n in G.nodes()]}

    else:
        # Invert 'partition' to get a dictionary of communities with their respective genes
        communities = {}
        for gene, comm in partition.items():
            communities.setdefault(comm, []).append(gene)

    # Initialize the partition_centrality dictionary
    partition_centrality = {comm: {} for comm in communities}

    # Calculate centrality measures for each community
    for comm, genes in communities.items():
        # Create a subgraph of G containing only nodes in this community
        subgraph = G.subgraph(genes)

        # Calculate degree centrality for the subgraph
        degree_centrality = nx.degree_centrality(subgraph)
        w_degree_centrality = {node: sum(data['weight'] for _, _, data in G.edges(node, data=True)) for node in
                             subgraph.nodes()}
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        w_betweenness_centrality = nx.betweenness_centrality(subgraph, weight='weight')
        closeness_centrality = nx.closeness_centrality(subgraph)
        w_closeness_centrality = calc_w_closeness_centrality(subgraph, weight='weight')
        eigenvector_centrality = nx.eigenvector_centrality(subgraph)
        w_eigenvector_centrality = nx.eigenvector_centrality(subgraph, weight='weight')
        katz_centrality_centrality = nx.katz_centrality(subgraph)
        w_katz_centrality_centrality = nx.katz_centrality(subgraph, weight='weight')

        # Store the centrality measures in partition_centrality for each gene
        for gene in genes:
            partition_centrality[comm][gene] = {
                'degree': degree_centrality[gene],
                'w_degree': w_degree_centrality[gene],
                'betweenness': betweenness_centrality[gene],
                'w_betweenness': w_betweenness_centrality[gene],
                'closeness': closeness_centrality[gene],
                'w_closeness': w_closeness_centrality[gene],
                'eigenvector': eigenvector_centrality[gene],
                'w_eigenvector': w_eigenvector_centrality[gene],
                'katz': katz_centrality_centrality[gene],
                'w_katz': w_katz_centrality_centrality[gene]
            }

    return partition_centrality
