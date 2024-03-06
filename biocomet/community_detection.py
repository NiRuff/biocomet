import community as community_louvain  # python-louvain package
import leidenalg
import numpy as np
import random

def apply_louvain(G, iterations=10, seed=None):

    # Apply Louvain method to the graph
    partition_list = []
    modularity_list = []

    if seed:
        partition = community_louvain.best_partition(G, weight='weight', seed=seed)
        modularity = community_louvain.modularity(partition, G, weight='weight')

        print("The modularity based on this seed is {}".format(modularity))

        return partition

    else:
        # Generate a list of random seeds, one for each iteration
        random_seeds = [random.randint(0, 10000) for _ in range(iterations)]

        for i, seed in enumerate(random_seeds):
            partition = community_louvain.best_partition(G, weight='weight', seed=seed)
            modularity = community_louvain.modularity(partition, G, weight='weight')

            partition_list.append(partition)
            modularity_list.append(modularity)

        max_value = max(modularity_list)
        max_index = modularity_list.index(max_value)

        print("The best modularity based on the networkx is {}".format(max_value))
        print("For reproducing exactly this community detection, "
              "run PPIGraph.community_detection(algorithm='louvain', seed={})".format(random_seeds[max_index]))

        return partition_list[max_index]

def apply_leiden(ig_graph, iterations=10, seed=None):
    # Apply Leiden algorithm to the graph
    partition_list = []
    modularity_list = []

    if seed:
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition,
                                             weights=ig_graph.es["weight"], n_iterations=-1, seed=seed)
        # Calculate modularity
        modularity = partition.modularity

        print("The modularity based on this seed is {}".format(modularity))

        membership = partition.membership

        # Map back to gene names
        gene_to_community = {ig_graph.vs[idx]['name']: community for idx, community in enumerate(membership)}

        return gene_to_community  # Create a dictionary mapping node names to communities

    else:
        # Generate a list of random seeds, one for each iteration
        random_seeds = [random.randint(0, 10000) for _ in range(iterations)]
        for i, seed in enumerate(random_seeds):
            # Apply the Leiden algorithm
            partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition,
                                                 weights=ig_graph.es["weight"], n_iterations=-1, seed=seed)
            # Calculate modularity
            modularity = partition.modularity

            partition_list.append(partition)
            modularity_list.append(modularity)

            max_value = max(modularity_list)
            max_index = modularity_list.index(max_value)

        print("The best modularity based on the networkx is {}".format(max_value))
        print("For reproducing exactly this community detection, "
              "run PPIGraph.community_detection(algorithm='leiden', seed={})".format(random_seeds[max_index]))

        best_partition = partition_list[max_index]
        membership = best_partition.membership

        # Map back to gene names
        gene_to_community = {ig_graph.vs[idx]['name']: community for idx, community in enumerate(membership)}

        return gene_to_community  # Create a dictionary mapping node names to communities

