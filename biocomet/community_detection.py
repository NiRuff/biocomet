import community as community_louvain  # python-louvain package
import leidenalg
import numpy as np


def apply_louvain(G, iterations=10):
    # Apply Louvain method to the graph
    partition_list = []
    modularity_list = []

    for i in range(iterations):
        # old louvain approach
        partition = community_louvain.best_partition(G, weight='weight')
        modularity = community_louvain.modularity(partition, G, weight='weight')

        partition_list.append(partition)
        modularity_list.append(modularity)

    max_value = max(modularity_list)
    max_index = modularity_list.index(max_value)

    # Get the weights of the igraph object
    weights = [data['weight'] for _, _, data in G.edges(data=True)]

    # Print the first few weights
    print("First few edge weights:", weights[:5])

    # Calculate and print summary statistics
    print("Mean weight:", np.mean(weights))
    print("Minimum weight:", np.min(weights))
    print("Maximum weight:", np.max(weights))

    print("The best modularity based on the networkx is {}".format(max_value))

    return partition_list[max_index]

def apply_leiden(ig_graph, iterations=10):
    # Apply Leiden algorithm to the graph
    partition_list = []
    modularity_list = []

    # Get the weights of the igraph object
    weights = ig_graph.es["weight"]

    # Print the first few weights
    print("First few edge weights:", weights[:5])

    # Calculate and print summary statistics
    print("Mean weight:", np.mean(weights))
    print("Minimum weight:", np.min(weights))
    print("Maximum weight:", np.max(weights))

    for i in range(iterations):
        # Apply the Leiden algorithm
        partition = leidenalg.find_partition(ig_graph, leidenalg.ModularityVertexPartition,
                                             weights=ig_graph.es["weight"], n_iterations=5)
        # Calculate modularity
        modularity = partition.modularity

        partition_list.append(partition)
        modularity_list.append(modularity)

    max_value = max(modularity_list)
    max_index = modularity_list.index(max_value)

    print("The best modularity based on the networkx is {}".format(max_value))

    best_partition = partition_list[max_index]
    membership = best_partition.membership

    # Map back to gene names
    gene_to_community = {ig_graph.vs[idx]['name']: community for idx, community in enumerate(membership)}

    return gene_to_community  # Create a dictionary mapping node names to communities

