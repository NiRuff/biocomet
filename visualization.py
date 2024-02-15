import matplotlib.pyplot as plt
import networkx as nx
from wordcloud import WordCloud
import pathlib
from collections import Counter
import nxviz as nv
import numpy as np
from matplotlib.lines import Line2D
import seaborn as sns
import matplotlib.patches as mpatches
import pandas as pd
import requests
from IPython.display import Image, display
import matplotlib.image as mpimg
import io
from nxviz import annotate
import xml.etree.ElementTree as ET
from io import BytesIO
from PIL import Image, ImageDraw
import re
import matplotlib.colors as mcolors

def plot_nv(G, sigPartition, plot_dir, legend=True, kind='ArcPlots'):

    pathlib.Path(plot_dir + "/" + kind + "/").mkdir(parents=True, exist_ok=True)

    community_sizes = Counter(sigPartition.values())

    if len(set(sigPartition.keys())) > 12:
        # Sort communities by size and keep only the 12 largest
        largest_communities = sorted(community_sizes, key=community_sizes.get, reverse=True)[:12]
        removed_communities = set(sigPartition.values()) - set(largest_communities)

        # Truncate G to include only nodes from the 12 largest communities
        nodes_to_remove = [node for node, comm in sigPartition.items() if comm not in largest_communities]
        G_trunc = G.copy()
        G_trunc.remove_nodes_from(nodes_to_remove)

        print(
            f"G has been truncated to include only the 12 largest communities. Communities removed: {removed_communities}")
    else:
        G_trunc = G.copy()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if kind == 'ArcPlots':
        g = nv.arc(G_trunc, node_color_by="community", group_by="community", edge_color_by="weight", edge_alpha_by="weight")

        nv.annotate.arc_group(G_trunc, group_by="community")

    elif kind == 'CircosPlots':
        g = nv.circos(G_trunc, node_color_by="community", group_by="community", edge_color_by="weight",
                      edge_alpha_by="weight")

        nv.annotate.circos_group(G_trunc, group_by="community")

    g.get_figure().set_size_inches(10, 10)

    plt.tight_layout()
    plt.autoscale()

    if legend:
        # Get edge weights
        weights = np.array([float(w) for w in nx.get_edge_attributes(G_trunc, 'weight').values()])
        # in case something with the weights went wrong and they are 0-1 scaled
        if all([num < 1 for num in weights]):
            weights = weights * 1000

        # Get min and max values
        min_wt = np.min(weights)
        max_wt = np.max(weights)

        # Create four evenly spaced values in this range
        # Make sure they are integers and divisible by 50
        legend_values = np.linspace(min_wt, max_wt, 4)
        legend_values = (np.round(legend_values / 50) * 50).astype(np.int64)

        cmap = plt.cm.viridis
        custom_lines = [Line2D([0], [0], color=cmap(i / 3.), lw=6) for i in range(4)]
        ax.legend(custom_lines, legend_values, title="Score")

    file_name = plot_dir + "/" + kind + "/PartitionedNetwork.png"
    print("Saving plots to %s" % file_name)
    plt.tight_layout()
    plt.savefig(file_name, dpi=300)


def plotWordclouds(funcAnnots, categories='default', plot_dir='.'):
    if type(categories) != str:
        pass
    elif categories.lower() == 'pathways':
        categories = ["KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'default':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'no_pmid':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro", ]
    elif categories.lower() == 'no_go':
        categories = ["KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro", ]
    # gather all categories
    elif categories.lower() == 'all':
        categories = set()
        for df in funcAnnots.values():
            categories.update(df['category'].unique())
        categories = list(categories)

    pathlib.Path(plot_dir + "/WordClouds/").mkdir(parents=True, exist_ok=True)

    # Create a color map
    colors = sns.color_palette('Dark2', len(categories)).as_hex()
    color_map = dict(zip(categories, colors))

    for commNum, df in funcAnnots.items():
        if categories != 'all':
            df = df[df['category'].isin(categories)]

        # Create a word cloud
        wc = WordCloud(background_color='white', width=1600, height=800,)
        weights = dict(zip(df['description'], -np.log10(df['fdr'])))
        try:
            wc.generate_from_frequencies(weights)
        except ValueError:
            # If no words are available, generate a word cloud with placeholder text
            wc.generate_from_text('No significant functional enrichment found for the specified databases.')

        # Recolor the words
        def color_func(word, *args, **kwargs):
            category = df.loc[df['description'] == word, 'category'].values[0]
            return color_map[category]

        wc.recolor(color_func=color_func)

        # Display the word cloud
        plt.figure(commNum)
        plt.tight_layout()
        plt.gcf().set_size_inches(15,8)
        plt.title(f'Community {commNum}: Functional Annotation')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')

        # Create legend
        patches = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        file_name = plot_dir + "/WordClouds/community " + str(commNum) + "'s_wordcloud.png"
        print("Saving word clouds to %s" % file_name)

        plt.savefig(file_name, dpi=300)
        plt.show()
        plt.close()


def plotPPI(PPIGraph):

    pathlib.Path(PPIGraph.plot_dir + "/PPI_networks/").mkdir(parents=True, exist_ok=True)

    if PPIGraph.physical:
        network_type = "physical"
    else:
        network_type = "functional"

    string_api_url = "https://string-db.org/api"
    output_format = "image"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])

    # create dict of commNum: all comm genes
    allCommGeneSets = dict()
    for commNum in pd.Series(PPIGraph.partition.values()).sort_values().unique():
        commGeneSet = [k for k, v in PPIGraph.partition.items() if v == commNum]
        allCommGeneSets[commNum] = commGeneSet

    # add comm -1 to funcAnnots:
    # PPIGraph.func_annotation[-1] = pd.DataFrame({'description': ['No significant functional enrichment'],
    #                                'fdr': [0.05],
    #                                'category': ['RCTM']
    #                                })

    for commNum, df in PPIGraph.func_annotation.items():

        # PPI network part
        commGeneSet = allCommGeneSets[commNum]

        params = {
            "identifiers": "%0d".join(commGeneSet),  # your protein
            "species": PPIGraph.organism,  # species NCBI identifier
            "network_flavor": "actions",  # show confidence links
            "caller_identity": "comet",  # your app name
            "required_score": str(PPIGraph.min_score),
            "network_type": network_type  # network type
        }

        response = requests.post(request_url, data=params)

        file_name = PPIGraph.plot_dir + "/PPI_networks/community " + str(commNum) + "'s_network.png"
        print("Saving interaction network to %s" % file_name)

        with open(file_name, 'wb') as fh:
            fh.write(response.content)
        #
        # image = Image(response.content)
        # display(image)


def plotWordCloudsPPI(PPIGraph, categories='default'):
    if type(categories) != str:
        pass
    elif categories.lower() == 'pathways':
        categories = ["KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'default':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM"]
    elif categories.lower() == 'no_pmid':
        categories = ["Process", "Function", "Component", "KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro", ]
    elif categories.lower() == 'no_go':
        categories = ["KEGG", "WikiPathways", "RCTM",
                      "NetworkNeighborAL", "SMART", "COMPARTMENTS", "Keyword", "TISSUES", "Pfam",
                      "MPO", "InterPro", ]
    # gather all categories
    elif categories.lower() == 'all':
        categories = set()
        for df in PPIGraph.func_annotation.values():
            categories.update(df['category'].unique())

    pathlib.Path(PPIGraph.plot_dir + "/WordCloudPPI_networks/").mkdir(parents=True, exist_ok=True)

    if PPIGraph.physical:
        network_type = "physical"
    else:
        network_type = "functional"

    # Create a color map
    colors = sns.color_palette('Dark2', len(categories)).as_hex()
    color_map = dict(zip(categories, colors))

    string_api_url = "https://string-db.org/api"
    output_format = "image"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])

    # create dict of commNum: all comm genes
    allCommGeneSets = dict()
    for commNum in pd.Series(PPIGraph.partition.values()).sort_values().unique():
        commGeneSet = [k for k, v in PPIGraph.partition.items() if v == commNum]
        allCommGeneSets[commNum] = commGeneSet

    # add comm -1 to funcAnnots:
    funcAnnots = PPIGraph.func_annotation.copy()
    # funcAnnots[-1] = pd.DataFrame({'description': ['No significant functional enrichment'],
    #                                'fdr': [0.05],
    #                                'category': ['RCTM']
    #                                })

    for commNum, df in funcAnnots.items():
        df = df[df['category'].isin(categories)]

        # Create a word cloud
        wc = WordCloud(background_color='white', width=1600, height=800,)
        weights = dict(zip(df['description'], -np.log10(df['fdr'])))
        try:
            wc.generate_from_frequencies(weights)

            # Recolor the words
            def color_func(word, *args, **kwargs):
                category = df.loc[df['description'] == word, 'category'].values[0]
                return color_map[category]

            wc.recolor(color_func=color_func)

        except ValueError:
            # If no words are available, generate a word cloud with placeholder text
            wc.generate_from_text('No significant functional enrichment found for the specified databases.')

        # PPI network part
        commGeneSet = allCommGeneSets[commNum]
        # commNum = pd.Series(sigPartition.values()).sort_values().unique()[i]

        params = {
            "identifiers": "%0d".join(commGeneSet),  # your protein
            "species": PPIGraph.organism,  # species NCBI identifier
            "network_flavor": "actions",  # show confidence links
            "caller_identity": "comet",  # your app name
            "required_score": str(PPIGraph.min_score),
            "network_type": network_type  # network type
        }

        response = requests.post(request_url, data=params)

        # Create a figure with two subplots (1 row, 2 columns)
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # Display the image in the first subplot
        img = mpimg.imread(io.BytesIO(response.content))
        axs[0].imshow(img)
        axs[0].axis('off')  # Hide the axes on the image plot

        # Display the word cloud
        # plt.figure(i)
        # plt.title(f'Community {i}: Functional Annotation')
        axs[1].imshow(wc)
        axs[1].axis('off')
        axs[1].set_title(f'Community {commNum}: Functional Annotation')

        # Create legend
        patches = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
        axs[1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        file_name = PPIGraph.plot_dir + "/WordCloudPPI_networks/community " + str(commNum) + "'s_ppi_wordcloud.png"
        print("Saving PPI word clouds to %s" % file_name)
        plt.tight_layout()
        plt.savefig(file_name, dpi=300)
        plt.show()
        plt.close()


def fetch_pathway_kgml(pathway_id):
    url = f"http://rest.kegg.jp/get/{pathway_id}/kgml"
    response = requests.get(url)
    if response.ok:
        return response.content  # Returns the content of the KGML file
    else:
        print(f"Failed to fetch KGML for pathway {pathway_id}")
        return None

def annotate_genes_on_pathway(pathway_id, expression_data, plot_dir=".", community=None):

    # ensure dir existence
    pathlib.Path(plot_dir + "/KEGG/").mkdir(parents=True, exist_ok=True)
    if community:
        pathlib.Path(plot_dir + "/KEGG/" + str(community) + "/").mkdir(parents=True, exist_ok=True)

    kgml_content = fetch_pathway_kgml(pathway_id)

    kegg_id_info = parse_kegg_ids_from_kgml(kgml_content)

    # Download the pathway image
    # Find the index of the first digit in the pathway ID
    match = re.search(r"\d", pathway_id)
    if match:
        index = match.start()
        organism_code = pathway_id[:index]
        image_url = f"https://www.kegg.jp/kegg/pathway/{organism_code}/{pathway_id}.png"
    else:
        raise AttributeError("pathway_id does not match pattern of 2-4 letter followed by a serie sof digits")

    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    draw = ImageDraw.Draw(img)

    # Update color scheme to use coolwarm colormap adjusted to regulation values
    min_expr, max_expr = min(expression_data.values()), max(expression_data.values())
    norm = mcolors.TwoSlopeNorm(vmin=min_expr, vcenter=0, vmax=max_expr)
    cmap = plt.get_cmap('coolwarm')

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(20, 25), gridspec_kw={'height_ratios': [4, 1]})

    # Plot the pathway image in the first subplot
    axs[0].imshow(img)
    axs[0].axis('off')

    # Prepare table data
    table_data = []
    # Assuming cmap and norm are defined as before
    # todo: handle what should happen if multiple aliases map to the same gene
    # !! refers to how to color, based on which expression?
    for gene, expression in expression_data.items():
        color_value = cmap(norm(expression))  # This returns a RGBA color
        color = tuple(int(255 * x) for x in color_value[:3])
        kegg_id = alias2kegg(gene, kegg_id_info)
        table_data.append([gene, kegg_id, expression])
        w, h = kegg_id_info[kegg_id]['width'], kegg_id_info[kegg_id]['height']
        draw.rectangle([kegg_id_info[kegg_id]['x'] - w / 2, kegg_id_info[kegg_id]['y'] - h / 2, kegg_id_info[kegg_id]['x'] + w / 2, kegg_id_info[kegg_id]['y'] + h / 2],
                       outline=color, width=3, fill=None)

    # Add table subplot
    col_labels = ["Gene ID", "Mapped Name", "Regulation"]
    axs[1].axis('off')
    axs[1].axis('tight')
    axs[1].table(cellText=table_data, colLabels=col_labels, loc='center')

    # Display colorbar for regulation values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axs[0], orientation='horizontal', fraction=0.046, pad=0.04, label='Regulation')

    plt.tight_layout()
    if community: # if specified
        plt.savefig(plot_dir + '/KEGG/' + str(community) + pathway_id + '_with_table.png', dpi=300)
    else: # if unspecified
        plt.savefig(plot_dir + "/KEGG/" + "/" + pathway_id + '_with_table.png', dpi=300)
    plt.show()

def alias2kegg(gene, kegg_id_info):
    for kegg_id, kegg_id_data in kegg_id_info.items():
        if gene in kegg_id_data["gene_aliases"]:
            return kegg_id



def parse_kegg_ids_from_kgml(kgml_content): # return 0,0,0,0 if data on gene is missing in the kegg file
    root = ET.fromstring(kgml_content)
    kegg_id_dict = {}
    for entry in root.findall(".//entry[@type='gene']"):
        graphics = entry.find('.//graphics')
        if graphics is not None:
            kegg_ids = entry.attrib['name'].split(" ")
            KEGG_OBJECT_gene_aliases = []
            KEGG_OBJECT = None
            for kegg_id in kegg_ids:
                response = requests.get("https://rest.kegg.jp/get/" + kegg_id)

                for line in response.text.split("\n"):
                    if line.startswith("SYMBOL"):
                        content = line.split(" ")
                        content = [x.strip(' ') for x in content]
                        aliases = [x for x in content if x not in ['', 'SYMBOL']]
                        for alias in aliases:
                            KEGG_OBJECT_gene_aliases.append(alias)

                    if line.startswith("ORTHOLOGY"):
                        KEGG_OBJECT_t = line.split("[EC:")[-1].strip("]")

                        # test if kegg object id contradicts each other
                        if KEGG_OBJECT and (KEGG_OBJECT != KEGG_OBJECT_t):
                            print('two different values for KEGG object found.')
                        else:
                            KEGG_OBJECT = KEGG_OBJECT_t
                        break

            x = int(graphics.attrib.get('x', 0))
            y = int(graphics.attrib.get('y', 0))
            width = int(graphics.attrib.get('width', 0))
            height = int(graphics.attrib.get('height', 0))
            kegg_id_dict['kegg_id'] = {'x': x, 'y': y, 'width': width, 'height': height, 'gene_aliases':KEGG_OBJECT_gene_aliases}
    return kegg_id_dict