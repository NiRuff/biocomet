import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import networkx as nx
import nxviz as nv
from nxviz import annotate
from wordcloud import WordCloud
import pathlib
from collections import Counter
import numpy as np
import seaborn as sns
import pandas as pd
from IPython.display import Image, display
import requests
import io
import re
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from io import BytesIO


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


def plotWordCloudsPPI(PPIGraph, categories='default', show=True):
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
        if show:
            plt.show()
        plt.close()


def visualize_KEGG(pathway_id, gene_reg_dict, organism, plot_dir=".", transparency=.5, community=None, show=True):
    gene_uniprot_dict = convert_gene_symbols_to_uniprot_mygene(gene_reg_dict.keys(), organism=organism)

    uniprot_reg_dict = {gene_uniprot_dict[k]: v for k, v in gene_reg_dict.items()}

    kegg_uniprots_dict = uniprot_to_kegg_dict(uniprot_reg_dict.keys())

    kegg_reg_dict = {k: [uniprot_reg_dict[uni] for uni in v if uni in uniprot_reg_dict] for k, v in
                     kegg_uniprots_dict.items()}

    annotate_genes_on_pathway(pathway_id, kegg_reg_dict, plot_dir=plot_dir, transparency=transparency, community=community, show=show)


def annotate_genes_on_pathway(pathway_id, kegg_reg_dict, plot_dir=".", transparency=.5, community=None, show=True):

    # ensure dir existence
    pathlib.Path(plot_dir + "/KEGG/").mkdir(parents=True, exist_ok=True)
    if community:
        pathlib.Path(plot_dir + "/KEGG/" + str(community) + "/").mkdir(parents=True, exist_ok=True)

    # Fetch the KGML content to get information about KEGG IDs
    kgml_content = fetch_pathway_kgml(pathway_id)
    graphics_info = parse_kegg_ids_from_kgml_v2(kgml_content)

    # Download the pathway image
    match = re.search(r"\d", pathway_id)
    if match:
        index = match.start()
        organism_code = pathway_id[:index]
        image_url = f"https://www.kegg.jp/kegg/pathway/{organism_code}/{pathway_id}.png"
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGBA")
    else:
        raise AttributeError("Invalid pathway ID format.")

    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    # Define color map and normalization
    min_expr = min(min(values) for values in kegg_reg_dict.values())
    max_expr = max(max(values) for values in kegg_reg_dict.values())
    norm = mcolors.TwoSlopeNorm(vmin=min_expr, vcenter=0, vmax=max_expr)
    cmap = plt.get_cmap('coolwarm')

    for graphic_id, (positional_info, kegg_ids) in graphics_info.items():
        x, y, w, h = positional_info['x'], positional_info['y'], positional_info['width'], positional_info['height']

        # Collect all regulations associated with the KEGG IDs of this graphic object
        all_regulations = [kegg_reg_dict[kegg_id] for kegg_id in kegg_ids if kegg_id in kegg_reg_dict]

        # Flatten the list of lists into a single list of regulations
        regulations = [reg for sublist in all_regulations for reg in sublist]

        num_regs = len(regulations)
        part_width = w / max(num_regs, 1)

        for i, reg in enumerate(regulations):
            color_value = cmap(norm(reg))
            color = tuple(int(255 * c) for c in color_value[:3]) + (
            int(255 * transparency),)  # Modify the alpha value as needed
            part_x = x - w / 2 + part_width * i
            draw.rectangle([part_x, y - h / 2, part_x + part_width, y + h / 2], fill=color)

    # Blend the overlay with the original image
    img_with_overlay = Image.alpha_composite(img, overlay)

    plt.figure(figsize=(10, 10))
    plt.imshow(img_with_overlay)
    plt.axis('off')

    plt.tight_layout()
    if community:  # if specified
        file_name = plot_dir + '/KEGG/' + str(community) + pathway_id + '_with_table.png'
    else:  # if unspecified
        file_name =plot_dir + "/KEGG/" + "/" + pathway_id + '_with_table.png'

    plt.savefig(file_name, dpi=300)
    print("Saving KEGG pathway word clouds to %s" %file_name)
    if show:
        plt.show()

def fetch_pathway_kgml(pathway_id):
    url = f"http://rest.kegg.jp/get/{pathway_id}/kgml"
    response = requests.get(url)
    if response.ok:
        return response.content  # Returns the content of the KGML file
    else:
        print(f"Failed to fetch KGML for pathway {pathway_id}")
        return None


def parse_kegg_ids_from_kgml_v2(kgml_content):
    root = ET.fromstring(kgml_content)
    graphics_dict = {}
    for entry in root.findall(".//entry[@type='gene']"):
        entry_id = entry.get('id')  # Unique identifier for each graphic object
        graphics = entry.find('.//graphics')
        if graphics is not None:
            # Extracting positional information
            x = int(graphics.attrib.get('x', 0))
            y = int(graphics.attrib.get('y', 0))
            width = int(graphics.attrib.get('width', 0))
            height = int(graphics.attrib.get('height', 0))
            positional_info = {'x': x, 'y': y, 'width': width, 'height': height}

            # Extracting KEGG IDs associated with this graphic object
            entry_names = entry.get('name', '').split()

            # Creating the dictionary entry
            graphics_dict[entry_id] = (positional_info, entry_names)
    return graphics_dict

def convert_gene_symbols_to_uniprot_mygene(gene_symbols, organism='9606'):
    base_url = "https://mygene.info/v3/query"
    gene_to_uniprot = {}  # Dictionary to store gene symbol to UniProt ID mappings
    for gene_symbol in gene_symbols:
        params = {
            'q': gene_symbol,
            'scopes': 'symbol',
            'fields': 'uniprot',
            'species': organism
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'hits' in data and len(data['hits']) > 0:
                # Extracting UniProt ID from the first hit
                hit = data['hits'][0]
                if 'uniprot' in hit and 'Swiss-Prot' in hit['uniprot']:
                    gene_to_uniprot[gene_symbol] = hit['uniprot']['Swiss-Prot']
                elif 'uniprot' in hit and 'TrEMBL' in hit['uniprot']:
                    gene_to_uniprot[gene_symbol] = hit['uniprot']['TrEMBL']
                else:
                    gene_to_uniprot[gene_symbol] = None
                    print(f"No UniProt ID found for the gene symbol: {gene_symbol}.")
            else:
                gene_to_uniprot[gene_symbol] = None
                print(f"No UniProt ID found for the gene symbol: {gene_symbol}.")
        else:
            print(f"Failed to fetch data from MyGene.info API for {gene_symbol}. Status code: {response.status_code}")
            gene_to_uniprot[gene_symbol] = None
    return gene_to_uniprot


def uniprot_to_kegg_dict(uniprot_ids):
    kegg_to_uniprots = {}  # Dictionary to store KEGG ID to UniProt IDs mappings
    for uniprot_id in uniprot_ids:
        url = f"http://rest.kegg.jp/conv/genes/uniprot:{uniprot_id}"
        response = requests.get(url)
        if response.status_code == 200:
            for line in response.text.strip().split('\n'):
                parts = line.split('\t')
                if len(parts) == 2:
                    kegg_id = parts[1]  # Extracting KEGG ID
                    if kegg_id not in kegg_to_uniprots:
                        kegg_to_uniprots[kegg_id] = [uniprot_id]
                    else:
                        kegg_to_uniprots[kegg_id].append(uniprot_id)
        else:
            print(f"Failed to fetch data for UniProt ID {uniprot_id}. Status code: {response.status_code}")
    return kegg_to_uniprots