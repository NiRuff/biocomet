import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='nxviz')

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
mpl.rcParams['font.family'] = "monospace"  # change default font family

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
from PIL import Image, ImageDraw, ImageColor
from io import BytesIO


def plot_nv(G, sigPartition, plot_dir='.', legend=True, kind='ArcPlots', show=True, background='transparent'):

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
    print("Saving network plots to %s" % file_name)
    change_background_color(plt.gcf(), plt.gca(), background)

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()



def plotWordclouds(funcAnnots, categories='default', plot_dir='.', show=True, background='transparent'):
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
        if background == 'transparent':
            wc = WordCloud(background_color=None, mode="RGBA", width=1600, height=800, )
        else:
            wc = WordCloud(background_color=background, width=1600, height=800,)
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
        plt.gcf().set_size_inches(15,8)
        plt.title(f'Community {commNum}: Functional Annotation')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')

        # Create legend
        patches = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')

        change_background_color(plt.gcf(), plt.gca(), background)

        file_name = plot_dir + "/WordClouds/community " + str(commNum) + "'s_wordcloud.png"
        print("Saving word clouds to %s" % file_name)

        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

def change_background_color(fig, ax, background):
    # Set the entire figure background color
    if background == 'transparent':
        fig.patch.set_alpha(0)  # Make the background of the figure transparent
        # For the subplots, in case you want them transparent as well
        ax.patch.set_alpha(0)
    else:
        # Convert color name or hex to RGBA tuple
        if isinstance(background, str):
            # Simple conversion for known colors, extend this as needed
            color_converter = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0)}
            new_background = color_converter.get(background.lower(), None)  # Default None if color is unknown
            if background.startswith('#'):  # For hex colors
                # Directly use the hex color for the facecolor
                new_background = background
        else:
            new_background = background  # Assuming background is already an RGBA tuple or hex color

        if new_background:
            fig.patch.set_facecolor(new_background)
            # Also set subplot backgrounds if needed
            ax.set_facecolor(new_background)


def plotPPI(PPIGraph, full_network=False, show=True, background='transparent'):
    pathlib.Path(PPIGraph.plot_dir + "/PPI_networks/").mkdir(parents=True, exist_ok=True)

    if PPIGraph.physical:
        network_type = "physical"
    else:
        network_type = "functional"

    string_api_url = "https://string-db.org/api"
    output_format = "image"
    method = "network"

    request_url = "/".join([string_api_url, output_format, method])

    if full_network:
        # add code for plotting the full network


        params = {
            "identifiers": "%0d".join(PPIGraph.gene_list),  # your protein
            "species": PPIGraph.organism,  # species NCBI identifier
            "network_flavor": "actions",  # show confidence links
            "caller_identity": "comet",  # your app name
            "required_score": str(PPIGraph.min_score),
            "network_type": network_type  # network type
        }

        response = requests.post(request_url, data=params)

        file_name = PPIGraph.plot_dir + "/PPI_networks/full_network.png"
        print("Saving interaction network to %s" % file_name)

        change_background_color(plt.gcf(), plt.gca(), background)

        with open(file_name, 'wb') as fh:
            fh.write(response.content)

        if show:
            image = Image.open(file_name)
            display(image)
        plt.close()


    else:

        # create dict of commNum: all comm genes
        allCommGeneSets = dict()
        for commNum in pd.Series(PPIGraph.partition.values()).sort_values().unique():
            commGeneSet = [k for k, v in PPIGraph.partition.items() if v == commNum]
            allCommGeneSets[commNum] = commGeneSet

        for commNum, commGeneSet in allCommGeneSets.items():

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

            change_background_color(plt.gcf(), plt.gca(), background)

            with open(file_name, 'wb') as fh:
                fh.write(response.content)

            if show:
                image = Image.open(file_name)
                display(image)


def plotWordCloudsPPI(PPIGraph, categories='default', show=True, background='transparent'):
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
        if background == 'transparent':
            wc = WordCloud(background_color=None, mode="RGBA", width=1600, height=800, )
        else:
            wc = WordCloud(background_color=background, width=1600, height=800, )

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

        change_background_color(fig, axs[0], background)
        change_background_color(fig, axs[1], background)

        print("Saving PPI word clouds to %s" % file_name)
        plt.savefig(file_name, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        plt.close()


def visualize_KEGG(pathway_id, gene_reg_dict, organism, plot_dir=".", transparency=.5, community=None, show=True, background='transparent'):
    gene_uniprot_dict = convert_gene_symbols_to_uniprot_mygene(gene_reg_dict.keys(), organism=organism)

    uniprot_reg_dict = {}
    for gene, uniprot in gene_uniprot_dict.items():
        if isinstance(uniprot, list):
            print(f"Multiple UniProt IDs found for {gene}.")
            # Join the UniProt IDs into a single string separated by ", "
            uni_ids_str = ", ".join(uniprot)
            for uni in uniprot:
                uniprot_reg_dict[uni] = gene_reg_dict[gene]
            # Print the gene mapping to the joined string of UniProt IDs
            print(f"Mapping {gene} to {uni_ids_str}.")
        else:
            uniprot_reg_dict[uniprot] = gene_reg_dict[gene]

    kegg_uniprots_dict = uniprot_to_kegg_dict(uniprot_reg_dict.keys())

    kegg_reg_dict = {k: [uniprot_reg_dict[uni] for uni in v if uni in uniprot_reg_dict] for k, v in
                     kegg_uniprots_dict.items()}

    annotate_genes_on_pathway(pathway_id, kegg_reg_dict, plot_dir=plot_dir, transparency=transparency, community=community, show=show, background=background)


def annotate_genes_on_pathway(pathway_id, kegg_reg_dict, plot_dir=".", transparency=.5, community=None, show=True, background='transparent'):

    # ensure dir existence
    pathlib.Path(plot_dir + "/KEGG/").mkdir(parents=True, exist_ok=True)
    if community is not None:
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
    # Adjust min_expr and max_expr to be le/ge than 0
    min_expr = min(min_expr, -1)
    max_expr = max(max_expr, +1)
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

    # Create a new ImageDraw object for img_with_overlay
    draw_overlay = ImageDraw.Draw(img_with_overlay)

    # Then proceed with your legend drawing code, but use draw_overlay instead of draw
    # Parameters for the legend remain the same
    legend_width = 100  # Width of the legend
    legend_height = 20  # Height of the legend
    margin = 10  # Margin from the top and right edges
    text_offset = 10  # Offset for the text below the legend

    # Calculate legend position
    legend_top = margin*1.5
    legend_right = img_with_overlay.width - margin*2

    # Create a gradient legend on img_with_overlay
    for i in range(legend_width):
        ratio = i / legend_width
        color_value = cmap(ratio)
        color = tuple(int(255 * c) for c in color_value[:3]) + (255,)  # Full opacity
        draw_overlay.rectangle([legend_right - legend_width + i, legend_top,
                                legend_right - legend_width + i + 1, legend_top + legend_height], fill=color)

    # Text annotations for vmin, vcenter, and vmax, using draw_overlay
    draw_overlay.text((legend_right - legend_width, legend_top + legend_height + text_offset), f"{min_expr:.1f}",
                      fill="black")
    draw_overlay.text((legend_right - legend_width / 2, legend_top + legend_height + text_offset), "0", fill="black")
    # Before drawing the text, calculate the text width
    max_expr_text = f"{max_expr:.1f}"
    text_width = draw_overlay.textlength(max_expr_text)
    draw_overlay.text((legend_right - text_width, legend_top + legend_height + text_offset), max_expr_text,
                      fill="black")

    # Continue to display or save img_with_overlay as before
    plt.figure(figsize=(20, 20))
    plt.imshow(img_with_overlay)
    plt.axis('off')

    change_background_color(plt.gcf(), plt.gca(), background)

    if community is not None:  # if specified
        file_name = plot_dir + '/KEGG/' + str(community) + '/' + pathway_id + '.png'
    else:  # if unspecified
        file_name = plot_dir + "/KEGG/" + pathway_id + '.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print("Saving KEGG pathways to %s" %file_name)
    if show:
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

def scale_parameters_based_on_network_size(G, base_node_size=1500, base_font_size=8, base_edge_width=4, base_fig_size=12):
    """
    Adjust node size, font size, and edge width based on the number of nodes in the graph.

    :param G: The graph for which to scale parameters.
    :param base_node_size: Base node size to scale from.
    :param base_font_size: Base font size to scale from.
    :param base_edge_width: Base edge width to scale from.
    :return: Tuple of (node_size, font_size, edge_width) after scaling.
    """
    num_nodes = len(G.nodes)

    # Define scaling factors - these values are adjustable based on desired appearance
    if num_nodes < 25:
        scale_factor = 1.0
    elif num_nodes < 50:
        scale_factor = 0.8
    elif num_nodes < 75:
        scale_factor = 0.6
    elif num_nodes < 100:
        scale_factor = 0.4
    elif num_nodes < 200:
        scale_factor = 0.3
    else:
        scale_factor = 0.25

    font_size = base_font_size / scale_factor
    fig_size = base_fig_size / scale_factor

    return font_size, fig_size

def plotRegNetworks(G, partition, plot_dir=".", full_network=False, community='all', show=True, background='transparent'):

    # ensure dir existence
    pathlib.Path(plot_dir + "/regNetworks/").mkdir(parents=True, exist_ok=True)

    if full_network: #community parameter ignored here
        # Create a colormap for the nodes based on their 'regulation' attribute
        cmap = plt.cm.coolwarm

        # Adjust vmin and vmax for specific community
        nodes = G.nodes
        regulations = [G.nodes[node]['regulation'] for node in nodes]
        vmin = min(regulations) if min(regulations) < -1 else -1
        vmax = max(regulations) if max(regulations) > 1 else 1
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        # Create the original greyscale colormap for edges
        cmap_grey = plt.cm.Greys
        dark_grey_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "dark_grey", cmap_grey(np.linspace(0.3, 1, 256))
        )

        # Calculate the layout of the graph
        pos = nx.kamada_kawai_layout(G)

        # Calculate dynamic sizes
        font_size_legend, fig_size = scale_parameters_based_on_network_size(G)

        # Prepare figure
        plt.figure(figsize=(fig_size, fig_size))
        plt.title(f'Network Visualization for Full Network')

        # Node colors and sizes based on 'regulation' attribute
        node_colors = [cmap(norm(G.nodes[node]['regulation'])) for node in G.nodes]

        # Normalize edge weights for width and alpha
        edge_weights_raw = np.array([G.edges[edge]['weight'] for edge in G.edges])
        edge_weights = edge_weights_raw * 4  # Now scaling with dynamic edge width
        edge_alphas = np.interp(edge_weights_raw, (edge_weights_raw.min(), edge_weights_raw.max()), (0.1, 1))

        # Draw the network with dynamic sizes
        nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color=edge_alphas, edge_cmap=dark_grey_cmap, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, node_size=1500, node_color=node_colors, cmap=cmap)

        # Adjust label drawing to use the dynamic font size
        for node, (x, y) in pos.items():
            text = node
            plt.text(x, y, text, fontsize=8, ha='center', va='center',
                     bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))

        # Draw color reference rectangle in the upper right corner
        ax = plt.gca()
        # Create a colorbar as a legend
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Regulation', fontsize=font_size_legend)  # Adjust the font size for the label here
        cbar.ax.invert_yaxis()  # Invert to match coolwarm orientation

        # Adjust font size for the tick labels
        cbar.ax.tick_params(labelsize=font_size_legend-2)

        plt.axis('off')
        file_name = plot_dir + '/regNetworks/fullNetwork.png'

        change_background_color(plt.gcf(), plt.gca(), background)

        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        print("Saving regulatory network to %s" % file_name)
        if show == True:
            plt.show()
        plt.close()

    else:

        if community == 'all':
            communities = set(partition.values())
            all_regulations = [G.nodes[node]['regulation'] for node in G.nodes]
            vmin = min(all_regulations) if min(all_regulations) < -1 else -1
            vmax = max(all_regulations) if max(all_regulations) > 1 else 1
        else:
            communities = [community]

        for comm in communities:

            # Create a colormap for the nodes based on their 'regulation' attribute
            cmap = plt.cm.coolwarm
            if community != 'all':
                # Adjust vmin and vmax for specific community
                nodes_in_community = [node for node, c in partition.items() if c == comm]
                regulations = [G.nodes[node]['regulation'] for node in nodes_in_community]
                vmin = min(regulations) if min(regulations) < -1 else -1
                vmax = max(regulations) if max(regulations) > 1 else 1
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

            # Create the original greyscale colormap for edges
            cmap_grey = plt.cm.Greys
            dark_grey_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "dark_grey", cmap_grey(np.linspace(0.3, 1, 256))
            )

            # Filter nodes by community using the partition dictionary
            nodes_in_community = [node for node, c in partition.items() if c == comm]
            G_sub = G.subgraph(nodes_in_community)

            # Calculate the layout of the graph
            pos = nx.kamada_kawai_layout(G_sub)

            # Prepare figure
            plt.figure(figsize=(12, 12))
            plt.title(f'Network Visualization for Community {comm}')

            # Node colors and sizes based on 'regulation' attribute
            node_colors = [cmap(norm(G_sub.nodes[node]['regulation'])) for node in G_sub.nodes]
            node_sizes = [1500 for node in G_sub.nodes]  # Example scaling

            # Normalize edge weights for width and alpha
            edge_weights_raw = np.array([G_sub.edges[edge]['weight'] for edge in G_sub.edges])
            edge_weights = edge_weights_raw * 4  # Adjust scaling as necessary
            edge_alphas = np.interp(edge_weights_raw, (edge_weights_raw.min(), edge_weights_raw.max()), (0.1, 1))

            # Draw the network
            nx.draw_networkx_edges(G_sub, pos, width=edge_weights, edge_color=edge_alphas, edge_cmap=dark_grey_cmap, alpha=0.5)
            nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes, node_color=node_colors, cmap=cmap)

            # Custom method to draw labels with outlines for readability
            for node, (x, y) in pos.items():
                text = node
                plt.text(x, y, text, fontsize=8, ha='center', va='center',
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.5))

            # Draw color reference rectangle in the upper right corner
            ax = plt.gca()
            # Create a colorbar as a legend
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
            cbar.set_label('Regulation')
            cbar.ax.invert_yaxis()  # Invert to match coolwarm orientation

            plt.axis('off')
            file_name = plot_dir + '/regNetworks/community_' + str(comm) + '.png'

            change_background_color(plt.gcf(), plt.gca(), background)

            plt.savefig(file_name, dpi=300, bbox_inches='tight')
            print("Saving regulatory networks to %s" %file_name)
            if show == True:
                plt.show()
            plt.close()