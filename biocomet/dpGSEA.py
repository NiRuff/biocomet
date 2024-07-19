# todo:
# map gene symbol names - do this before adding data to /data
# also make it long format
# add data to /data
# implement dpGSEA
# report as is and also opposite regulation direction with
# commenting on assumed directionality (ctrl/condition and vice versa)
# run this code and also the original python script to compare results !!!

# maybe give option to plot the subnetworks combined with these reults to also mark the driver genes here

import pandas as pd
import numpy as np
import random
import os
import math
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import json
from collections import defaultdict
import gzip

'''
A drug-gene target enrichment technique utilizing a modified GSEA approach. 

dpGSEA enriches on a proto-matrix based on a user-defined cutoff (these matrices need to be built by the user or the user can utilize the ones included with the script). This proto-matrix summarizes drug-gene perturbation profiles in a format iterable by this enrichment. The output includes three different measures along with the "leading-edge" genes as a .csv output file.

- Enrichment score (ES) - this score is interpreted the same way the standard GSEA enrichment score. It reflects the degree to which a complimentary or matching drug gene profile is overrepresented at the top of a ranked list.
- Enrichment score p-value (ES_pvalue) - the statistical significance of the enrichment score for a single drug gene set.
- Target compatibility p-value (TC_pvalue) - a p-value reflecting the quantity and magnitude of statistical significance of differentially expressed genes that 
  match or antagonize a drug profile. This statistical test compares the modulation of the leading edge genes against random modulation.
- Driver Genes aka leading edge genes (driver_genes) - this lists genes that appear in the ranked list at or before the point at which the running sum reaches its maximum deviation from zero. These genes are often interpreted as the genes driving an enrichment or modulation of drug-gene and differential expression analysis.
 
 inspired by dpGSEA 0.9.4 (https://github.com/sxf296/drug_targeting/tree/master)

'''

# todo: when computing ES - also compute -ES by label switch to report both: sig for case/ctrl and ctrl/case
# same goes for TCS
# check Fig 2 here:https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03929-0#Sec3

# also validate mayaan, chembl and dpGSEA sets by appyling to table 1 data to see if the suspected drugs pop up

def reduceInput(drug_df, gene_list, organism, source, min_genes=1):
    dict_path = os.path.join(os.path.dirname(__file__), 'data', 'drug_gene_mapping_corrected.json.gz')

    # Load the dictionary from the JSON file
    with gzip.open(dict_path, 'rt', encoding='utf-8') as fp:
        loaded_dict = json.load(fp)

    # Prepare keys to check in the dictionary
    keys_to_check = [f"{organism}_{source}"]
    if f"{organism}_{source}" not in loaded_dict:
        print(f"{keys_to_check} not found. Using all organisms and sources available.")
        keys_to_check = list(loaded_dict.keys())  # Use all keys if specific one is not found

    print("Keys to check:", keys_to_check)

    # Initialize a dictionary to collect drug IDs with a set to track unique genes
    drug_id_to_genes = defaultdict(set)
    for key in keys_to_check:
        for gene in gene_list:
            if gene in loaded_dict[key]:
                for drug_id in loaded_dict[key][gene]:
                    drug_id_to_genes[drug_id].add(gene)  # Add gene to the set of unique genes for this drug_id

    # Now, filter drug IDs by min_genes, checking the length of the set of genes
    filtered_drug_ids = [drug_id for drug_id, genes in drug_id_to_genes.items() if len(genes) >= min_genes]
    #print("Filtered Drug IDs based on unique gene count:", filtered_drug_ids)

    #print([(drug_id, genes) for drug_id, genes in drug_id_to_genes.items() if len(genes) >= min_genes])

    # Subset the drug_df for only those entries that show any of these IDs in the 'drug' column
    reduced_drug_df = drug_df[drug_df['drug'].isin(filtered_drug_ids)]

    return reduced_drug_df


# Wrapper function
def dp_GSEA(gene_list, reg_list, sig_list, iterations=1000,
            seed=None, matchProfile=False, organism=9606, min_genes=1,
            drug_gene_reg=None, drug_ref=None, FDR_treshold=0.05,
            low_values_significant=None, source='validated'):

    print('dpGSEA was first implemented by Fang et al., 2021.'
          ' Please cite https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-020-03929-0')

    if low_values_significant is None:
        # Modify sig_list if necessary. If interpreted as p, lower is better. Otherwise, higher is better
        if all(x <= 1 for x in sig_list):
            print('All values are <= 1 in sig_list. Assuming sig_list to be p_values or similar confidence intervals.'
                  ' Low values will be assumed connected to high significance.')
            low_values_significant = True

    if low_values_significant:
        # Find the minimum positive value greater than 0 in sig_list
        min_positive = min(x for x in sig_list if x > 0)
        min_positive_log = -math.log10(min_positive)  # Compute its negative logarithm

        # Update sig_list, applying -math.log10(x) or assigning the log of the min positive value for x <= 0
        sig_list = [-math.log10(x) if x > 0 else min_positive_log for x in sig_list]

    # check if drug_gene_reg is == None, if yes, drug_gene_reg is defaulted to single-drug-perturbations of mayaanlab
    if drug_gene_reg == None:

        print('File for drug gene associations defaults to the single drug perturbations'
              ' provided by Wang et al., 2016. Pelase cite https://doi.org/10.1038/ncomms12846')

        drug_gene_reg = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'single_drug_perturbations.tsv.gz'),
                                    sep="\t", index_col=0, compression='gzip')

        print('Initial number of drug gene associations: ', drug_gene_reg.shape[0])

        if source == 'validated':
            drug_gene_reg = drug_gene_reg[~drug_gene_reg['drug'].str.startswith('drug:P')]
            print('Filtering for source. Remaining number of drug gene associations: ', drug_gene_reg.shape[0])
        elif source == 'predicted':
            drug_gene_reg = drug_gene_reg[drug_gene_reg['drug'].str.startswith('drug:P')]
            print('Filtering for source. Remaining number of drug gene associations: ', drug_gene_reg.shape[0])
        else:
            print('source parameter invalid or not set. Keeping all entries - validated and predicted ones')

        if drug_ref == None:
            # set corresponding drug_ref
            drug_ref = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'drug_ref.tsv'), sep="\t", index_col=0)

        if (organism == 9606) or (organism == '9606'):
            organism = 'human'
        elif (organism == 10092) or (organism == '10092'):
            organism = 'mouse'
        elif (organism == 10116) or (organism == '10116'):
            organism = 'rat'

        if organism != None:
            org_drug_ids = drug_ref[drug_ref['organism']==organism].id
            drug_gene_reg = drug_gene_reg[drug_gene_reg['drug'].isin(org_drug_ids)]
            print('Filtering for organism. Remaining number of drug gene associations: ', drug_gene_reg.shape[0])

        # subsample to exclude all drug that are not related to at least x genes
        drug_gene_reg = reduceInput(drug_gene_reg, gene_list, organism, source, min_genes=min_genes)
        print('Filtering out drugs with less than'
              ' ' + str(min_genes) + ' associated genes. '
                                     'Remaining number of drug gene associations: ', drug_gene_reg.shape[0])

    # somehow check output file name or otherwise return in the end the results and set them as attributes for the PPI object
    outputFileName = './results.tsv'

    # Perform the processing and merging of the tables to create the rank table
    rankTable = tablePreprocessing(gene_list, reg_list, sig_list, drug_gene_reg)

    dt = dpGSEA(rankTable, iterations=iterations, seed=seed, matchProfile=matchProfile,)
    results = dt.resultsTable()

    # Filter for drugs with NES_FDR and NTCS_FDR less than the threshold
    significant_drugs = results[
        (results['NES_FDR'] < FDR_treshold) & (results['NTCS_FDR'] < FDR_treshold)]

    # Print the significant drugs
    print("Significant drugs according to NES and NTCS metrics:")
    print(significant_drugs[['drug', 'NES_FDR', 'NTCS_FDR']])

    # also print content of drug_reg for sig. results:
    print(drug_ref[drug_ref['id'].isin(significant_drugs.drug)])

    print('Writing results...')
    results.to_csv(path_or_buf=outputFileName, index=False, sep='\t')

    return results


def tablePreprocessing(gene_list, reg_list, sig_list, drug_gene_reg):

    genedict = {
        'gene': gene_list,
        'reg': reg_list,
        'sig': sig_list,
    }

    geneTable = pd.DataFrame(genedict)

    # assumed to have the following columns: drug, gene, drug_up
    drugTable = drug_gene_reg

    # Merges the columns on genes based on the drugRef, remove any NAs generated from merge and ranks by sig
    rankTable = pd.merge(drugTable, geneTable, on='gene', how='left')

    rankTable = rankTable[rankTable['reg'].notna()]
    rankTable = rankTable.sort_values(by='sig', ascending=False).reset_index()

    # Determines the gene direction based on the FC of the DE, 1 is up-regulated while 0 is down-regulated
    rankTable.loc[rankTable.reg > 0, 'reg_up'] = 1
    rankTable.loc[rankTable.reg < 0, 'reg_up'] = 0

    # Remove signal weaker than can be represented by float
    rankTable = rankTable[rankTable.sig != 0]

    # Assign as ints for faster comp later on
    rankTable.drug_up = rankTable.drug_up.astype(int)
    rankTable.reg_up = rankTable.reg_up.astype(int)
    rankTable['sig'] = rankTable.sig.round(6)

    return rankTable



# This class performs the enrichment itself, multiple instances of the class is called when performing multi-processing
class dpGSEA:

    def __init__(self, rankTable, iterations, seed, matchProfile=False):
        random.seed(seed)
        self.rankTable = rankTable
        self.indexLen = len(self.rankTable)
        self.iterations = iterations
        self.matchProfile = matchProfile

    def drugList(self):
        return self.rankTable['drug'].unique()

    def getDrugIndexes(self, drug):
        rankTable = self.rankTable
        matchProfile = self.matchProfile

        if matchProfile:
            ind = rankTable[(rankTable.reg_up == rankTable.drug_up) & (rankTable.drug == drug)].index
        else:
            ind = rankTable[(rankTable.reg_up != rankTable.drug_up) & (rankTable.drug == drug)].index

        if ind.size != 0:
            return np.asarray(ind)
        else:
            return None

    def getNullIndexes(self, drug):
        try:
            resampleNum = len(self.getDrugIndexes(drug))
            if resampleNum == 0:
                return None

            # Pre-allocate the array for performance
            result = np.empty((self.iterations, resampleNum), dtype=int)

            # Generate samples for each iteration
            for i in range(self.iterations):
                result[i] = np.random.choice(self.indexLen, resampleNum, replace=False)

            return result
        except Exception as e:
            print(f"Error encountered: {e}")
            return None


    def getMaxDeviations(self, index, getTable=False):

        np.set_printoptions(threshold=10)

        if index is None:
            raise TypeError('Provided index is None')

        else:
            # Assigns variable to instance rank table
            rankTable = self.rankTable

            # Finds total sum of for brownian bridge
            totalSum = sum(rankTable.sig)

            if len(index.shape) == 1:

                # Calculates the total sum for hits
                hitSum = sum(rankTable.sig[index])

                # Negative step for "misses" weighted by the T-statistic
                rankTable['step'] = -1 * rankTable.sig / (totalSum - hitSum)

                # Calculates the "hit" steps (the comprehension loop will save time on smaller group sizes)
                rankTable.loc[index, 'step'] = [rankTable.sig[n] / hitSum for n in index]

                # Calculates the cumulative sum for the brownian bridge
                rankTable['cumsum'] = np.cumsum(rankTable.step)

                # Calculates cumulative sum and finds max deviation and index
                maxDeviation = max(rankTable['cumsum'])
                maxDeviationIndex = float(rankTable['cumsum'].idxmax())

                if getTable:
                    return rankTable

                else:
                    return {'maxDeviation': maxDeviation,
                            'maxDeviationIndex': 1 - (maxDeviationIndex / self.indexLen)}

            else:

                # non loop approach
                maxDeviationList = np.array([])
                maxDeviationIndexList = np.array([])

                # Add a column for group IDs, initializing with -1
                rankTable['group_id'] = -1

                # Assign group IDs based on 'index'
                for group_id, indices in enumerate(index):
                    rankTable.loc[indices, 'group_id'] = group_id

                # Calculate 'hitSum' for each group
                grouped = rankTable.groupby('group_id')
                rankTable['hitSum'] = grouped['sig'].transform('sum')

                # Iterate over each group to update steps and calculate max deviations
                for group_id, group_data in rankTable.groupby('group_id'):
                    if group_id == -1:  # Skip if group_id was not assigned
                        continue

                    # Calculates the total sum for hits
                    hitSum = sum(group_data['sig'])

                    # Calculate "miss" step values within this group context
                    rankTable.loc[:, 'step'] = -1 * rankTable.loc[:, 'sig'] / (
                                totalSum - hitSum)

                    # Update "hit" steps for this group
                    rankTable.loc[group_data.index, 'step'] = group_data['sig'] / hitSum

                    # Calculates cumulative sum and finds max deviation and index
                    cumSum = np.cumsum(rankTable.step)
                    maxDeviation = max(cumSum)
                    maxDeviationIndex = float(cumSum.idxmax())

                    # Adds to the max deviations and index of max deviations arrays
                    maxDeviationList = np.append(maxDeviationList, maxDeviation)
                    maxDeviationIndexList = np.append(maxDeviationIndexList, maxDeviationIndex)

                maxDeviationListNorm = maxDeviationList / np.mean(maxDeviationList)
                maxDeviationIndexList = 1 - (maxDeviationIndexList / self.indexLen)
                maxDeviationIndexListNorm = maxDeviationIndexList / np.mean(maxDeviationIndexList)

                return {'maxDeviation': maxDeviationList,
                        'maxDeviationNorm': maxDeviationListNorm,
                        'maxDeviationIndex': maxDeviationIndexList,
                        'maxDeviationIndexNorm': maxDeviationIndexListNorm}

    def getStats(self, drug, drugIndex=None, drugMaxDev=None, nullIndex=None, nullMaxDev=None):
        if drugIndex is None and drugMaxDev is None:
            drugIndex = self.getDrugIndexes(drug)
            drugMaxDev = self.getMaxDeviations(drugIndex)

        if nullIndex is None and nullMaxDev is None:
            nullIndex = self.getNullIndexes(drug)
            nullMaxDev = self.getMaxDeviations(nullIndex)

        iterations = self.iterations + 0.
        rankTable = self.rankTable

        # Sets the enrichment score, enrichment score p, and target compatibility score
        es = drugMaxDev['maxDeviation']
        nes = es / np.mean(nullMaxDev['maxDeviation'])
        esp = sum(nullMaxDev['maxDeviation'] > drugMaxDev['maxDeviation']) / iterations

        tcs = drugMaxDev['maxDeviationIndex']
        ntcs = tcs / np.mean(nullMaxDev['maxDeviationIndex'])
        tcp = sum(nullMaxDev['maxDeviationIndex'] > drugMaxDev['maxDeviationIndex']) / iterations

        # Finds leading edge genes
        driverGeneIndexes = drugIndex[drugIndex <= (-(drugMaxDev['maxDeviationIndex'] - 1) * self.indexLen) + 0.5]
        genes = list(rankTable.loc[driverGeneIndexes, 'gene'])

        # Returns dict of results
        return {'drug': drug,
                'ES': es,
                'NES': nes,
                'ES_p': esp,
                'TCS': tcs,
                'NTCS': ntcs,
                'TCS_p': tcp,
                'genes': genes}

    def process_drug(self, drug):
        # This method encapsulates the processing of a single drug
        # Adapted to work with parallel execution
        drugIndex = self.getDrugIndexes(drug)

        if drugIndex is None or len(drugIndex) == 0:
            return None  # Indicate that this drug was skipped

        drugMaxDev = self.getMaxDeviations(drugIndex)
        nullKey = len(drugIndex)

        if nullKey not in self.nullDistDict:
            nullIndex = self.getNullIndexes(drug)
            nullMaxDev = self.getMaxDeviations(nullIndex)
            self.nullDistDict[nullKey] = nullMaxDev
        else:
            nullMaxDev = self.nullDistDict[nullKey]
            nullIndex = None  # Or appropriate dummy value

        stats = self.getStats(drug, drugIndex=drugIndex, drugMaxDev=drugMaxDev,
                              nullIndex=nullIndex, nullMaxDev=nullMaxDev)
        return stats, nullMaxDev

    def resultsTable(self):
        drugList = self.drugList()
        self.nullDistDict = {}  # Shared state, accessed read-only in parallel tasks

        max_workers = os.cpu_count() // 3  # Use half of the available CPUs
        print(f"Parallelization started. Using up to {max_workers} processes.")

        results = []
        nullNESDist = []
        nullNTCSDist = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map each drug to a future object
            future_to_drug = {executor.submit(self.process_drug, drug): drug for drug in drugList}

            completed = 0
            for future in as_completed(future_to_drug):
                drug = future_to_drug[future]
                try:
                    result = future.result()
                    if result:
                        drugStats, nullMaxDev = result
                        results.append(drugStats)
                        nullNESDist.extend(list(nullMaxDev['maxDeviationNorm']))
                        nullNTCSDist.extend(list(nullMaxDev['maxDeviationIndexNorm']))
                except Exception as exc:
                    print(f"{drug} generated an exception: {exc}")
                finally:
                    completed += 1
                    print(f"Completed {completed}/{len(drugList)} drugs")

        # Construct the results table from the accumulated results
        resultsTable = pd.DataFrame(results)

        resultsTable = resultsTable.dropna(axis=1, how='all')  # Filter out all-NA columns if needed

        # Inspect the structure of the filtered DataFrame
        print("Shape of the results table:", resultsTable.shape)
        print("First few rows of the results table:", resultsTable.head())

        print('Calculating FDRs...')

        # Calculate FDRs similarly to your existing logic
        for score_type, null_dist in [('NES', nullNESDist), ('NTCS', nullNTCSDist)]:
            fdr_column = f'{score_type}_FDR'
            resultsTable[fdr_column] = resultsTable[score_type].apply(
                lambda x: np.mean([null_score >= x for null_score in null_dist])
            )

        return resultsTable

