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

# Wrapper function
def dp_GSEA(gene_list, reg_list, sig_list, iterations=1000,
            seed=None, matchProfile=False,
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
        drug_gene_reg = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'single_drug_perturbations.tsv'), sep="\t", index_col=0)

        if source == 'validated':
            drug_gene_reg = drug_gene_reg[~drug_gene_reg['drug'].str.startswith('drug:P')]

        print('File for drug gene associations defaults the single drug perturbations'
              ' provided by Wang et al., 2016. Pelase cite https://doi.org/10.1038/ncomms12846')
        if drug_ref == None:
            # set corresponding drug_ref
            drug_ref = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'drug_ref.tsv'), sep="\t", index_col=0)


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


def tablePreprocessing(gene_list, reg_list, sig_list, drug_gene_reg):

    genedict = {
        'gene': gene_list,
        'reg': reg_list,
        'sig': sig_list,
    }

    geneTable = pd.DataFrame(genedict)

    # assumed to have the following columns: drug, gene, drug_up
    drugTable = drug_gene_reg

    # print col names of finished drug and gene tables - both contain gene to match on?
    print(geneTable.columns, drugTable.columns)

    # Merges the columns on genes based on the drugRef, remove any NAs generated from merge and ranks by sig
    rankTable = pd.merge(drugTable, geneTable, on='gene', how='left')
    print(rankTable)

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
        iterations = self.iterations
        try:
            resampleNum = len(self.getDrugIndexes(drug))
            if resampleNum != 0:
                return np.array(
                    [np.random.choice(self.indexLen, resampleNum, replace=False) for _ in range(iterations)])
            else:
                return None
        except:
            return None

    def getMaxDeviations(self, index, getTable=False):
        if index is not None:
            if len(index.shape) == 1:
                # Assigns variable to instance rank table
                rankTable = self.rankTable

                # Finds total sum of for brownian bridge
                totalSum = sum(rankTable.sig)

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
                # Assigns variable to instance rank table
                rankTable = self.rankTable
                iterations = self.iterations
                # Iterate through all indexes
                totalSum = sum(rankTable.sig)

                maxDeviationList = np.array([])
                maxDeviationIndexList = np.array([])

                for i in index:
                    # Calculates the total sum for hits
                    hitSum = sum(rankTable.sig[n] for n in i)

                    # Negative step for "misses" weighted by the T-statistic
                    rankTable['step'] = -1 * rankTable.sig / (totalSum - hitSum)

                    # Calculates the "hit" steps (the comprehension loop will save time on smaller group sizes)
                    rankTable.loc[i, 'step'] = [rankTable.sig[n] / hitSum for n in
                                                i]  # faster for shorter <200 lists

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

    def resultsTable(self):
        drugList = self.drugList()
        drugListLen = len(drugList)
        iterations = self.iterations

        resultsTable = pd.DataFrame(columns=['drug', 'ES', 'NES', 'ES_p', 'TCS', 'NTCS', 'TCS_p', 'genes'])
        nullDistDict = {}
        nullNESDist = []
        nullNTCSDist = []

        drugCounter = 1

        # Initialize an empty list to collect DataFrames or Series to append
        rows_to_append = []
        for drug in drugList:
            print('DxCL: ' + str(drugCounter) + ' of ' + str(drugListLen) + ', ' + drug)
            drugCounter += 1
            drugIndex = self.getDrugIndexes(drug)

            if drugIndex is not None:
                drugMaxDev = self.getMaxDeviations(drugIndex)
                nullKey = len(drugIndex)

                if nullKey in nullDistDict.keys():
                    nullMaxDev = nullDistDict[nullKey]

                else:
                    nullIndex = self.getNullIndexes(drug)
                    nullMaxDev = self.getMaxDeviations(nullIndex)
                    nullDistDict[nullKey] = nullMaxDev

                drugStats = self.getStats(drug, drugIndex=drugIndex, drugMaxDev=drugMaxDev, nullIndex=nullIndex,
                                          nullMaxDev=nullMaxDev)
                # Add drugStats to the list
                rows_to_append.append(pd.DataFrame(drugStats))

                # Assuming nullMaxDev is a dictionary or DataFrame you're working with in your loop
                nullNESDist = nullNESDist + list(nullMaxDev['maxDeviationNorm'])
                nullNTCSDist = nullNTCSDist + list(nullMaxDev['maxDeviationIndexNorm'])

        # Filter out all-NA columns from each DataFrame in rows_to_append before concatenation
        rows_to_append_filtered = [df.dropna(axis=1, how='all') for df in rows_to_append]

        # Now concatenate resultsTable with the filtered list
        resultsTable = pd.concat([resultsTable] + rows_to_append_filtered, ignore_index=True)


        print('Calculating FDRs...')

        # Calculate the FDR for NES and NTCS based on the null distributions
        for score_type, null_dist in [('NES', nullNESDist), ('NTCS', nullNTCSDist)]:
            fdr_column = f'{score_type}_FDR'

            # Calculate the FDR values
            # For each observed score, calculate the proportion of null distribution scores that are greater or equal
            resultsTable[fdr_column] = resultsTable[score_type].apply(
                lambda x: np.mean([null_score >= x for null_score in null_dist])
            )

        # After this, values in NES_FDR and NTCS_FDR columns represent the FDR for each drug's NES and NTCS, respectively.
        # Scores resulting in FDR values below 0.05 can be considered significant.

        return resultsTable

