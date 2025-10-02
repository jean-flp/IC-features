# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 22:51:45 2025

@author: Emanu
"""

import pandas as pd

#essential = pd.read_csv("essential_genes.csv")

jes = pd.read_csv("STRING_man_resultado.csv")
new = pd.read_csv("man_calculateFeatures.tsv", sep = " ", dtype = {0: str})
tp = pd.read_csv("man_TP.tsv", sep = " ")
#info = pd.read_csv("man_ProteinInfo.txt", sep = "\t")

new = new.drop('Edge Clustering Coefficient', axis = 1)
new = new.drop('Subgrap Centrality', axis = 1)

#Sce = 6539
#Dme = 13578
#Cel = 18501
#Man = 11826
new = new.drop(new.index[11826:])

new = pd.merge(new, tp, on = 'Protein_key', how = 'inner')
all_features = pd.merge(new, jes, on = 'Protein_key', how = 'inner')

#all_features = pd.merge(all_features, info, left_on = 'Protein_key', right_on= '#string_protein_id')
#all_features['Protein_key'] = all_features['Protein_key'].astype(str)

#Sce = DEG2001
#Dme = DEG2007
#Cel = DEG2002
#essentialFiltered = essential[essential['Code_Organism'] == 'DEG2007'][['Gene', 'Locus']]

"""
def is_essential(protein_id):
    for Gene in essentialFiltered.Gene:
        Gene = str(Gene)
        if protein_id == Gene:
            return 1
    return 0

all_features['IsEssential'] = 0
all_features['IsEssential'] = all_features['preferred_name'].astype(str).apply(lambda x: is_essential(x))


essenciais = all_features[all_features['IsEssential'] == 1]

final = all_features.drop(['#string_protein_id', 'protein_size', 'annotation'], axis = 1)
"""
all_features.to_csv('man_PPIFeatures.tsv', sep = " ", index = False)


