# -*- coding: utf-8 -*-
"""
Created on Sun May 18 17:08:22 2025

@author: Emanu
"""
#%%
import pandas as pd

#%%
organismo = 'sce'
PPI = f'Resultados_Contexto/PPI_{organismo}.tsv'
emboss = f'Resultados_Emboss/Emboss_{organismo}.tsv'
sequencia = f'Resultados_Sequencia/Sequence_{organismo}.tsv'

#%%
dfPPI = pd.read_csv(PPI, sep = ' ')
dfEmboss = pd.read_csv(emboss, sep = ' ')
dfSequencia = pd.read_csv(sequencia, sep = ' ')

#%%
dfSeq = pd.merge(dfSequencia, dfEmboss, left_on = 'Locus', right_on = 'GeneID')
dfAll = pd.merge(dfSeq, dfPPI, left_on = 'Locus', right_on = 'Protein_key', how = 'inner')
dfAll = dfAll.drop(['GeneID', 'Protein_key',  'preferred_name'], axis = 1)
dfAll.to_csv(f'Todas_Features/Features_{organismo}.csv', index = False)


