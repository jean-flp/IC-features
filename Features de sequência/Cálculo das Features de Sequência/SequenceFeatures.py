# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 11:41:28 2025

@author: Emanu
"""

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def lerFastaBio(arquivo):
    arquivoFasta = SeqIO.parse(open(arquivo),'fasta') #lê o arquivo com o Biopython
    
    dict_fasta = {} 

    for i in arquivoFasta:
        dict_fasta[i.id] = str(i.seq) 

    return dict_fasta

def aromaticity(protein):
    
    '''
    Calculate the aromaticity according to Lobry, 1994.

    Calculates the aromaticity value of a protein according to Lobry, 1994.
    It is simply the relative frequency of Phe+Trp+Tyr.
    '''
    
    X = ProteinAnalysis(protein)
    
    return X.aromaticity()



# Reescrita da função do Biopython, dado uma discrepância de trabalhos citados
def secondary_structure_fraction(protein):
    
    '''
    Amino acids in helix: A, E, L, M, Q, R.
    Amino acids in Turn: D, G, N, K, P, S
    Amino acids in sheet: C, F, H, I, T, W, V, Y

    Returns a tuple of three floats (Helix, Turn, Sheet)
    
    These are beta sheets, alpha helixes, and turns (where the residues change direction).

    '''
    X = ProteinAnalysis(protein)
    
    aa_percentages = X.get_amino_acids_percent()
    
    helix = sum(aa_percentages[r] for r in "AELMQR")
    
    turn = sum(aa_percentages[r] for r in "DGNKPS")
    
    sheet = sum(aa_percentages[r] for r in "CFHITWVY")
    
    return helix, turn, sheet
    


def get_amino_acids_percent(protein, param):
    # Calculate the amino acid content in percentages.
    
    X = ProteinAnalysis(protein)
    
    return X.get_amino_acids_percent()[param]


def count_amino_acids(protein, param):
    # Calculate Count standard amino acids, return a dict.
    
    X = ProteinAnalysis(protein)
    
    return X.count_amino_acids()[param]

aminoacids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']

organismo = 'dme'
file = f'proteomas/proteoma_{organismo}.fa'

proteoma = lerFastaBio(file)

df_bio = pd.DataFrame(list(proteoma.items()), columns=['Locus', 'Sequence'])

df_bio['Sequence_Length'] = df_bio.apply(lambda x: len(x.Sequence), axis=1)

df_bio['Aromaticity'] = df_bio.apply(lambda x: aromaticity(x.Sequence), axis=1)

df_bio['Sec_Struct_Helix'], df_bio['Sec_Struct_Turn'], df_bio['Sec_Struct_Sheet'] = zip(*df_bio.apply(
    lambda x: secondary_structure_fraction(x.Sequence), axis=1))

for am in aminoacids:
    df_bio['Percent_'+am] = df_bio.apply(lambda x: get_amino_acids_percent(x.Sequence, am), axis=1)
    
df_bio.to_csv(f'Resultados_Sequencia/Sequence_{organismo}.tsv', sep = ' ', index = False)