# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 12:24:50 2025

@author: Emanu
"""

import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import re
import time

def lerFastaBio(arquivo):
    arquivoFasta = SeqIO.parse(open(arquivo),'fasta') #lê o arquivo com o Biopython
    
    dict_fasta = {} 

    for i in arquivoFasta:
        dict_fasta[i.id] = str(i.seq) 

    return dict_fasta

def run_pepstats(sequence, email='blazingflu@email.com', title='elegans'):
    url = 'https://www.ebi.ac.uk/Tools/services/rest/emboss_pepstats/run'
    headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Accept': 'text/plain'}
    data = {
        'email': email,
        'title': title,
        'sequence': sequence,
        'termini': 'true',
        'mono': 'true'
    }
    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        return response.text.strip()
    else:
        raise Exception(f"Failed to submit job: {response.status_code} {response.text}")

def check_status(job_id):
    url = f'https://www.ebi.ac.uk/Tools/services/rest/emboss_pepstats/status/{job_id}'
    headers = {'Accept': 'text/plain'}
    response = requests.get(url, headers=headers)
    return response.text.strip()

def fetch_result(job_id):
    url = f'https://www.ebi.ac.uk/Tools/services/rest/emboss_pepstats/result/{job_id}/out'
    headers = {'Accept': 'text/plain'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch result: {response.status_code} {response.text}")
        
def parse_pepstats_result(text):
    
    chunks = text.split("PEPSTATS of")[1:]
    parsed_data = []
    
    for chunk in chunks:
        lines = chunk.strip().splitlines()
        
        first_line = lines[0]
        gene_id = first_line.split()[0]
        
        general_props = {}
        pI_match = re.search(r'Isoelectric Point = ([\d\.]+)', chunk)
    
        if pI_match:
            general_props['IsoelectricPoint'] = float(pI_match.group(1))
    
        property_groups = {}
        prop_section = re.search(r'Property\s+Residues\s+Number\s+Mole%\n((?:.+\n)+)', chunk)
        if prop_section:
            for line in prop_section.group(1).splitlines():
                line = line.strip()
                if line and '=' not in line:
                    match = re.match(r'(.+?)\s+(\d+)\s+([\d\.]+)', line)
                    if match:
                        property_name = match.group(1).split()[0]
                        number = int(match.group(2))
                        property_groups[f"{property_name}_Number"] = number

        entry = {'GeneID': gene_id, **general_props, **property_groups}
        
        parsed_data.append(entry)
    df = pd.DataFrame(parsed_data)
    return df


def process_sequences(sequence_list):
    """
    sequence_list: list with sequences
    """
    all_dataframes = []  # list to hold individual dataframes
    i = 0
    for seq in sequence_list:
        print(f"Processing chunk {i}")
        i += 1
        try:
            job_id = run_pepstats(seq)
            time.sleep(2)  # short pause before checking

            # Wait for job completion
            status = check_status(job_id)
            while status not in ["FINISHED", "ERROR", "FAILURE"]:
                time.sleep(3)
                status = check_status(job_id)

            if status == "FINISHED":
                result_text = fetch_result(job_id)
                df = parse_pepstats_result(result_text)  # Parse ONLY the current result
                all_dataframes.append(df)
                
            else:
                print("Job failed for one sequence.")

        except Exception as e:
            print(f"Error processing: {e}")

    final_df = pd.concat(all_dataframes, ignore_index=True)
    return final_df


def sequencias500(seq_dict):
    emboss_strings = []
    current_chunk = ""
    count = 0
    
    for gene_id, seq in seq_dict.items():
        current_chunk += f">{gene_id}\n{seq}\n"
        
        count += 1
        if count % 500 == 0:
            emboss_strings.append(current_chunk)
            current_chunk = ""
    if current_chunk:
        emboss_strings.append(current_chunk)
    return emboss_strings

organismo = 'mus'
file_name = f'proteomas/proteoma_{organismo}.fa'
dict_seq = lerFastaBio(file_name)
chunks = sequencias500(dict_seq)

final = process_sequences(chunks)
final.to_csv(f'Resultados_Emboss/Emboss_{organismo}.tsv', sep = ' ', index = False)

