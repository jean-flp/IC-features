#%%
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from model.strategy_model import ModelContext
from huggingface_hub import login
from dotenv import load_dotenv
import os
from Bio import SeqIO
import re
import numpy as np
import pandas as pd

load_dotenv()
__HF_KEY_TOKEN__ = os.getenv("HF_KEY_TOKEN")
#print(__HF_KEY_TOKEN__)

login(__HF_KEY_TOKEN__)

#%%
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)

try:
    torch.cuda.init()
    print("CUDA inicializada com sucesso")
except Exception as e:
    print("ERRO:", repr(e))

#%%
#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# models
#"bert":"GrimSqueaker/proteinBERT"protein bert: https://academic.oup.com/bioinformatics/article/38/8/2102/6502274// KERAS
#ProteinBert - DEVERÁ SER EXECUTADO EM SEPARADO PELO ARQUIVO "ProteinBert.py" pois precisa de uma versão diferente de python para executá-lo !!!!!
calculo_feature_dir = os.getcwd()
print(calculo_feature_dir)

path_data_processed = os.path.abspath(
    os.path.join(calculo_feature_dir, "../../../data/processed/embeddings/sequence/proteinbert")
)
print(path_data_processed)
path_data_raw = os.path.abspath(
    os.path.join(calculo_feature_dir, "../../../data/raw/string/fasta")
)
print(path_data_raw)
print(device)
#%%
models_name = {
    "esm":"facebook/esm2_t36_3B_UR50D",#Pytorch
    "prot":"Rostlab/prot_t5_xl_uniref50" #Pytorch
}
#Models 
models = {
    "esm":{"model":None,"tokenizer":None},
    "prot":{"model":None,"tokenizer":None}
}

#%%
for k, model_name in models_name.items():
    context = ModelContext()
    context.setStrategy(model_name)
    context.initializeModel(device, model_name)
    models[k]["model"] = context

#%%
def lerFastaBio(arquivo):
    arquivoFasta = SeqIO.parse(open(arquivo), 'fasta')

    dict_fasta = {}

    for i in arquivoFasta:
        dict_fasta[i.id] = str(i.seq)

    return dict_fasta
# calcular embeddings dividindo por bucket

pastas = os.listdir(path_data_raw)

for pasta in pastas:

    file_name = os.listdir(f"{path_data_raw}/{pasta}")[0]
    file_path_to_process = os.path.join(f"{path_data_raw}\\{pasta}\\{file_name}")

    print(pasta)
    print(file_name)
    print(file_path_to_process)

    dict_seq = lerFastaBio(file_path_to_process)

    protein_ids = list(dict_seq.keys())
    sequences_raw = list(dict_seq.values())

    # limpeza robusta
    clean_ids = []
    clean_seqs = []
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
    for pid, seq in zip(protein_ids, sequences_raw):

        if not isinstance(seq, str):
            continue

        seq = seq.upper()
        seq = re.sub(r"[^A-Z]", "", seq)

        # remove vazias ou muito pequenas
        if len(seq) < 5:
            continue

        # remove aminoácidos inválidos
        if not set(seq).issubset(VALID_AA):
            continue

        clean_ids.append(pid)
        clean_seqs.append(seq)

    df_sequencias = pd.DataFrame({
        "protein_id": clean_ids,
        "sequence": clean_seqs
    })

    sorted_df = df_sequencias.sort_values(
        by="sequence",
        key=lambda x: x.str.len(),
        ignore_index=True
    )

    n_buckets = 32
    bucket_size = int(np.ceil(len(sorted_df) / n_buckets))

    buckets = {}