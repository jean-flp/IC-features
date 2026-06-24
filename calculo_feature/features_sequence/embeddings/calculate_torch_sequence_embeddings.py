#%%
import sys
import torch
from model.strategy_model import ModelContext
from huggingface_hub import login
from dotenv import load_dotenv
import os
from Bio import SeqIO
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
print(project_root)

sys.path.append(project_root)
from calculo_feature.synthesizeUtils import train_test_between_files

__X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__ = train_test_between_files()
sys.path.remove(project_root)

import gc

load_dotenv()
__HF_KEY_TOKEN__ = os.getenv("HF_KEY_TOKEN")
__SEED__ = int(os.getenv("SEED"))
__TEST_SIZE__ = float(os.getenv("TESTE_SIZE"))

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
    os.path.join(calculo_feature_dir, "../../../data/processed/embeddings/sequence/")
)
print(path_data_processed)
path_data_raw = os.path.abspath(
    os.path.join(calculo_feature_dir, "../../../data/raw/string/fasta")
)
print(path_data_raw)
print(device)
#%%
models_name = {
    "esm":"facebook/esm2_t33_650M_UR50D",#Pytorch
    "prot":"Rostlab/prot_t5_xl_uniref50" #Pytorch
}
#Models 
models = {
    "esm":{"model":None,"tokenizer":None},
    "prot":{"model":None,"tokenizer":None}
}

def lerFastaBio(arquivo):
    arquivoFasta = SeqIO.parse(open(arquivo), 'fasta')

    dict_fasta = {}

    for i in arquivoFasta:
        dict_fasta[i.id] = str(i.seq)

    return dict_fasta
# calcular embeddings dividindo por bucket

#%%
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
pastas = os.listdir(path_data_raw)
context = ModelContext()

for k, model_name in models_name.items():

    with context.setStrategy(model_name) as model:

        print(f"\nModel: {model_name}")

        model.initializeModel(device=device, model_string=model_name)

        embeddings_list = []

        for pasta in pastas:
            file_name = os.listdir(f"{path_data_raw}/{pasta}")[0]
            file_path_to_process = os.path.join(f"{path_data_raw}\\{pasta}\\{file_name}")

            dict_seq = lerFastaBio(file_path_to_process)

            protein_ids = list(dict_seq.keys())
            sequences_raw = list(dict_seq.values())
            clean_ids = []
            clean_seqs = []
            VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

            for pid, seq in zip(protein_ids, sequences_raw):
                if not isinstance(seq, str):
                    continue

                seq = seq.upper()
                seq = re.sub(r"[^A-Z]", "", seq)

                if len(seq) < 5:
                    continue

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

            buckets = []

            for bucket_id in range(n_buckets):
                inicio = bucket_id * bucket_size
                fim = min((bucket_id + 1) * bucket_size, len(sorted_df))

                bucket_df = sorted_df.iloc[inicio:fim]

                if len(bucket_df) == 0:
                    continue

                buckets.append({
                    "data": bucket_df,
                    "max_len": bucket_df["sequence"].str.len().max()
                })
            batch_size = 16
            max_model_len = 1022  # limite seguro para ESM2 3B (evita explosão de memória)

            pbar_buckets = tqdm(
                buckets,
                desc=f"Processing {k}",
                dynamic_ncols=True,
                ascii=True,
                mininterval=0.1
            )
            for bucket_idx, bucket in enumerate(pbar_buckets):

                sequences_all = list(bucket["data"]["sequence"])
                protein_ids_all = list(bucket["data"]["protein_id"])

                max_len_bucket = int(bucket["max_len"])

                pbar_buckets.set_postfix({
                    "bucket": bucket_idx,
                    "n_seq": len(sequences_all),
                    "max_len": max_len_bucket
                })

                print(
                    f"Bucket {bucket_idx}: "
                    f"{len(sequences_all)} seqs | "
                    f"max_len={max_len_bucket}"
                )

                if max_len_bucket > 1000:
                    batch_size = 1
                elif max_len_bucket > 700:
                    batch_size = 2
                elif max_len_bucket > 400:
                    batch_size = 4
                else:
                    batch_size = 8

                batch_pbar = tqdm(
                    range(0, len(sequences_all), batch_size),
                    desc=f"Bucket {bucket_idx} batches",
                    leave=False,
                    dynamic_ncols=True,
                    ascii=True,
                    mininterval=0.1
                )

                for i in batch_pbar:

                    batch_seqs = sequences_all[i:i + batch_size]
                    batch_ids = protein_ids_all[i:i + batch_size]

                    batch_pbar.set_postfix({
                        "batch_size": len(batch_seqs)
                    })

                    model.tokenize(
                        sequences=batch_seqs,
                        padding=True,
                        max_length=min(bucket["max_len"], max_model_len)
                    )

                    embeddings = model.computeSentenceEmbeddings()

                    if torch.is_tensor(embeddings):
                        embeddings = embeddings.detach().cpu().numpy()

                    embedding_dim = embeddings.shape[1]

                    colunas_embedding = [
                        f"{k}_{i}"
                        for i in range(embedding_dim)
                    ]


                    df_temp = pd.DataFrame(embeddings,columns=colunas_embedding)
                    df_temp.insert(0, "protein_id", batch_ids)

                    embeddings_list.append(df_temp)
                    del embeddings

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
            

        df_embeddings_sentence = pd.concat(
            embeddings_list,
            ignore_index=True
        )

        df_man = df_embeddings_sentence[
            df_embeddings_sentence["protein_id"].apply(lambda x: x.split('.')[0]).isin({'6183'})
        ]

        df_mus = df_embeddings_sentence[
            df_embeddings_sentence["protein_id"].apply(lambda x: x.split('.')[0]).isin({'10090'})
        ]

        df_org_modelos = df_embeddings_sentence[
            ~df_embeddings_sentence["protein_id"].apply(lambda x: x.split('.')[0]).isin({'6183', '10090'})
        ]

        pca_man = df_man.drop(columns=["protein_id"]).values
        pca_mus = df_mus.drop(columns=["protein_id"]).values
        pca_org_modelos = df_org_modelos.drop(columns=["protein_id"]).values

        #Filter proteins
        X_train, X_test, y_train, y_test = (__X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__)

        data_to_fit = df_org_modelos[df_org_modelos["protein_id"].isin(X_train["protein_id"])]

        scaler = StandardScaler()
        scaler_fit = scaler.fit(data_to_fit.drop(columns=["protein_id"]).values)

        pca_org_modelos_stand = scaler_fit.transform(pca_org_modelos)
        pca_man_stand = scaler_fit.transform(pca_man)
        pca_mus_stand = scaler_fit.transform(pca_mus)

        pca_data_fit  = scaler_fit.transform(data_to_fit.drop(columns=["protein_id"]).values) 

        pca_calc = PCA(n_components=0.95, random_state=__SEED__)
        pca_calc_fit = pca_calc.fit(pca_data_fit)

        pca_fin_org = pca_calc_fit.transform(pca_org_modelos_stand)
        pca_fin_man = pca_calc_fit.transform(pca_man_stand)
        pca_fin_mus = pca_calc_fit.transform(pca_mus_stand)

        #renomeando as colunas
        n_components_final = pca_fin_org.shape[1]

        colunas_pca = [
            f"{k}_pca_{i}"
            for i in range(n_components_final)
        ]

        df_pos_pca_org = pd.DataFrame(pca_fin_org, columns=colunas_pca)
        df_pos_pca_man = pd.DataFrame(pca_fin_man, columns=colunas_pca)
        df_pos_pca_mus = pd.DataFrame(pca_fin_mus, columns=colunas_pca)

        df_pos_pca_org.insert(0, "protein_id", list(df_org_modelos["protein_id"]))
        df_pos_pca_man.insert(0, "protein_id", list(df_man["protein_id"]))
        df_pos_pca_mus.insert(0, "protein_id", list(df_mus["protein_id"]))

        path_pasta_model = os.path.join(path_data_processed,k)
        # saves
        df_pos_pca_org.to_csv(os.path.join(path_pasta_model, f"all.global.embedding.pca.{k}.tsv"), sep=' ', index=False)
        df_pos_pca_man.to_csv(os.path.join(path_pasta_model, f"6183.global.embedding.pca.{k}.tsv"), sep=' ', index=False)
        df_pos_pca_mus.to_csv(os.path.join(path_pasta_model, f"10090.global.embedding.pca.{k}.tsv"), sep=' ', index=False)
        df_embeddings_sentence.to_csv(os.path.join(path_pasta_model, f"global.embedding.{k}.tsv"), sep=' ', index=False)
