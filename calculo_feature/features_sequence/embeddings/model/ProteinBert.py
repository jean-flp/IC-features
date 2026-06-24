##################################################################################################
# Necessário python 3.8.10
##################################################################################################
#%%
from __future__ import annotations
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import (
    get_model_with_hidden_layers_as_outputs
)
from Bio import SeqIO
import numpy as np
import torch
import torch.nn.functional as F
import os
import re
import sys
from dotenv import load_dotenv

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
print(project_root)

sys.path.append(project_root)
from calculo_feature.synthesizeUtils import train_test_between_files

__X_TRAIN__, __X_TEST__, __Y_TRAIN__, __Y_TEST__ = train_test_between_files()
sys.path.remove(project_root)

load_dotenv()
__SEED__ = os.getenv("__SEED__")
# =============================================================================
# PATHS
# =============================================================================
#%%
calculo_feature_dir = os.getcwd()
print(calculo_feature_dir)

path_data_processed = os.path.abspath(
    os.path.join(calculo_feature_dir, "../../../../data/processed/embeddings/sequence/proteinbert")
)

path_data_raw = os.path.abspath(
    os.path.join(calculo_feature_dir, "../../../../data/raw/string/fasta")
)

print(path_data_processed)
print(path_data_raw)

#%%
def lerFastaBio(arquivo):
    arquivoFasta = SeqIO.parse(open(arquivo), 'fasta')

    dict_fasta = {}

    for i in arquivoFasta:
        dict_fasta[i.id] = str(i.seq)

    return dict_fasta

def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


class ProteinBERT:

    def __init__(self, seq_len: int = 1024, batch_size: int = 8):

        self.seq_len = seq_len
        self.batch_size = batch_size

        pretrained_model_generator, input_encoder = load_pretrained_model()

        self.input_encoder = input_encoder

        self.model = get_model_with_hidden_layers_as_outputs(
            pretrained_model_generator.create_model(seq_len)
        )

    def computeResidueEmbeddings(self, sequences: list[str]) -> torch.Tensor:

        encoded_x = self.input_encoder.encode_X(sequences, self.seq_len)

        local_representations, _ = self.model.predict(
            encoded_x,
            batch_size=self.batch_size,
            verbose=0
        )

        return torch.tensor(local_representations, dtype=torch.float32)

    @staticmethod
    def mean_pooling(residue_embeddings, sequences):

        pooled = []

        for emb, seq in zip(residue_embeddings, sequences):
            seq_len = len(seq)
            emb_valid = emb[:seq_len]
            pooled.append(emb_valid.mean(dim=0))

        return torch.stack(pooled)

    def computeSentenceEmbeddings(self, sequences, normalize=True):

        # batching manual (EVITA CRASH DO TOKENIZER)
        all_embs = []

        for batch in batch_iter(sequences, self.batch_size):

            residue_embeddings = self.computeResidueEmbeddings(batch)

            protein_embeddings = self.mean_pooling(residue_embeddings, batch)

            if normalize:
                protein_embeddings = F.normalize(protein_embeddings, p=2, dim=1)

            all_embs.append(protein_embeddings)

        return torch.cat(all_embs, dim=0)

    def computePCAEmbeddings(self, sequences, n_components=100):

        embeddings = self.computeSentenceEmbeddings(sequences, normalize=True)

        embeddings_np = embeddings.cpu().numpy()

        pca = PCA(n_components=n_components, random_state=42)

        reduced = pca.fit_transform(embeddings_np)

        print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")

        return reduced

#%%
df_embeddings_sentence = pd.DataFrame()

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

    for bucket_id in range(n_buckets):

        inicio = bucket_id * bucket_size
        fim = min((bucket_id + 1) * bucket_size, len(sorted_df))

        bucket_df = sorted_df.iloc[inicio:fim]

        if len(bucket_df) == 0:
            continue

        buckets[bucket_id] = {
            "data": bucket_df,
            "max_len": bucket_df["sequence"].str.len().max()
        }
    for bucket in buckets.values():
        adp = ProteinBERT(batch_size=32, seq_len=int(bucket["max_len"]+2))

        embeddings = adp.computeSentenceEmbeddings(
            sequences=list(bucket["data"]["sequence"]),
            normalize=False
        )

        embedding_dim = embeddings.shape[1]

        colunas_embedding = [
            f"proteinbert_{i}"
            for i in range(embedding_dim)
        ]


        df_temp = pd.DataFrame(embeddings.cpu().numpy(), columns=colunas_embedding)
        df_temp.insert(0, "protein_id", list(bucket["data"]["protein_id"])[:len(df_temp)])

        df_embeddings_sentence = pd.concat(
            [df_embeddings_sentence, df_temp],
            ignore_index=True
        )
#%%
#separar mus e man para evitar data leakeage 
# man 6183 e mus 10090
df_man = df_embeddings_sentence[(df_embeddings_sentence["protein_id"].apply(lambda x:x.split('.')[0])).isin({'6183'})]
df_mus = df_embeddings_sentence[(df_embeddings_sentence["protein_id"].apply(lambda x:x.split('.')[0])).isin({'10090'})]

df_org_modelos = df_embeddings_sentence[~(df_embeddings_sentence["protein_id"].apply(lambda x:x.split('.')[0])).isin({'6183','10090'})]

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

pca_data_fit = scaler_fit.transform(data_to_fit.drop(columns=["protein_id"]).values)

pca_calc = PCA(n_components=0.95, random_state=__SEED__)
pca_calc_fit = pca_calc.fit(pca_data_fit)

pca_fin_org = pca_calc_fit.transform(pca_org_modelos_stand)
pca_fin_man = pca_calc_fit.transform(pca_man_stand)
pca_fin_mus = pca_calc_fit.transform(pca_mus_stand)

#renomeando as colunas
n_components_final = pca_fin_org.shape[1]

colunas_pca = [
    f"proteinbert_pca_{i}"
    for i in range(n_components_final)
]

df_pos_pca_org =  pd.DataFrame(pca_fin_org, columns=colunas_pca)
df_pos_pca_man =  pd.DataFrame(pca_fin_man, columns=colunas_pca)
df_pos_pca_mus =  pd.DataFrame(pca_fin_mus, columns=colunas_pca)

df_pos_pca_org.insert(0,"protein_id",list(df_org_modelos["protein_id"]))
df_pos_pca_man.insert(0,"protein_id",list(df_man["protein_id"]))
df_pos_pca_mus.insert(0,"protein_id",list(df_mus["protein_id"]))


nome_arquivo_final_pca_global_org = os.path.join(
    path_data_processed,
    "all.global.embedding.pca.proteinbert.tsv"
)
nome_arquivo_final_pca_global_man = os.path.join(
    path_data_processed,
    "6183.global.embedding.pca.proteinbert.tsv"
)
nome_arquivo_final_pca_global_mus = os.path.join(
    path_data_processed,
    "10090.global.embedding.pca.proteinbert.tsv"
)

nome_arquivo_final_global = os.path.join(
    path_data_processed,
    "global.embedding.proteinbert.tsv"
)

df_pos_pca_org.to_csv(nome_arquivo_final_pca_global_org,sep=' ',index=False)
df_pos_pca_man.to_csv(nome_arquivo_final_pca_global_man,sep=' ',index=False)
df_pos_pca_mus.to_csv(nome_arquivo_final_pca_global_mus,sep=' ',index=False)
df_embeddings_sentence.to_csv(nome_arquivo_final_global,sep=' ',index=False)
