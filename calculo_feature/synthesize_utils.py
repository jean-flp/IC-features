#%%
import os
import time
import pandas as pd
from Bio import SeqIO
import re
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

load_dotenv()
__SEED__ = int(os.getenv("SEED"))
__TEST_SIZE__ = float(os.getenv("TESTE_SIZE"))

print(__SEED__)
print(__TEST_SIZE__)


#%%
seconds_ini = time.time()

calculo_feature_dir = os.getcwd()

path_data_processed = os.path.abspath(
    os.path.join(calculo_feature_dir, "../data/processed/util")
)

path_raw_fasta_data = os.path.abspath(
    os.path.join(calculo_feature_dir, "../data/raw/string/fasta")
)

path_essential_raw = os.path.abspath(
    os.path.join(calculo_feature_dir, "../data/raw/essential")
)

print(path_data_processed)
print(path_raw_fasta_data)

dict_df = {}

#%%

# percorre TODAS as pastas e subpastas
def lerFastaBio(arquivo):
    arquivoFasta = SeqIO.parse(open(arquivo), 'fasta')

    dict_fasta = {}

    for i in arquivoFasta:
        dict_fasta[i.id] = str(i.seq)

    return dict_fasta
def toCompare_train_test_datasets():
    clean_ids = []
    clean_seqs = []
    excluded_folders = ["man","mus"]

    for root, dirs, files in os.walk(path_raw_fasta_data):
        for file in files:
            if any(pasta in root for pasta in excluded_folders):
                continue
            # pega apenas TSV
            if file.endswith(".fa"):

                file_path = os.path.join(root, file)

                print(f"\nARQUIVO ENCONTRADO:\n{file_path}")
                dict_seq = lerFastaBio(file_path)

                protein_ids = list(dict_seq.keys())
                sequences_raw = list(dict_seq.values())
                
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
    df_essential = pd.read_csv(os.path.join(path_essential_raw,"essential_genes.csv"),sep=',')
    
    print(f"PROTEINAS DF_ESSENTIAL ANTES MERGE: {df_essential["Locus"].nunique()}")

    merged_df = pd.merge(sorted_df,df_essential,how="left",left_on="protein_id",right_on="Locus")
    merged_df["essential"] = merged_df["Locus"].notna().astype(int)
    
    print(f"PROTEINAS MERGE_DF DEPOIS MERGE: {merged_df["Locus"].nunique()}")
    print("Proteínas totais:", len(merged_df))
    print("Essenciais:", merged_df["essential"].sum())
    print("Não essenciais:", (merged_df["essential"] == 0).sum())

    

    cols_to_drop = [
        "essential",
        "Locus",
        "Gene",
        "Essential_Genes",
        "Function",
        "Reference",
        "Code_Gene_DEG",
        "Seq_Gene",
        "Seq_Prot",
    ]
    X = merged_df.drop(columns=cols_to_drop)
    y = merged_df["essential"]

    x_train, x_test, y_train,y_test = train_test_split(X,y,test_size=__TEST_SIZE__,random_state=__SEED__,stratify=y)
    return (x_train, x_test, y_train,y_test)
#%%
x,y,z,w = toCompare_train_test_datasets()
print(x.info())
print(y.info())
print(z.info())
print(w.info())
print(z.value_counts(normalize=True))
print(w.value_counts(normalize=True))
# %%
