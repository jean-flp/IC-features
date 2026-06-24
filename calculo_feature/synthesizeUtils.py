# calculo_feature/synthesizeUtils.py
from pathlib import Path
import os
import re

import pandas as pd
from Bio import SeqIO
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent

PATH_RAW_FASTA = BASE_DIR.parent / "data" / "raw" / "string" / "fasta"
PATH_ESSENTIAL = BASE_DIR.parent / "data" / "raw" / "essential"


def ler_fasta_bio(arquivo: str) -> dict:

    dict_fasta = {}

    with open(arquivo, "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            dict_fasta[record.id] = str(record.seq)

    return dict_fasta


def train_test_between_files():

    load_dotenv()

    seed = int(os.getenv("SEED"))
    test_size = float(os.getenv("TESTE_SIZE"))

    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    clean_ids = []
    clean_seqs = []

    excluded_folders = {"man", "mus"}

    for root, dirs, files in os.walk(PATH_RAW_FASTA):

        if any(folder in root for folder in excluded_folders):
            continue

        for file in files:

            if not file.endswith(".fa"):
                continue

            fasta_path = os.path.join(root, file)

            print(f"LENDO FASTA: {fasta_path}")

            dict_seq = ler_fasta_bio(fasta_path)

            for pid, seq in dict_seq.items():

                if not isinstance(seq, str):
                    continue

                seq = seq.upper()
                seq = re.sub(r"[^A-Z]", "", seq)

                if len(seq) < 5:
                    continue

                if not set(seq).issubset(valid_aa):
                    continue

                clean_ids.append(pid)
                clean_seqs.append(seq)

    if len(clean_ids) == 0:
        raise RuntimeError(
            f"Nenhuma proteína encontrada em {PATH_RAW_FASTA}"
        )

    df_sequences = pd.DataFrame(
        {
            "protein_id": clean_ids,
            "sequence": clean_seqs,
        }
    )

    df_sequences["seq_len"] = df_sequences["sequence"].str.len()

    df_sequences = df_sequences.sort_values(
        by="seq_len",
        ignore_index=True
    )

    df_essential = pd.read_csv(
        PATH_ESSENTIAL / "essential_genes.csv"
    )

    merged_df = pd.merge(
        df_sequences,
        df_essential,
        how="left",
        left_on="protein_id",
        right_on="Locus"
    )

    merged_df["essential"] = (
        merged_df["Locus"]
        .notna()
        .astype(int)
    )

    print(f"Proteínas totais: {len(merged_df)}")
    print(f"Essenciais: {merged_df['essential'].sum()}")
    print(
        f"Não essenciais: {(merged_df['essential'] == 0).sum()}"
    )

    X = merged_df[["protein_id"]]
    y = merged_df["essential"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test