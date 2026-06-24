#%%
import torch

print(torch.__version__)
print(torch.version.cuda)

#%%
import os

import pandas as pd
from dotenv import load_dotenv

from Node2VecEmbeddingExtractor import (
    Node2VecEmbeddingExtractor
)

load_dotenv()

NODE2VEC_DIMENSION = int(
    os.getenv("NODE2VEC_DIMENSION", 128)
)

NODE2VEC_WALK_LENGTH = int(
    os.getenv("NODE2VEC_WALK_LENGTH", 20)
)

NODE2VEC_NUM_WALKS = int(
    os.getenv("NODE2VEC_NUM_WALKS", 10)
)

NODE2VEC_WORKERS = int(
    os.getenv("NODE2VEC_WORKERS", 0)
)

NODE2VEC_WINDOW = int(
    os.getenv("NODE2VEC_WINDOW", 10)
)

NODE2VEC_EPOCHS = int(
    os.getenv("NODE2VEC_EPOCHS", 10)
)

BASE_DIR = os.getcwd()

RAW_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../../../data/raw/string/ppi"
    )
)

OUTPUT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "../../../../data/processed/embeddings/context/node2vec"
    )
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print(
    "PyTorch Geometric Node2Vec pipeline initialized"
)

print(
    BASE_DIR
)

print(
    RAW_DIR
)

print(
    OUTPUT_DIR
)

#%%
for organism in sorted(os.listdir(RAW_DIR)):

    organism_dir = os.path.join(
        RAW_DIR,
        organism
    )

    files = os.listdir(
        organism_dir
    )

    if not files:
        continue

    file_path = os.path.join(
        organism_dir,
        files[0]
    )

    print(
        f"\nProcessing organism: {organism}"
    )

    interaction_df = pd.read_csv(
        file_path,
        sep=" "
    )

    extractor = Node2VecEmbeddingExtractor(
        dimensions=NODE2VEC_DIMENSION,
        walk_length=NODE2VEC_WALK_LENGTH,
        num_walks=NODE2VEC_NUM_WALKS,
        workers=NODE2VEC_WORKERS,
        window=NODE2VEC_WINDOW,
        epochs=NODE2VEC_EPOCHS
    )

    embeddings_df = extractor.fit_transform(
        interaction_df
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{organism}_node2vec.csv"
    )

    embeddings_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved embeddings -> {output_path}"
    )

print("\nFinished.")
# %%
