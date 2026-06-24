import numpy as np
import pandas as pd
import torch


class Node2VecUtils:

    @staticmethod
    def build_protein_mapping(interactions_df: pd.DataFrame):
        proteins = sorted(
            set(interactions_df["protein1"])
            .union(set(interactions_df["protein2"]))
        )

        protein_to_id = {
            protein: idx
            for idx, protein in enumerate(proteins)
        }

        id_to_protein = {
            idx: protein
            for protein, idx in protein_to_id.items()
        }

        return protein_to_id, id_to_protein

    @staticmethod
    def build_edge_index(
        interactions_df: pd.DataFrame,
        protein_to_id: dict
    ) -> torch.Tensor:

        src = (
            interactions_df["protein1"]
            .map(protein_to_id)
            .to_numpy(dtype=np.int64)
        )

        dst = (
            interactions_df["protein2"]
            .map(protein_to_id)
            .to_numpy(dtype=np.int64)
        )

        # STRING/PPI geralmente é não-direcionado
        edge_index = np.vstack(
            (
                np.concatenate([src, dst]),
                np.concatenate([dst, src])
            )
        )

        return torch.from_numpy(edge_index)