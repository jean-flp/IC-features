import pandas as pd
import torch
from torch_geometric.nn import Node2Vec

from Node2VecUtils import Node2VecUtils


class Node2VecEmbeddingExtractor:

    def __init__(
        self,
        dimensions: int,
        walk_length: int,
        num_walks: int,
        workers: int,
        window: int,
        epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 0.01,
        device: str = None
    ):

        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.window = window

        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = None
        self.protein_to_id = None
        self.id_to_protein = None

    def fit(self, interaction_df: pd.DataFrame):

        self.protein_to_id, self.id_to_protein = (
            Node2VecUtils.build_protein_mapping(interaction_df)
        )

        edge_index = Node2VecUtils.build_edge_index(
            interaction_df,
            self.protein_to_id
        ).to(self.device)

        self.model = Node2Vec(
            edge_index=edge_index,
            embedding_dim=self.dimensions,
            walk_length=self.walk_length,
            context_size=self.window,
            walks_per_node=self.num_walks,
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True
        ).to(self.device)

        loader = self.model.loader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )

        optimizer = torch.optim.SparseAdam(
            self.model.parameters(),
            lr=self.learning_rate
        )

        self.model.train()

        for epoch in range(self.epochs):

            total_loss = 0.0

            for pos_rw, neg_rw in loader:

                pos_rw = pos_rw.to(self.device)
                neg_rw = neg_rw.to(self.device)

                optimizer.zero_grad()

                loss = self.model.loss(
                    pos_rw,
                    neg_rw
                )

                loss.backward()

                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(loader)

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"- loss={avg_loss:.6f}"
            )

        return self

    def transform(self) -> pd.DataFrame:

        self.model.eval()

        with torch.no_grad():
            embeddings = (
                self.model.embedding.weight
                .detach()
                .cpu()
                .numpy()
            )

        rows = []

        for node_id, protein in self.id_to_protein.items():

            rows.append(
                [protein, *embeddings[node_id]]
            )

        columns = (
            ["protein_id"]
            + [f"n2v_{i}" for i in range(self.dimensions)]
        )

        return pd.DataFrame(
            rows,
            columns=columns
        )

    def fit_transform(
        self,
        interaction_df: pd.DataFrame
    ) -> pd.DataFrame:

        self.fit(interaction_df)

        return self.transform()

    def save_embeddings(
        self,
        path: str
    ):

        embeddings_df = self.transform()

        embeddings_df.to_csv(
            path,
            index=False
        )