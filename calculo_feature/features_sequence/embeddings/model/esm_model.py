from model.BaseModel import BaseModel
from model.Exceptions import (
    LoadingModelException,
    TokensModelException
)

from transformers import (
    EsmModel,
    EsmTokenizer
)

import torch


class Esm(BaseModel):

    def __init__(
        self,
        device: str,
        model_string: str,
    ):

        try:

            model = EsmModel.from_pretrained(
                model_string,
            ).to(device)

            model.eval()

            tokenizer = EsmTokenizer.from_pretrained(
                model_string,
                do_lower_case=False
            )

            super().__init__(
                model=model,
                tokenizer=tokenizer,
                device=device
            )

            self.tokens = None

        except Exception as err:
            raise LoadingModelException(err)

    def TokenizerInput(
        self,
        sequences,
        padding=True,
        #truncation=True,
        max_length=None
    ):

        try:

            if isinstance(sequences, str):
                sequences = [sequences]

            inputs = self.tokenizer(
                sequences,
                return_tensors="pt",
                padding=padding,
                #truncation=truncation,
                max_length=max_length
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            return inputs

        except Exception as err:
            raise TokensModelException(err)

    def computeTokens(
        self,
        sequences
    ):

        try:

            self.tokens = self.TokenizerInput(
                sequences=sequences
            )

            return self.tokens

        except Exception as err:
            raise TokensModelException(err)

    def mean_pooling(
        self,
        embeddings,
        attention_mask,
        remove_special_tokens=True
    ):

        if remove_special_tokens:

            # remove <cls> e <eos>
            embeddings = embeddings[:, 1:-1]

            attention_mask = attention_mask[:, 1:-1]

        mask = attention_mask.unsqueeze(-1)

        masked_embeddings = embeddings * mask

        sum_embeddings = masked_embeddings.sum(dim=1)

        seq_lengths = mask.sum(dim=1)

        pooled_embeddings = (
            sum_embeddings / seq_lengths
        )

        return pooled_embeddings

    def computeSentenceEmbeddings(
        self,
        sequences,
        pooling_strategy="mean",
        remove_special_tokens=True,
        normalize=False
    ):

        try:

            inputs = self.computeTokens(
                sequences=sequences
            )

            with torch.no_grad():

                outputs = self.model(**inputs)

                embeddings = outputs.last_hidden_state

                if pooling_strategy == "mean":

                    embeddings = self.mean_pooling(
                        embeddings=embeddings,
                        attention_mask=inputs[
                            "attention_mask"
                        ],
                        remove_special_tokens=remove_special_tokens
                    )

                elif pooling_strategy == "cls":

                    embeddings = embeddings[:, 0]

                else:

                    raise ValueError(
                        f"Pooling strategy "
                        f"{pooling_strategy} "
                        f"not supported."
                    )

                if normalize:

                    embeddings = torch.nn.functional.normalize(
                        embeddings,
                        p=2,
                        dim=1
                    )

            return embeddings

        except Exception as err:
            raise TokensModelException(err)

    def computeResidueEmbeddings(
        self,
        sequences,
        remove_special_tokens=True
    ):

        try:

            inputs = self.computeTokens(
                sequences=sequences
            )

            with torch.no_grad():

                outputs = self.model(**inputs)

                embeddings = outputs.last_hidden_state

                if remove_special_tokens:

                    embeddings = embeddings[:, 1:-1]

            return embeddings

        except Exception as err:
            raise TokensModelException(err)