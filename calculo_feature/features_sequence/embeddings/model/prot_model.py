from model.BaseModel import BaseModel
from model.Exceptions import LoadingModelException, TokensModelException

from transformers import (
    T5Tokenizer,
    T5EncoderModel
)

import torch
import torch.nn.functional as F
import re


class ProteinT5(BaseModel):

    def __init__(
        self,
        device: str,
        model_string: str
    ):

        try:

            model = T5EncoderModel.from_pretrained(
                model_string
            ).to(device)

            tokenizer = T5Tokenizer.from_pretrained(
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
        sentences: list[str]
    ) -> dict:

        # substitui aminoácidos raros/ambíguos
        sequences_formatted = [
            " ".join(
                list(
                    re.sub(r"[UZOB]", "X", sequence)
                )
            )
            for sequence in sentences
        ]

        self.tokens = self.tokenizer(
            sequences_formatted,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        return self.tokens

    def computeTokens(self):

        try:

            if self.tokens is None:
                raise TokensModelException(self)

            input_ids = self.tokens["input_ids"].to(
                self.device
            )

            attention_mask = self.tokens[
                "attention_mask"
            ].to(self.device)

            self.model.eval()

            with torch.no_grad():

                model_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            return model_output

        except Exception as err:
            raise err

    @staticmethod
    def mean_pooling(
        model_output,
        attention_mask
    ):

        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            1
        ) / torch.clamp(
            input_mask_expanded.sum(1),
            min=1e-9
        )

    def computeSentenceEmbeddings(self):

        model_output = self.computeTokens()

        sentence_embeddings = self.mean_pooling(
            model_output,
            self.tokens["attention_mask"].to(
                self.device
            )
        )
        #normalize por sua magnitude 
        sentence_embeddings = F.normalize(
            sentence_embeddings,
            p=2, # norma euclidiana 
            dim=1
        )

        return sentence_embeddings