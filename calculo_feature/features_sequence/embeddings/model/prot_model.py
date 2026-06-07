from model.BaseModel import BaseModel
from model.Exceptions import LoadingModelException, TokensModelException

from transformers import (
    T5Tokenizer,
    T5EncoderModel
)

import torch
import re


class ProteinT5(BaseModel):

    def __init__(
        self,
        device: str,
        model_string: str
    ):

        try:

            model = T5EncoderModel.from_pretrained(
                model_string,
                torch_dtype=torch.float16
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
        sentences: list[str],
        padding=True,
        max_length=None
    ) -> dict:

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
            truncation=True,
            padding=padding,
            max_length=max_length,
            return_tensors="pt"
        )

        return self.tokens

    def computeTokens(self):

        try:

            if self.tokens is None:
                raise TokensModelException(self)

            input_ids = self.tokens[
                "input_ids"
            ].to(self.device)

            attention_mask = self.tokens[
                "attention_mask"
            ].to(self.device)

            self.model.eval()

            with torch.inference_mode():

                model_output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            return model_output

        except Exception as err:
            raise err

    @staticmethod
    def mean_pooling(
        hidden_states,
        attention_mask
    ):
        token_embeddings = hidden_states.last_hidden_state

        input_mask_expanded = (
            attention_mask
            .unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        pooled = torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9
        )

        return pooled

    def computeSentenceEmbeddings(
        self,
        remove_eos=True
    ):

        model_output = self.computeTokens()

        sentence_embeddings = self.mean_pooling(
            hidden_states=model_output,
            attention_mask=self.tokens[
                "attention_mask"
            ].to(self.device)
        )

        return sentence_embeddings