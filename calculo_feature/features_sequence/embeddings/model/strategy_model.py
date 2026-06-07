from model.BaseModel import BaseModel
from model.esm_model import Esm
from model.prot_model import ProteinT5

import torch
import gc

model_dict = {
    "facebook/esm2_t36_3B_UR50D":Esm, #"facebook/esm2_t36_3B_UR50D": Esm,
    "Rostlab/prot_t5_xl_uniref50": ProteinT5
}

class ModelContext:
    def __init__(self):
        self.strategy: BaseModel = None

    def setStrategy(self, model_name: str):
        strategy_class = model_dict.get(model_name)

        if strategy_class is None:
            raise ValueError(f"Modelo não suportado: {model_name}")

        self.strategy_class = strategy_class
        self.model_name = model_name

        return self

    def initializeModel(self, device: str, model_string: str):
        self.strategy = self.strategy_class(device, model_string)
        return self

    def tokenize(self, sequences, padding=True, max_length=None):
        return self.strategy.TokenizerInput(
            sequences,
            padding=padding,
            max_length=max_length
        )
    def computeTokens(self):
        return self.strategy.computeTokens()
    
    def computeSentenceEmbeddings(self):
        return self.strategy.computeSentenceEmbeddings().detach().cpu().numpy()
    
    def clear(self):
        import gc
        if self.strategy is not None:
            del self.strategy
            self.strategy = None

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()