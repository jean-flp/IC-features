from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self,tokenizer,model,device,labels=None):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.labels = labels
        self.tokens = None
    @abstractmethod
    def TokenizerInput(self,sentences):
        """This method must be implemented by all subclasses."""
        pass
    @abstractmethod
    def computeTokens(self):
        """This method must be implemented by all subclasses."""
        pass
    @abstractmethod
    def computeSentenceEmbeddings(self):
        """This method must be implemented by all subclasses."""
        pass
    @abstractmethod
    def mean_pooling(self):
        """This method must be implemented by all subclasses."""
        pass
    

# self.tokens = self.tokenizer(
#             sentences,
#             padding=True,
#             truncation=True,
#             return_tensors='pt'
#             ).to(self.model.device)
#         return self.tokens