# api/Exception/LoadingModelException.py

class LoadingModelException(Exception):
    """Custom exception for model loading errors."""

    def __init__(self, model_path: str, message: str = None):
        if message is None:
            message = f"Erro ao carregar o modelo: {model_path}"
        super().__init__(message)
        self.model_path = model_path

# api/Exception/TokensModelException.py

class TokensModelException(Exception):
    """Custom exception for tokens loading errors."""

    def __init__(self, message: str = None):
        if message is None:
            message = f"Erro ao carregar o tokens"
        super().__init__(message)
        self.tokens = None