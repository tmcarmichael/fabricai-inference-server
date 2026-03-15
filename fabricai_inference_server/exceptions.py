"""
Custom exception types.
"""


class QueueFullError(Exception):
    def __init__(self):
        self.message = "Request queue is full. Too many requests in progress."
        super().__init__(self.message)


class ModelNotFoundError(Exception):
    def __init__(self, detail: str = "Model not found"):
        self.message = detail
        super().__init__(self.message)


class BackendUnavailableError(Exception):
    def __init__(self, backend: str, detail: str = ""):
        self.message = f"Backend '{backend}' is unavailable"
        if detail:
            self.message += f": {detail}"
        super().__init__(self.message)


class BackendError(Exception):
    def __init__(self, backend: str, detail: str = ""):
        self.message = f"Backend '{backend}' error"
        if detail:
            self.message += f": {detail}"
        super().__init__(self.message)
