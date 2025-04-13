"""
exceptions.py

A list of app-specific exceptions with description.
"""


class QueueFullException(Exception):
    """
    Raised when the request queue is at maximum capacity.
    """

    def __init__(
        self, message: str = "Request queue is full. Too many requests in progress."
    ):
        super().__init__(message)
        self.message = message


class ModelNotFoundException(Exception):
    """
    Raised when the specified model is missing or can't be loaded.
    """

    def __init__(self, message: str = "Model file not found"):
        super().__init__(message)
        self.message = message
