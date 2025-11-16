class EOProcessorError(Exception):
    """Base class for exceptions in this module."""
    pass

class InvalidArgumentError(EOProcessorError):
    """Raised when an invalid argument is passed to a function."""
    pass

class FileNotFoundError(EOProcessorError):
    """Raised when a file is not found."""
    pass

class ComputationError(EOProcessorError):
    """Raised when a computation fails."""
    pass
