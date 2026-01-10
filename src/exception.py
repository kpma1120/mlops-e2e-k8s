import sys


class CustomException(Exception):
    """Custom exception class that wraps original exceptions with detailed context."""

    def __init__(self, original_exception: Exception):
        """
        Initialize with the original exception and formatted error message."""
        # keep original Exception object
        self.original_exception = original_exception
        # generate detailed error message
        self.error_message = self.get_detailed_error_message(original_exception)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(original_exception: Exception) -> str:
        """Generate detailed error message including file name and line number."""
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return (
                f"Error in {file_name}, line {line_number}: {str(original_exception)}"
            )
        return str(original_exception)

    def __str__(self) -> str:
        """Return the formatted error message string."""
        return self.error_message
