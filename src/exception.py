import sys


class CustomException(Exception):
    def __init__(self, original_exception: Exception):
        # keep original Exception object
        self.original_exception = original_exception
        # generate detailed error message
        self.error_message = self.get_detailed_error_message(original_exception)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(original_exception: Exception) -> str:
        _, _, exc_tb = sys.exc_info()
        if exc_tb is not None:
            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno
            return f"Error in {file_name}, line {line_number}: {str(original_exception)}"
        return str(original_exception)

    def __str__(self) -> str:
        return self.error_message
