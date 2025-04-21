import logging

def setup_logging(logging_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, logging_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def catch_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.exception("Exception in %s", func.__name__)
            raise e
    return wrapper