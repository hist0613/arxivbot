import time
from functools import wraps
from logger import logger
from settings import TIME_PAUSE_SEC


def llm_retry(max_trials):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            exceptions = []
            for attempt in range(max_trials):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    exceptions.append(e)
                    time.sleep(attempt * TIME_PAUSE_SEC)
            # raise Exception(f"All {max_trials} attempts failed. Errors: {exceptions}")
            logger.warning(f"All {max_trials} attempts failed. Errors: {exceptions}")
            return ""

        return wrapper

    return decorator
