import diskcache
from dash import DiskcacheManager
import multiprocessing

cache = diskcache.Cache("./cache")
background_callback_manager = DiskcacheManager(cache)

def acquire_lock(LOCK_KEY, TIMEOUT=30):
    """Try to acquire a lock, return True if successful."""
    if cache.get(LOCK_KEY):
        return False
    cache.set(LOCK_KEY, True, expire=TIMEOUT)
    return True

def release_lock(LOCK_KEY):
    cache.delete(LOCK_KEY)
