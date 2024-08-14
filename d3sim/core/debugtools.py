import contextvars 
import contextlib
class DebugStoreContext:
    def __init__(self) -> None:
        self.store = {}

DEBUG_STORE_CTX: contextvars.ContextVar[DebugStoreContext | None] = contextvars.ContextVar("InlinerKernelLaunchContext", default=None)

@contextlib.contextmanager
def enter_debug_store(ctx: DebugStoreContext):
    token = DEBUG_STORE_CTX.set(ctx)
    try:
        yield ctx
    finally:
        DEBUG_STORE_CTX.reset(token)

@contextlib.contextmanager
def create_enter_debug_store():
    ctx = DebugStoreContext()
    token = DEBUG_STORE_CTX.set(ctx)
    try:
        yield ctx
    finally:
        DEBUG_STORE_CTX.reset(token)


def has_debug_store():
    return DEBUG_STORE_CTX.get() is not None


def save_to_store(key, value):
    ctx = DEBUG_STORE_CTX.get()
    if ctx is not None:
        ctx.store[key] = value

def get_from_store(key):
    ctx = DEBUG_STORE_CTX.get()
    if ctx is not None:
        return ctx.store.get(key)
    return None