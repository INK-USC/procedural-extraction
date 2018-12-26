from tqdm import tqdm

_method_adaptors = dict()

def register_dist_adaptor(method_name):
    def decorator(func):
        _method_adaptors[method_name] = func
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
        return wrapper
    return decorator

def getNearestMethod(method_name, parser):
    """
    all candidates toked
    all protocol untoked
    input:
    [
        (protocol, candidate1, candidate2, ...),
        (protocol, candidate1, candidate2, ...),
        (protocol, candidate1, candidate2, ...),
        ...
    ]
    output:
    [
        nearest_idx1,
        nearest_idx2,
        nearest_idx3,
        ...
    ]
    """
    return _method_adaptors[method_name](parser)

def getMethodNames():
    return list(_method_adaptors.keys())
        