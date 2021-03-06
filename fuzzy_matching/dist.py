_method_adaptors = dict()

def register_dist_adaptor(method_name):
    def decorator(func):
        _method_adaptors[method_name] = func
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
        return wrapper
    return decorator

def get_nearest_method(method_name, parser):
    """
    all candidates toked
    all protocol untoked
    input:
    queries:
    [
        (protocol, (candidate, sen_id, start, K), (candidate, sen_id, start, K), ...)
        (protocol, (candidate, sen_id, start, K), (candidate, sen_id, start, K), ...)
        (protocol, (candidate, sen_id, start, K), (candidate, sen_id, start, K), ...)
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

def get_method_names():
    return list(_method_adaptors.keys())
        