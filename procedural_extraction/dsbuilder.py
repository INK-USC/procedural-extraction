_type_builders = dict()

def register_dsbuilder(type_name):
    def decorator(func):
        _type_builders[type_name] = func
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
        return wrapper
    return decorator

def get_builder(type_name, parser):
    return _type_builders[type_name](parser)

def get_builder_names():
    return list(_type_builders.keys())