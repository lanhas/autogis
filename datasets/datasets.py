from pathlib import Path


DEFAULT_ROOT = Path.cwd() / 'datasets'
datasets = {}


def register(name):
    def decorator(cls):
        datasets[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if kwargs.get('root_path') is None:
        kwargs['root_path'] = DEFAULT_ROOT / name
    dataset = datasets[name](**kwargs)
    return dataset
