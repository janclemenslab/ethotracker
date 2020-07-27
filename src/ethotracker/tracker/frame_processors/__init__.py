# register all modules
import pkgutil
processor_names = [submodule.name for submodule in pkgutil.iter_modules(__path__)]

def use(processor_name):
    if processor_name not in processor_names:
        raise ValueError(f'Unknown FrameProcessor "{processor_name}" - needs to be one of {processor_names}.')
    # import submodule and return Prc and init functions
    module = __import__(__name__+'.'+processor_name, fromlist='.')
    return module.Prc, module.init
