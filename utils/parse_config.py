
def parse_model_config(path):
    """Parses the model configuration file"""
    fp = open(path, 'r', encoding='utf-8')
    lines = fp.readlines()
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines] # get rid of fringe whitespaces
    module_defs = dict()
    model_name = "default"
    for line in lines:
        if line.startswith('['):
            model_name = line[1:-1].rstrip()
            module_defs[model_name] = dict()
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[model_name][key.rstrip()] = value.strip()

    return module_defs

def parse_data_config(path):
    """Parses the data and server configuration file"""
    options = dict()
    with open(path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip()
            if line == '' or line.startswith('#'):
                continue
            key, value = line.split('=')
            options[key.strip()] = value.strip()
    return options
