from configparser import ConfigParser
def getConfig(config_file):
    parser = ConfigParser()
    parser.read(config_file)

    _config_ints = [(key,int(value)) for key,value in parser.items('ints')]
    _config_floats = [(key,float(value)) for key,value in parser.items('floats')]
    _config_strings = [(key,str(value)) for key,value in parser.items('strings')]
    _config_lists = [(key,list(value)) for key,value in parser.items('lists')]
    return dict(_config_ints + _config_floats + _config_strings + _config_lists)