import configparser
'''
This parser file parses the config file.
The config file should be divided into sections.
and a key value pair per section is generated.
'''

def parse(file_name):
    parser = configparser.ConfigParser()
    parser.read(file_name)
    conf={}
    for sect in parser.sections():
        conf[sect]={}
        for k,v in parser.items(sect):
            conf[sect][k]=v
    return conf