import argparse as ap

class CommandLineArgumentParser:
    def __init__(self):
        # create parser
        self.parser = ap.ArgumentParser(description='A CFD solver for quick 2D equation prototyping.')

        # add arguments
        self.parser.add_argument('-i', '--input', help='input file for simulation', required=True)
        
        # parse arguments from command line arguments
        self.arguments = self.__parse()
    
    def __parse(self):
        return self.parser.parse_args()