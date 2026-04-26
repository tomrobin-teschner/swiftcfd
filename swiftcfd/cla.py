import argparse as ap

class CommandLineArgumentParser:
    def __init__(self):
        # create parser
        self.parser = ap.ArgumentParser(description='A CFD solver for quick 2D equation prototyping.')

        # add arguments
        self.parser.add_argument('-i', '--input', help = 'input file for simulation',
                                                  required = False)
        self.parser.add_argument('-t', '--train', help = 'train ML model based on available datasets',
                                                  action = 'store_true',
                                                  required = False,
                                                  default = False)
        self.parser.add_argument('-v', '--variables', help = 'variables to use for training',
                                                  required = False)
        self.parser.add_argument('-m', '--model', help = 'ML model to use for training and inference',
                                                  choices = ['mlp', 'rnn', 'lstm', 'transformer'],
                                                  required = False)
        
        # parse arguments from command line arguments
        self.arguments = self.__parse()
    
        # perform check that all required parameters are available to continue
        self.__check_arguments()

    def __parse(self):
        return self.parser.parse_args()

    def __check_arguments(self):
        if not self.arguments.train and self.arguments.input is None:
            raise Exception('Input file is required for simulation.')

        if self.arguments.train and self.arguments.model is None:
            raise Exception('ML model is required for training.')

        if self.arguments.train and self.arguments.variables is None:
            raise Exception('Variables are required for training.')