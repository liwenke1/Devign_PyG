import logging
from os import stat


class Mylogging():
    def __init__(self):
        self.log_file = 'log.txt'
        logging.basicConfig(filename=self.log_file, level=logging.DEBUG,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        filemode='a')
    
    @staticmethod
    def info(message):
        logging.info(message)
    
    @staticmethod
    def debug(message):
        logging.debug(message)