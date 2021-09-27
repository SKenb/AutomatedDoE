from Common import Common
from Common import Factor

import logging

    

def main():
    
    set1 = Factor.getDefaultFactorSet()

    print(set1)

def initializeDoE():
    pass

def evaluation():
    pass

if __name__ == '__main__':

    Common.initLogging()
    logging.info("Start main DoE program")

    main()
    