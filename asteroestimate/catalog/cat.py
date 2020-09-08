import pandas as pd
import os

class catalog():
    """ Class for handling a catalog """

    def __init__(self):
        self.get_data()

    def get_data(self):
        df = pd.read_csv('~/Downloads/tess/TESS_CVZ_brightgiants_reduced_270820.csv')
        print(df.head())

if __name__ == '__main__':
    cat = catalog()
