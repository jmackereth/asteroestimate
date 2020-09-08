import pandas as pd
import os

class catalog():
    """ Class for handling a catalog """

    def __init__(self, catafile='~/Downloads/tess/TESS_CVZ_brightgiants_reduced_270820.csv'):
        self.catafile= catafile
        self.get_data()

    def get_data(self):
        self.df = pd.read_csv(self.catafile)
        print(self.df.head())

    def make_cut_good(self):
        self.df_good = self.df.loc[self.df.numax_dnu_consistent == 1]
        self.df_good = self.df_good.loc[self.df_good.lum_flag_BHM == 1]

    def print(self):
        print(f'Length of full data frame {len(self.df)}')
        print(f'Length of good data frame {len(self.df_good)}')

    def print_columns(self):
        for i in self.df.columns:
            print(i)

if __name__ == '__main__':
    cat = catalog()
    cat.print()
    cat.make_cut_good()
    cat.print()
