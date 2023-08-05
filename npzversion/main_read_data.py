import sys
import numpy as np

def fetch_data(f_name):
    dat = np.load(f_name)
    return dat['z'], dat['A1']

def main():
    f_name = 'my_results.npz'
    z, A1 = fetch_data(f_name)
    print(z)
    print(A1)

main()
