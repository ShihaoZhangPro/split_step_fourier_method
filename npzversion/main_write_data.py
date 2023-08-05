import sys
import numpy as np

def main():

    # -- create mock-data
    z = np.asarray(range(10))
    A1 = np.asarray(range(10))+ 1j * np.asarray(range(10,20))

    # -- store data
    results = {
        "z"  : z,
        "A1" : A1
    }
    f_name = 'my_results'
    np.savez_compressed(f_name, **results)

main()
