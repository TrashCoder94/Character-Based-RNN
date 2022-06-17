import random, sys

import numpy as np

def main():
    test = np.zeros((2, 3))
    lengthOfTest = len(test)
    print("Test has " + str(lengthOfTest) + " elements")
    print(test)

    test += [4, 5, 6]
    lengthOfTest = len(test)
    print("Now test has " + str(lengthOfTest) + " elements")
    print(test)

if __name__ == '__main__':
    main()