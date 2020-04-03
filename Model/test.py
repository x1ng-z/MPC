import numpy as np
import sys
if __name__=='__main__':
    for i in range(0, len(sys.argv)):
        strs = sys.argv[i]
        print(i)
        if i==2:
            print(int(strs))
