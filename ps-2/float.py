import numpy as np
import matplotlib.pyplot as plt


def main():
    actual_num = 100.98763
    flt_num = np.float32(actual_num)

    int32bits = flt_num.view(np.int32) #convert 32 bit float to 32 bit integer,
    #store the memory it uses

    binrep = '{:032b}'.format(int32bits) # print the 32 bit integer in its binary representation

    print("\nNumpy's 32-bit floating point representation of ",actual_num,": ")
    print("\tSign: ",binrep[0])
    print("\tExponent: ",binrep[1:8])
    print("\tMantissa: ",binrep[9:])

    back2float = int32bits.view(np.float32)

    #difference between floating point representation and the actual number  
    print("\nDifference from actual num: ",back2float-actual_num)
    
if __name__ == "__main__":
    main()
    
