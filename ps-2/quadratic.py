import math
import numpy as np

def quadratic(a,b,c):
    
    if (b>0):
        x1 = np.float64((2*c)/(-b - math.sqrt((b**2) - (4*a*c))))
        x2 = np.float64((-b - math.sqrt(b**2 - 4*a*c))/(2*a))

    else:
        x1 = np.float64((-b + math.sqrt(b**2 - 4*a*c))/(2*a))
        x2 = np.float64((2*c)/(-b + math.sqrt((b**2) - (4*a*c))))

    print ("x1: ",x1," x2: ",x2)
    print(np.abs(x1 - (- 1.e-6)))
    print(np.abs(x2 - (- 0.999999999999e+6)))
    return x1,x2

def main():
    quadratic(0.001,1000,0.001)
    
if __name__ == "__main__":
    main()
