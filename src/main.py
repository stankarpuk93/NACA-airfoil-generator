import re
import numpy as np
from NACA_airfoils import NACA_airfoils

# Version 1.0
# Author:   S. Karpuk
# Created:  08.2020
# Modified:

'''
    NACA airfoil generator

    The code creates user-specified NACA-airfoils
    Available:
        NACA 1-series
        NACA 4-digit
        NACA modified 4-digit
        NACA 5-digit
        NACA 6-series

'''

def main():
    print('NACA airfoil generator')
    print('----------------------\nAvailable airfoil types: ')
    print('                 1. NACA 1-series (ex. NACA 16-012)')
    print('                 2. NACA 4-series (ex. NACA 0012 or  NACA 0012-63)')
    print('                 3. NACA  5-digit (ex. NACA 23015 or NACA 23012-63)')
    
    airfoil = input("Enter your name : ")

    airfoil_digits = np.asarray(re.findall(r'\d+', airfoil))
    airfoil_specs = [int(i) for i in airfoil_digits] 

    NACA = NACA_airfoils()

    # NACA 1-series airfoils
    if len(airfoil_specs) == 2 and airfoil_specs[0] // 10 == 1:
        x,y,x0,yc = NACA.NACA_1_series(airfoil_digits)
    
    # NACA 4-digit airfoils
    elif len(airfoil_digits[0]) == 4 or (len(airfoil_digits[0]) == 4 and len(airfoil_digits[1]) == 2):
        x,y,x0,yc = NACA.NACA_4_digit(airfoil_digits)

    # NACA 5-digit airfoils
    elif len(airfoil_digits[0]) == 5 or (len(airfoil_digits[0]) == 5 and len(airfoil_digits[1]) == 2):
        print('available 5-digit airfoils: \n')
        print('        NACA 210__, NACA 220__, NACA 230__, NACA 240__, NACA 250__ \n')
        print('                    NACA 221__, NACA 231__, NACA 241__, NACA 251__ \n')                   
        x,y,x0,yc = NACA.NACA_5_digit(airfoil_digits)  
    
    else:
        print('Incorrect airfoil definition. Check the input')

    # Write airfoil into a file
    NACA.write_airfoil(airfoil_digits,x,y)

    # Plot the airfoil
    NACA.plot_airfoil(airfoil,x,y,x0,yc)

    return




if __name__ == '__main__': 
    main()    