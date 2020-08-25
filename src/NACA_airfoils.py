import re
import numpy as np
import matplotlib.pyplot as plt


class NACA_airfoils():

    def __init__(self):
        theta  = np.linspace(0, np.pi, 60)
        self.x = 0.5*abs(1-np.cos(theta))


    def NACA_1_series(self,airfoil_digits):
        """
            Generates a NACA 1-series airfoil
        """

        # extract airfoil properties
        m  = (int(airfoil_digits[0]) % 10) / 10
        cl = (int(airfoil_digits[1]) // 100) / 10
        t  = (int(airfoil_digits[1]) % 100) / 100

        # obtain x-coordinates and split them
        x0 = self.x

        # calculate the camber line
        yc   = np.zeros(len(x0));       dyc  = np.zeros(len(x0))
        yc   = -0.079577*cl*(np.multiply(x0[1:len(x0)-1],np.log(x0[1:len(x0)-1])) + np.multiply(1-x0[1:len(x0)-1],np.log(1-x0[1:len(x0)-1])))
        dyc  = -0.079577*cl*(np.log(x0[1:len(x0)-1]) - np.log(1-x0[1:len(x0)-1]))

        # calculate thickness
        yt = np.zeros(len(x0))  
        for i in range(len(x0)):
            if x0[i] < m:
                yt[i] = t*(0.989665*x0[i]**0.5-0.239250*x0[i]-0.041*x0[i]**2-0.5594*x0[i]**3)
            else:
                yt[i] = t*(0.01+2.325*(1-x0[i])-3.42*(1-x0[i])**2+1.46*(1-x0[i])**3)

        # create the arifoil
        xl  = np.zeros(len(x0));    xu  = np.zeros(len(x0));   yl  = np.zeros(len(x0));     yu  = np.zeros(len(x0))

        xl[1:len(xl)-1] = x0[1:len(xl)-1] + np.multiply(-yt[1:len(yt)-1],np.sin(np.arctan(dyc)))
        xu[1:len(xu)-1] = x0[1:len(xu)-1] + np.multiply(yt[1:len(yt)-1],np.sin(np.arctan(dyc)))
        yl[1:len(yl)-1] = yc - np.multiply(yt[1:len(yt)-1],np.cos(np.arctan(dyc)))
        yu[1:len(yu)-1] = yc + np.multiply(yt[1:len(yt)-1],np.cos(np.arctan(dyc)))

        # group surfaces into one data set
        x = np.concatenate((np.flip(xu[0:len(xu)-1]),xl[1:len(xl)-1]))
        y = np.concatenate((np.flip(yu[0:len(yu)-1]),yl[1:len(yl)-1]))

        return x, y, x0, yc

    def NACA_4_digit(self,airfoil_digits):
        """
            Generates a NACA 4-digit airfoil
        """

        # extract airfoil properties
        m  = (int(airfoil_digits[0]) // 1000) / 100
        p  = ((int(airfoil_digits[0]) // 100) % 10) / 10
        t  = (int(airfoil_digits[0]) % 100) / 100
          
        # obtain x-coordinates and split them
        x0 = self.x

        # calculate the camber line
        yc   = np.zeros(len(x0));       dyc  = np.zeros(len(x0))

        if m != 0 and p != 0:
            for i in range(len(x0)):
                if x0[i] <= p:
                    yc[i]  = m/p**2 * (2*p*x0[i] - x0[i]**2)
                    dyc[i] = 2*m/p*(1-x0[i]/p)
                else:
                    yc[i] = m/(1-p)**2 * ((1-2*p) + 2*p*x0[i] - x0[i]**2)
                    dyc[i] = 2*m/(1-p)**2*(p-x0[i])

        # calculate thickness
        if len(airfoil_digits) == 2:
            Ile = int(airfoil_digits[1]) // 10
            m1  = (int(airfoil_digits[1]) % 10) / 10

            d1_data = np.array([[0.2, 0.200],[0.3, 0.234],[0.4, 0.315],[0.5, 0.465],[0.6, 0.700]])

            # calculate coefficeints required for thickness calculations
            d0 = 0.002
            for i in range(np.shape(d1_data)[0]):
                if m1 == d1_data[i,0]:
                    d1 = d1_data[i,1]

            Dleft  = np.array([[(1-m1)**2, (1-m1)**3],[-2*(1-m1), -3*(1-m1)**2]])
            Dright = np.array([[0.1-d0-d1*(1-m1)], [d1]])
            Dans   = np.linalg.solve(Dleft, Dright)

            d2 = Dans[0];   d3 = Dans[1]

            a0 = 0.2969*t/0.2*Ile/6 

            Aleft  = np.array([[m1, m1**2, m1**3],[1, 2*m1, 3*m1**2],[0,2,6*m1]])
            Aright = np.array([ [0.1-a0*m1**0.5],[-0.5*a0*m1**(-0.5)],[(2*d1*(1-m1)-0.588)/(1-m1)**2+0.25*a0*m1**(-1.5)]])   
            Aans   = np.linalg.solve(Aleft,Aright)

            a1 = Aans[0];   a2 = Aans[1];   a3 = Aans[2]

            # calculate thickness
            yt = np.zeros(len(x0))
            for i in range(len(yt)):
                if x0[i] <= m1:
                    yt[i] = 5*t*(a0*x0[i]**0.5+a1*x0[i]+a2*x0[i]**2+a3*x0[i]**3)
                else:
                    yt[i] = 5*t*(d0*x0[i]**0.5+d1*(1-x0[i])+d2*(1-x0[i])**2+d3*(1-x0[i])**3)

        else:
            yt = 5*t*(0.2969*np.power(x0,0.5)-0.126*x0-0.3516*np.power(x0,2)+0.2843*np.power(x0,3)-0.1015*np.power(x0,4))

        # create the arifoil
        xl  = np.zeros(len(x0));    xu  = np.zeros(len(x0));   yl  = np.zeros(len(x0));     yu  = np.zeros(len(x0))

        xl = x0 + np.multiply(-yt,np.sin(np.arctan(dyc)))
        xu = x0 + np.multiply(yt,np.sin(np.arctan(dyc)))
        yl = yc - np.multiply(yt,np.cos(np.arctan(dyc)))
        yu = yc + np.multiply(yt,np.cos(np.arctan(dyc)))

        # group surfaces into one data set
        x = np.concatenate((np.flip(xu[0:len(xu)-1]),xl[1:len(xl)-1]))
        y = np.concatenate((np.flip(yu[0:len(yu)-1]),yl[1:len(yl)-1]))

        return x, y, x0, yc   

    def NACA_5_digit(self,airfoil_digits):
        """
            Generates a NACA 5-digit airfoil
        """

        # input default values for the camber line
        profile0 = np.array([[210, 220, 230, 240, 250],[0.05, 0.1, 0.15, 0.2, 0.25], 
                            [0.058, 0.126, 0.2025, 0.29, 0.391],[361.4, 51.64, 15.957, 6.643, 3.23]])
        profile1 = np.array([[221, 231, 241, 251],[0.1, 0.15, 0.2, 0.25], 
                            [0.13,0.217,0.318,0.441],[51.99, 15.793, 6.52, 3.191], 
                            [0.000764, 0.00677, 0.0303, 0.1355]])

        # extract airfoil properties
        camb  = int(airfoil_digits[0]) // 100
        s     = (int(airfoil_digits[0]) // 100) % 10
        t     = (int(airfoil_digits[0]) % 100) / 100

        # obtain x-coordinates and split them
        x0 = self.x

        # calculate the camber line
        yc   = np.zeros(len(x0))
        dyc  = np.zeros(len(x0))

        if s == 0:
            try:
                for j in range(5):
                    if camb == profile0[0,j]:
                        m  = profile0[2,j]
                        k1 = profile0[3,j]
            except ValueError:
                print("Enter a correct 5-digit airfoil")

            for i in range(len(x0)):
                if x0[i] <= m:
                    yc[i]  = k1/6*(x0[i]**3-3*m*x0[i]**2+m**2*(3-m)*x0[i])
                    dyc[i] = k1*(0.5*x0[i]**2-m*x0[i]+m**2*(3-m)/6)
                else:
                    yc[i]  = k1*m**3/6*(1-x0[i])
                    dyc[i] = -k1*m**3/6  

        elif s == 1:
            try:
                for j in range(4):
                    if camb == profile1[0,j]:
                        m    = profile1[2,j]
                        k1   = profile1[3,j]
                        k2k1 = profile1[4,j]
            except ValueError:
                print("Enter a correct 5-digit airfoil")       

            for i in range(len(x0)):
                if x0[i] <= m:
                    yc[i]  = k1/6*((x0[i]-m)**3-k2k1*x0[i]*(1-m)**3-m**3*x0[i]+m**3)
                    dyc[i] = k1/6*(3*(x0[i]-m)**2-k2k1*(1-m)**3-m**3)
                else:
                    yc[i]  = k1/6*(k2k1*(x0[i]-m)**3-k2k1*x0[i]*(1-m)**3-m**3*x0[i]+m**3)
                    dyc[i] = k1/6*(3*k2k1*(x0[i]-m)**2-k2k1*(1-m)**3-m**3) 

        else:
            raise ValueError("Enter a correct 5-digit airfoil")

        # calculate thickness
        if len(airfoil_digits) == 2:
            Ile = int(airfoil_digits[1]) // 10
            m1  = (int(airfoil_digits[1]) % 10) / 10

            d1_data = np.array([[0.2, 0.200],[0.3, 0.234],[0.4, 0.315],[0.5, 0.465],[0.6, 0.700]])

            # calculate coefficeints required for thickness calculations
            d0 = 0.002
            for i in range(np.shape(d1_data)[0]):
                if m1 == d1_data[i,0]:
                    d1 = d1_data[i,1]

            Dleft  = np.array([[(1-m1)**2, (1-m1)**3],[-2*(1-m1), -3*(1-m1)**2]])
            Dright = np.array([[0.1-d0-d1*(1-m1)], [d1]])
            Dans   = np.linalg.solve(Dleft, Dright)

            d2 = Dans[0];   d3 = Dans[1]

            a0 = 0.2969*t/0.2*Ile/6 

            Aleft  = np.array([[m1, m1**2, m1**3],[1, 2*m1, 3*m1**2],[0,2,6*m1]])
            Aright = np.array([ [0.1-a0*m1**0.5],[-0.5*a0*m1**(-0.5)],[(2*d1*(1-m1)-0.588)/(1-m1)**2+0.25*a0*m1**(-1.5)]])   
            Aans   = np.linalg.solve(Aleft,Aright)

            a1 = Aans[0];   a2 = Aans[1];   a3 = Aans[2]

            # calculate thickness
            yt = np.zeros(len(x0))
            for i in range(len(yt)):
                if x0[i] <= m1:
                    yt[i] = 5*t*(a0*x0[i]**0.5+a1*x0[i]+a2*x0[i]**2+a3*x0[i]**3)
                else:
                    yt[i] = 5*t*(d0*x0[i]**0.5+d1*(1-x0[i])+d2*(1-x0[i])**2+d3*(1-x0[i])**3)

        else:
            yt = 5*t*(0.2969*np.power(x0,0.5)-0.126*x0-0.3516*np.power(x0,2)+0.2843*np.power(x0,3)-0.1015*np.power(x0,4))

        # create the arifoil
        xl = x0 + np.multiply(-yt,np.sin(np.arctan(dyc)))
        xu = x0 + np.multiply(yt,np.sin(np.arctan(dyc)))
        yl = yc - np.multiply(yt,np.cos(np.arctan(dyc)))
        yu = yc + np.multiply(yt,np.cos(np.arctan(dyc)))

        # group surfaces into one data set
        x = np.concatenate((np.flip(xu[0:len(xu)-1]),xl[1:len(xl)-1]))
        y = np.concatenate((np.flip(yu[0:len(yu)-1]),yl[1:len(yl)-1]))

        return x, y, x0, yc

    def plot_airfoil(self,airfoil,x,y,x0,yc):
        """
            Plots an airfoil
        """

        fig, ax = plt.subplots()
        ax.plot(x, y,'k')
        ax.plot(x0[:len(yc)], yc)
        ax.axis('equal')
        ax.set(xlim=(0, 1), ylim=(-0.5, 0.5))
        plt.grid(True)

        fig.savefig(airfoil + ".png")
        plt.show()

        return

    def write_airfoil(self,airfoil_digits,x,y):
        """
            Writes an airfoil into a text file
        """
        airfoil_name = airfoil_digits[0]
        if len(airfoil_digits) == 2:
            airfoil_name = airfoil_digits[0] + airfoil_digits[1]

        f = open('NACA' + airfoil_name + '.dat','w')     
        f.write('NACA ' + airfoil_name + '\n')    

        for i in range(len(x)):
            f.write(str(x[i]) + '\t' + str(y[i]) + '\n')

        f.close()

        return



