# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt

Hf_29815_H2O = -241.83  # this is Hf (kJ/(mol))
s_29815_H2O = 188.84  # (J/(mol*K))

Hf_29815_CH4 = -74.87  # this is Hf (kJ/(mol))
s_29815_CH4 = 186.25  # (J/(mol*K)) P = 1bar

Hf_29815_CO = -110.53  # this is Hf (kJ/(mol))
s_29815_CO = 197.66  # (J/(mol*K)) P = 1bar

Hf_29815_CO2 = -393.52  # this is Hf (kJ/(mol))
s_29815_CO2 = 213.79  # (J/(mol*K)) P = 1bar

Hf_29815_C2H2 = 226.73  # this is Hf (kJ/(mol))
s_29815_C2H2 = 200.93  # (J/(mol*K)) P = 1bar

Hf_29815_H2 = 0.0  # this is Hf (kJ/(mol))
s_29815_H2 = 130.68  # (J/(mol*K)) P = 1bar

Hf_29815_TiO = 54.39 # this is Hf (kJ/(mol))
s_29815_TiO = 233.46 # (J/(mol*K))

Hf_29815_TiO2 = -305.43 # this is Hf (kJ/(mol))
s_29815_TiO2 = 260.14 # (J/(mol*K))

Hf_29815_Ti = 473.63 # this is Hf (kJ/(mol))
s_29815_Ti = 180.30 # (J/(mol*K))

R = 8.314e-3 # kJ/(mol K)
P_0 = 1.0 #bar
P_1 = 1.0 #bar


select_abu = input("What elementary abundances you want to use? \n \
            [1] Lodders 2003 \n \
            [2] Asplund et al. 2009 \n \
            [3] Heng 2016 aprox + Ti from Lodders (Just for TiCHO) \n \
            Answer ('1', '2' or '3'): ")

if select_abu == "1":
    # Protosolar
    # SOLAR ELEMENTARY ABUNDANCES: (Lodders et al, 2003)
    n_H = 1.0
    n_He = 0.0968277856261
    n_Ti = 9.54992586021e-8
    n_V = 1.09647819614e-8
    n_O = 6.02559586074e-4
    n_C = 2.75422870334e-4
    n_N = 8.12830516164e-5
    n_S = 1.62181009736e-5
    n_Na = 2.23872113857e-6
    n_K = 1.44543977075e-7
    n_Fe = 3.2359365693e-5
elif select_abu == "2":
    # Present-day photosphere
    # This file contains elemental solar abundances as found by Asplund et al. 2009.
    # http://adsabs.harvard.edu/abs/2009ARA%26A..47..481A
    n_H = 1.0
    n_He = 10**(10.93 - 12.) * n_H
    n_Ti = 10**(4.95 - 12.) * n_H
    n_V = 10**(3.93 - 12.) * n_H
    n_O = 10**(8.69 - 12.) * n_H
    n_C = 10**(8.43 - 12.) * n_H
    n_N = 10**(7.83 - 12.) * n_H
    n_S = 10**(7.12 - 12.) * n_H
    n_Na = 10**(6.24 - 12.) * n_H
    n_K = 10**(5.03 - 12.) * n_H
    n_Fe = 10**(7.5 - 12.) * n_H
elif select_abu == "3":
    # Kevin et al. used n_0 = 5e-4 and rate_CO_solar = 0.5 (hence n_C = 2.5e-4)
    n_O = 5.0e-4
    n_C = 2.5e-4
    n_Ti = 9.54992586021e-8
    n_N = 1e-4

# Using Looders Data: ratio_CO_solar = 0.46

# If you want to generate abundances with C / O = X, for example,
# simply increase the carbon by a factor of (X / [C / O_solar]),
# so the new ratio of carbon to oxygen will be X.

rate_CO_solar = n_C/n_O  # This number is C/O = 0.46 for abundances of Lodders and C/O = 0.5 for Kevin et al.

def dH(A,B,C,D,E,F,H,t): # kJ/(mol)
    '''
    :param A, B, C, D, E , F, H:
    Shomate equation parameters for thermochemical functions. Data taken from the NIST database.
    :param t: Time. It's define as t=T/1000 (type = array)
    :return: d Enthalpy
    '''
    dH_MOLECULE = A*t + B*t**2/2 + C*t**3/3 + D*t**4/4 - E/t + F - H
    return dH_MOLECULE

def s(A,B,C,D,E,G,t): # J/(mol*K)
    '''
    :param A, B, C, D, E , F, H:
    Shomate equation parameters for thermochemical functions. Data taken from the NIST database.
    :param t: Time. It's define as t=T/1000 (type = array)
    :return: Entropy
    '''
    s_MOLECULE = A*np.log(t) + B*t + C*t**2/2 + D*t**3/3 - E/(2*t**2) + G
    return s_MOLECULE

def dG(dH,t,dS):
    '''
    :param dH: Enthalpy
    :param t: Time. It's define as t=T/1000 (type = array)
    :param dS: Entropy
    :return: Gibbs Fre Energy
    '''
    dG = dH - t*dS
    return dG

# enthalpy and free energy of reaction at 298.15 K for one reaction (e.g. CO2 + H2 <-> CO + H2O)

def Hrxn_29815(Hf_29815_reactant1,Hf_29815_reactant2,Hf_29815_product1,Hf_29815_product2):
    Hrxn_29815 = Hf_29815_product1 + Hf_29815_product2 - Hf_29815_reactant1 - Hf_29815_reactant2
    return Hrxn_29815

def Srxn_29815(S_29815_reactant1,S_29815_reactant2,S_29815_product1,S_29815_product2):
    Srxn_29815 = S_29815_product1 + S_29815_product2 - S_29815_reactant1 - S_29815_reactant2
    return Srxn_29815

def Grxn_29815(Hrxn_29815,Srxn_29815):
    Grxn_29815 = Hrxn_29815 - 298.15*(Srxn_29815)/1000 # kJ/mol
    return Grxn_29815

def molecule_H_S(Hf_29815_molecule, s_29815_molecule,A,B,C,D,E,F,G,H,t):
    dH_molecule = dH(A, B, C, D, E, F, H,t)
    h_molecule = dH_molecule + Hf_29815_molecule
    s_molecule = s(A, B, C, D, E, G,t)
    ds_molecule = s_molecule - s_29815_molecule
    return dH_molecule, h_molecule, s_molecule, ds_molecule

def root_searcher(A, T, j):

    count_1bar = 0
    count_P_wanted = 0

    if j == 0:
        n_H2O_1bar = []
    if j == 1:
        n_H2O_P_wanted = []

    name = ['1bar','P_'+P_wanted_str+'bar']
    file_roots_calculation = open('roots_calculation_'+name[j]+'.dat', 'w')

    for i in np.arange(26):

        file_roots_calculation.write("# Coefficients (A) \n")
        file_roots_calculation.write(str(A[0][i]) + '|' + str(A[1][i]) + '|' + str(A[2][i]) + '|' + str(A[3][i]) + '|' + str(A[4][i]) + '|' + str(A[5][i]) + '|' + str(A[6][i]))
        roots = poly.polyroots([A[0][i], A[1][i], A[2][i], A[3][i], A[4][i], A[5][i], A[6][i]])
        file_roots_calculation.write("\n# Roots for T=" + str(T[i]))

        list_dif_roots = []
        list_pos_roots = []

        for w in np.arange(len(roots)):

            two_n_O = 2.0 * n_O  # We know that n_H2O should be approx. 2*n_O at low temperatures
            possible_n_H2O = roots[w]
            dif = np.abs(possible_n_H2O - two_n_O)
            file_roots_calculation.write("\nPossible ñ_H2O (root " + str(w + 1) + "): " + str(possible_n_H2O))
            file_roots_calculation.write("\n2 ñ_O (We know that ñ_H2O should be approx. this): " + str(two_n_O))

            if ('{0:.1f}'.format(possible_n_H2O.real) == '{0:.1f}'.format(two_n_O) or (possible_n_H2O.real <= two_n_O and possible_n_H2O.real > 0.0)) and possible_n_H2O.imag == 0.0:

                file_roots_calculation.write("\n~~~~~ MATCH 1313 ~~~~~(T=" + str(T[i]) + ") | Possible_n_H2O.real=" + str(possible_n_H2O.real) + " | two_n_O=" + '{0:.6f}'.format(two_n_O) + '\n')
                list_dif_roots.append(dif)
                list_pos_roots.append(possible_n_H2O.real)

        array_dif_roots = np.array(list_dif_roots)
        array_pos_roots = np.array(list_pos_roots)

        idx = np.where(array_dif_roots == array_dif_roots.min())[0]

        if j == 0:
            print(idx)
            print(array_pos_roots[idx])
            n_H2O_1bar.append(float(array_pos_roots[idx]))
            count_1bar += 1  # This number should be 26 at the end of the loops (because we have 26 temperatures)

        if j == 1:
            print(idx)
            print(array_pos_roots[idx])
            n_H2O_P_wanted.append(float(array_pos_roots[idx]))
            count_P_wanted += 1

    if j == 0:
        print("***count_1bar*** = ", count_1bar)
        file_roots_calculation.write("\n***count_1bar*** = " + str(count_1bar))
        return n_H2O_1bar
    if j == 1:
        print("***count_P_wanted*** = ", count_P_wanted)
        file_roots_calculation.write("\n***count_P_wanted*** = " + str(count_P_wanted))
        return n_H2O_P_wanted

    print("(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))")
    file_roots_calculation.write("\n(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))\n")

    file_roots_calculation.close()

def root_searcher_1(A, T, j):

    count_1bar = 0
    count_P_wanted = 0

    if j == 0:
        n_H2O_1bar = []
    if j == 1:
        n_H2O_P_wanted = []

    name = ['1bar','P_'+P_wanted_str+'bar']
    file_roots_calculation = open('roots_calculation_'+name[j]+'.dat', 'w')

    for i in np.arange(26):

        file_roots_calculation.write("# Coefficients (A) \n")
        file_roots_calculation.write(str(A[0][i]) + '|' + str(A[1][i]) + '|' + str(A[2][i]) + '|' + str(A[3][i]) + '|' + str(A[4][i]) + '|' + str(A[5][i]) + '|' + str(A[6][i]))
        roots = poly.polyroots([A[0][i], A[1][i], A[2][i], A[3][i], A[4][i], A[5][i], A[6][i]])
        file_roots_calculation.write("\n# Roots for T=" + str(T[i]))

        list_dif_roots = []
        list_pos_roots = []

        for w in np.arange(len(roots)):

            two_n_O = 2.0 * n_O  # We know that n_H2O should be approx. 2*n_O at low temperatures
            possible_n_H2O = roots[w]
            dif = np.abs(possible_n_H2O - two_n_O)
            file_roots_calculation.write("\nPossible ñ_H2O (root " + str(w + 1) + "): " + str(possible_n_H2O))
            file_roots_calculation.write("\n2 ñ_O (We know that ñ_H2O should be approx. this): " + str(two_n_O))

            if ('{0:.1f}'.format(possible_n_H2O.real) == '{0:.1f}'.format(two_n_O) or (possible_n_H2O.real <= two_n_O and possible_n_H2O.real > 0.0)) and possible_n_H2O.imag == 0.0:

                file_roots_calculation.write("\n~~~~~ MATCH 1313 ~~~~~(T=" + str(T[i]) + ") | Possible_n_H2O.real=" + str(possible_n_H2O.real) + " | two_n_O=" + '{0:.6f}'.format(two_n_O) + '\n')
                list_dif_roots.append(dif)
                list_pos_roots.append(possible_n_H2O.real)

        array_dif_roots = np.array(list_dif_roots)
        array_pos_roots = np.array(list_pos_roots)

        idx = np.where(array_pos_roots == array_pos_roots.min())[0]

        if j == 0:
            n_H2O_1bar.append(float(array_pos_roots[idx]))
            count_1bar += 1  # This number should be 26 at the end of the loops (because we have 26 temperatures)

        if j == 1:
            n_H2O_P_wanted.append(float(array_pos_roots[idx]))
            count_P_wanted += 1

    if j == 0:
        print("***count_1bar*** = ", count_1bar)
        file_roots_calculation.write("\n***count_1bar*** = " + str(count_1bar))
        return n_H2O_1bar
    if j == 1:
        print("***count_P_wanted*** = ", count_P_wanted)
        file_roots_calculation.write("\n***count_P_wanted*** = " + str(count_P_wanted))
        return n_H2O_P_wanted

    print("(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))")
    file_roots_calculation.write("\n(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))\n")

    file_roots_calculation.close()

def root_searcher_3(A, T, j):

    count_1bar = 0
    count_P_wanted = 0

    if j == 0:
        n_H2O_1bar = []
    if j == 1:
        n_H2O_P_wanted = []

    name = ['P_1bar','P_'+P_wanted_str+'bar']
    file_roots_calculation = open('roots_calculation_'+name[j]+'.dat', 'w')

    for i in np.arange(26):

        file_roots_calculation.write("# Coefficients (A) \n")
        file_roots_calculation.write(str(A[0][i]) + '|' + str(A[1][i]) + '|' + str(A[2][i]) + '|' + str(A[3][i]) + '|' + str(A[4][i]) + '|' + str(A[5][i]) + '|' + str(A[6][i]) + '|' + str(A[7][i]) + '|' + str(A[8][i]) + '|' + str(A[9]))
        roots = poly.polyroots([A[0][i], A[1][i], A[2][i], A[3][i], A[4][i], A[5][i], A[6][i], A[7][i], A[8][i], A[9][i]])
        file_roots_calculation.write("\n# Roots for T=" + str(T[i]))

        list_dif_roots = []
        list_pos_roots = []

        for w in np.arange(len(roots)):

            two_n_O = 2.0 * n_O  # We know that n_H2O should be approx. 2*n_O at low temperatures
            possible_n_H2O = roots[w]
            dif = np.abs(possible_n_H2O - two_n_O)
            file_roots_calculation.write("\nPossible ñ_H2O (root " + str(w + 1) + "): " + str(possible_n_H2O))
            file_roots_calculation.write("\n2 ñ_O (We know that ñ_H2O should be approx. this): " + str(two_n_O))

            if ('{0:.1f}'.format(possible_n_H2O.real) == '{0:.1f}'.format(two_n_O) or (possible_n_H2O.real <= two_n_O and possible_n_H2O.real > 0.0)) and possible_n_H2O.imag == 0.0:

                file_roots_calculation.write("\n~~~~~ MATCH 1313 ~~~~~(T=" + str(T[i]) + ") | Possible_n_H2O.real=" + str(possible_n_H2O.real) + " | two_n_O=" + '{0:.6f}'.format(two_n_O) + '\n')
                list_dif_roots.append(dif)
                list_pos_roots.append(possible_n_H2O.real)

        array_dif_roots = np.array(list_dif_roots)
        array_pos_roots = np.array(list_pos_roots)

        idx = np.where(array_dif_roots == array_dif_roots.min())[0]

        if j == 0:
            n_H2O_1bar.append(float(array_pos_roots[idx]))
            count_1bar += 1  # This number should be 26 at the end of the loops (because we have 26 temperatures)

        if j == 1:
            n_H2O_P_wanted.append(float(array_pos_roots[idx]))
            count_P_wanted += 1

    if j == 0:
        print("***count_1bar*** = ", count_1bar)
        file_roots_calculation.write("\n***count_1bar*** = " + str(count_1bar))
        return n_H2O_1bar
    if j == 1:
        print("***count_P_wanted*** = ", count_P_wanted)
        file_roots_calculation.write("\n***count_P_wanted*** = " + str(count_P_wanted))
        return n_H2O_P_wanted

    print("(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))")
    file_roots_calculation.write("\n(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))\n")

    file_roots_calculation.close()

def root_searcher_4(A, T, j):

    count_1bar = 0
    count_P_wanted = 0

    if j == 0:
        n_H2O_1bar = []
    if j == 1:
        n_H2O_P_wanted = []

    name = ['1bar','P_'+P_wanted_str+'bar']
    file_roots_calculation = open('roots_calculation_'+name[j]+'.dat', 'w')

    for i in np.arange(26):

        file_roots_calculation.write("# Coefficients (A) \n")

        file_roots_calculation.write(str(A[0][i]) + '|' + str(A[1][i]) + '|' + str(A[2][i]) + '|' + str(A[3]))
        roots = poly.polyroots([A[0][i], A[1][i], A[2][i], A[3]])
        file_roots_calculation.write("\n# Roots for T=" + str(T[i]))

        list_dif_roots = []
        list_pos_roots = []

        for w in np.arange(len(roots)):

            two_n_O = 2.0 * n_O  # We know that n_H2O should be approx. 2*n_O at low temperatures
            possible_n_H2O = roots[w]
            dif = np.abs(possible_n_H2O - two_n_O)
            file_roots_calculation.write("\nPossible ñ_H2O (root " + str(w + 1) + "): " + str(possible_n_H2O))
            file_roots_calculation.write("\n2 ñ_O (We know that ñ_H2O should be approx. this): " + str(two_n_O))

            if ('{0:.1f}'.format(possible_n_H2O.real) == '{0:.1f}'.format(two_n_O) or (possible_n_H2O.real <= two_n_O and possible_n_H2O.real > 0.0)) and possible_n_H2O.imag == 0.0:

                file_roots_calculation.write("\n~~~~~ MATCH 1313 ~~~~~(T=" + str(T[i]) + ") | Possible_n_H2O.real=" + str(possible_n_H2O.real) + " | two_n_O=" + '{0:.6f}'.format(two_n_O) + '\n')
                list_dif_roots.append(dif)
                list_pos_roots.append(possible_n_H2O.real)

        array_dif_roots = np.array(list_dif_roots)
        array_pos_roots = np.array(list_pos_roots)

        idx = np.where(array_dif_roots == array_dif_roots.min())[0]

        if j == 0:
            n_H2O_1bar.append(float(array_pos_roots[idx]))
            count_1bar += 1  # This number should be 26 at the end of the loops (because we have 26 temperatures)

        if j == 1:
            n_H2O_P_wanted.append(float(array_pos_roots[idx]))
            count_P_wanted += 1

    if j == 0:
        print("***count_1bar*** = ", count_1bar)
        file_roots_calculation.write("\n***count_1bar*** = " + str(count_1bar))
        return n_H2O_1bar
    if j == 1:
        print("***count_P_wanted*** = ", count_P_wanted)
        file_roots_calculation.write("\n***count_P_wanted*** = " + str(count_P_wanted))
        return n_H2O_P_wanted

    print("(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))")
    file_roots_calculation.write("\n(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))\n")

    file_roots_calculation.close()

def insert_one_temp(temperature, P_wanted):

    file_T_H_S_molecules = open('T_H_S_molecules.dat', 'w')

    file_T_H_S_molecules.write("\n        Molecule  |  Temperature |       H       |       S       |")
    file_T_H_S_molecules.write("\n     ----------------------------------------------------------------------------------")

    t = temperature/1000

    #       H2O

    if temperature >= 500.0 and temperature <= 1700.0:

        # 500-1700 K valid temperature range
        A = 30.09200
        B = 6.832514
        C = 6.793435
        D = -2.534480
        E = 0.082139
        F = -250.8810
        G = 223.3967
        H = -241.8264

    elif temperature > 1700.0 and temperature <= 6000.0:

        # 1700-6000 K valid temperature range
        A = 41.96426
        B = 8.622053
        C = -1.499780
        D = 0.098119
        E = -11.15764
        F = -272.1797
        G = 219.7809
        H = -241.8264

    else:
        print("ERROR: Enter a valid temperature between 500K and 6000K!")

    dH_H2O, h_H2O, s_H2O, ds_H2O = molecule_H_S(Hf_29815_H2O, s_29815_H2O, A, B, C, D, E, F, G, H, t)

    file_T_H_S_molecules.write("\n \t" + "H2O" + str(temperature) + "  |" + str(h_H2O) + "|" + str(s_H2O) + "|")

    #       CH4

    if temperature >= 298.0 and temperature <= 1300.0:
        # 298-1300 K valid temperature range
        A = -0.73029
        B = 108.4773
        C = -42.52157
        D = 5.862788
        E = 0.678565
        F = -76.84376
        G = 158.7163
        H = -74.87310

    elif temperature > 1300.0 and temperature <= 6000.0:

        # 1300-6000 K valid temperature range
        A = 85.81217
        B = 11.26467
        C = -2.114146
        D = 0.138190
        E = -26.42221
        F = -153.5327
        G = 224.4143
        H = -74.87310

    else:
        print("ERROR: Enter a valid temperature between 298K and 6000K!")

    dH_CH4, h_CH4, s_CH4, ds_CH4 = molecule_H_S(Hf_29815_CH4, s_29815_CH4, A, B, C, D, E, F, G, H, t)

    file_T_H_S_molecules.write("\n \t" + "CH4" + str(temperature) + "  |" + str(h_CH4) + "|" + str(s_CH4) + "|")

    #       CO

    if temperature >= 298.0 and temperature <= 1300.0:
        # 298-1300 K valid temperature range
        A = 25.56759
        B = 6.096130
        C = 4.054656
        D = -2.671301
        E = 0.131021
        F = -118.0089
        G = 227.3665
        H = -110.5271

    elif temperature > 1300.0 and temperature <= 6000.0:

        # 1300-6000 K valid temperature range
        A = 35.15070
        B = 1.300095
        C = -0.205921
        D = 0.013550
        E = -3.282780
        F = -127.8375
        G = 231.7120
        H = -110.5271

    else:
        print("ERROR: Enter a valid temperature between 298K and 6000K!")

    dH_CO, h_CO, s_CO, ds_CO = molecule_H_S(Hf_29815_CO, s_29815_CO, A, B, C, D, E, F, G, H, t)

    file_T_H_S_molecules.write("\n \t" + "CO" + str(temperature) + "  |" + str(h_CO) + "|" + str(s_CO) + "|")

    #       CO2

    if temperature >= 298.0 and temperature <= 1200.0:
        # 298-1200 K valid temperature range
        A = 24.99735
        B = 55.18696
        C = -33.69137
        D = 7.948387
        E = -0.136638
        F = -403.6075
        G = 228.2431
        H = -393.5224

    elif temperature > 1200.0 and temperature <= 6000.0:

        # 1200-6000 K valid temperature range
        A = 58.16639
        B = 2.720074
        C = -0.492289
        D = 0.038844
        E = -6.447293
        F = -425.9186
        G = 263.6125
        H = -393.5224

    else:
        print("ERROR: Enter a valid temperature between 298K and 6000K!")

    dH_CO2, h_CO2, s_CO2, ds_CO2 = molecule_H_S(Hf_29815_CO2, s_29815_CO2, A, B, C, D, E, F, G, H, t)

    file_T_H_S_molecules.write("\n \t" + "CO2" + str(temperature) + "  |" + str(h_CO2) + "|" + str(s_CO2) + "|")


    #       C2H2

    if temperature >= 298.0 and temperature <= 1100.0:

        # 298-1100 K valid temperature range
        A = 40.68697
        B = 40.73279
        C = -16.17840
        D = 3.669741
        E = -0.658411
        F = 210.7067
        G = 235.0052
        H = 226.7314

    elif temperature > 1100.0 and temperature <= 6000.0:

        # 1100-6000 K valid temperature range
        A = 67.47244
        B = 11.75110
        C = -2.021470
        D = 0.136195
        E = -9.806418
        F = 185.4550
        G = 253.5337
        H = 226.7314

    else:
        print("ERROR: Enter a valid temperature between 298K and 6000K!")

    dH_C2H2, h_C2H2, s_C2H2, ds_C2H2 = molecule_H_S(Hf_29815_C2H2, s_29815_C2H2, A, B, C, D, E, F, G, H, t)

    file_T_H_S_molecules.write("\n \t" + "C2H2" + str(temperature) + "  |" + str(h_C2H2) + "|" + str(s_C2H2) + "|")

    #       H2

    if temperature >= 298.0 and temperature <= 1000.0:
        # 298-1000 K valid temperature range
        A = 33.066178
        B = -11.363417
        C = 11.432816
        D = -2.772874
        E = -0.158558
        F = -9.980797
        G = 172.707974
        H = 0.0

    elif temperature > 1000.0 and temperature <= 2500.0:
        # 1000-2500 K valid temperature range
        A = 18.563083
        B = 12.257357
        C = -2.859786
        D = 0.268238
        E = 1.977990
        F = -1.147438
        G = 156.288133
        H = 0.0

    elif temperature > 2500.0 and temperature <= 6000.0:

        # 2500-6000 K valid temperature range
        A = 43.413560
        B = -4.293079
        C = 1.272428
        D = -0.096876
        E = -20.533862
        F = -38.515158
        G = 162.081354
        H = 0.0

    else:
        print("ERROR: Enter a valid temperature between 298K and 6000K!")

    dH_H2, h_H2, s_H2, ds_H2 = molecule_H_S(Hf_29815_H2, s_29815_H2, A, B, C, D, E, F, G, H, t)

    file_T_H_S_molecules.write("\n \t" + "H2" + str(temperature) + "  |" + str(h_H2) + "|" + str(s_H2) + "|")

    file_T_H_S_molecules.close()

    # reaction 1: CH4 + H2O <-> CO + 3 H2

    print("\n# Reaction 1: CH4 + H2O <-> CO + 3 H2")

    Hrxn1_29815 = Hrxn_29815(Hf_29815_CH4, Hf_29815_H2O, Hf_29815_CO, 3 * Hf_29815_H2)
    Srxn1_29815 = Srxn_29815(s_29815_CH4, s_29815_H2O, s_29815_CO, 3 * s_29815_H2)
    Grxn1_29815 = Grxn_29815(Hrxn1_29815, Srxn1_29815)

    Hrxn1 = Hrxn1_29815 + dH_CO + 3 * dH_H2 - dH_CH4 - dH_H2O
    Grxn1 = Hrxn1 - temperature * (s_CO + 3 * s_H2 - s_CH4 - s_H2O) / 1000

    # Equilibrium constant calculation (K')

    K1_P_wanted = (P_0 / P_wanted) ** 2 * np.exp(-Grxn1 / (R * temperature))

    print("Ready :D! \n")

    # reaction 2: CO2 + H2 <-> CO + H2O

    print("#reaction 2: CO2 + H2 <-> CO + H2O")

    Hrxn2_29815 = Hrxn_29815(Hf_29815_CO2, Hf_29815_H2, Hf_29815_CO, Hf_29815_H2O)
    Srxn2_29815 = Srxn_29815(s_29815_CO2, s_29815_H2, s_29815_CO, s_29815_H2O)
    Grxn2_29815 = Grxn_29815(Hrxn2_29815, Srxn2_29815)

    Hrxn2 = Hrxn2_29815 + dH_CO + dH_H2O - dH_CO2 - dH_H2
    Grxn2 = Hrxn2 - temperature * (s_CO + s_H2O - s_CO2 - s_H2) / 1000


    # Equilibrium constant calculation (K')

    K2_sin_presion = np.exp(-Grxn2 / (R * temperature))

    print("Ready :D! \n")

    # reaction 3: 2 CH4 <-> C2H2 + 3 H2

    print("#reaction 3: 2 CH4 <-> C2H2 + 3 H2")

    Hrxn3_29815 = Hrxn_29815(2 * Hf_29815_CH4, 0.0, Hf_29815_C2H2, 3 * Hf_29815_H2)
    Srxn3_29815 = Srxn_29815(2 * s_29815_CH4, 0.0, s_29815_C2H2, 3 * s_29815_H2)
    Grxn3_29815 = Grxn_29815(Hrxn3_29815, Srxn3_29815)

    Hrxn3 = Hrxn3_29815 + dH_C2H2 + 3 * dH_H2 - 2 * dH_CH4
    Grxn3 = Hrxn3 - temperature * (s_C2H2 + 3 * s_H2 - 2 * s_CH4) / 1000

    # Equilibrium constant calculation (K')

    K3_P_wanted = (P_0 / P_wanted) ** 2 * np.exp(-Grxn3 / (R * temperature))

    print("Ready :D! \n")

    return K1_P_wanted, K2_sin_presion, K3_P_wanted

def classic_calculation():

    T = np.linspace(500, 1700, 13)  # degrees K
    t = T / 1000

    #       H2O

    # 500-1700 K valid temperature range
    A = 30.09200
    B = 6.832514
    C = 6.793435
    D = -2.534480
    E = 0.082139
    F = -250.8810
    G = 223.3967
    H = -241.8264

    dH_H2O_1 = dH(A, B, C, D, E, F, H,t)
    h_H2O_1 = dH_H2O_1 + Hf_29815_H2O
    s_H2O_1 = s(A, B, C, D, E, G,t)
    # ds_H2O_1 = s_H2O_1 - s_29815_H2O

    T = np.linspace(1800, 3000, 13)  # degrees K
    t = T / 1000

    # 1700-6000 K valid temperature range
    A = 41.96426
    B = 8.622053
    C = -1.499780
    D = 0.098119
    E = -11.15764
    F = -272.1797
    G = 219.7809
    H = -241.8264

    dH_H2O_2 = dH(A, B, C, D, E, F, H,t)
    h_H2O_2 = dH_H2O_2 + Hf_29815_H2O
    s_H2O_2 = s(A, B, C, D, E, G,t)
    # ds_H2O_2 = s_H2O_2 - s_29815_H2O

    dH_H2O = np.array(list(dH_H2O_1) + list(dH_H2O_2))
    h_H2O = np.array(list(h_H2O_1) + list(h_H2O_2))
    s_H2O = np.array(list(s_H2O_1) + list(s_H2O_2))
    # ds_H2O = np.array(list(ds_H2O_1) + list(ds_H2O_2))

    file_T_H_S_H2O = open('T_H_S_H2O.dat', 'w')

    file_T_H_S_H2O.write("# WATER (H2O)\n")
    file_T_H_S_H2O.write("# Temperature\tdH\tH\tS\n")

    for i in np.arange(26):
        file_T_H_S_H2O.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_H2O[i]) + "\t" + str(h_H2O[i]) + "\t" + str(s_H2O[i]) + "\n")

    file_T_H_S_H2O.close()

    #       CH4

    T = np.linspace(500, 1300, 9)  # degrees K
    t = T / 1000

    # 298-1300 K valid temperature range
    A = -0.73029
    B = 108.4773
    C = -42.52157
    D = 5.862788
    E = 0.678565
    F = -76.84376
    G = 158.7163
    H = -74.87310

    dH_CH4_1 = dH(A, B, C, D, E, F, H,t)
    h_CH4_1 = dH_CH4_1 + Hf_29815_CH4
    s_CH4_1 = s(A, B, C, D, E, G,t)
    # ds_CH4_1 = s_CH4_1 - s_29815_CH4

    T = np.linspace(1400, 3000, 17)  # degrees K
    t = T / 1000

    # 1300-6000 K valid temperature range
    A = 85.81217
    B = 11.26467
    C = -2.114146
    D = 0.138190
    E = -26.42221
    F = -153.5327
    G = 224.4143
    H = -74.87310

    dH_CH4_2 = dH(A, B, C, D, E, F, H,t)
    h_CH4_2 = dH_CH4_2 + Hf_29815_CH4
    s_CH4_2 = s(A, B, C, D, E, G,t)
    # ds_CH4_2 = s_CH4_2 - s_29815_CH4

    dH_CH4 = np.array(list(dH_CH4_1) + list(dH_CH4_2))
    h_CH4 = np.array(list(h_CH4_1) + list(h_CH4_2))
    s_CH4 = np.array(list(s_CH4_1) + list(s_CH4_2))
    # ds_CH4 = np.array(list(ds_CH4_1) + list(ds_CH4_2))

    file_T_H_S_CH4 = open('T_H_S_CH4.dat', 'w')

    file_T_H_S_CH4.write("# METHANE (CH4)\n")
    file_T_H_S_CH4.write("# Temperature\tdH\tH\tdS\tS\n")

    for i in np.arange(26):
        file_T_H_S_CH4.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_CH4[i]) + "\t" + str(h_CH4[i]) + "\t" + str(s_CH4[i]) + "\n")

    file_T_H_S_CH4.close()

    #       CO

    T = np.linspace(500, 1300, 9)  # degrees K
    t = T / 1000

    # 298-1300 K valid temperature range
    A = 25.56759
    B = 6.096130
    C = 4.054656
    D = -2.671301
    E = 0.131021
    F = -118.0089
    G = 227.3665
    H = -110.5271


    dH_CO_1 = dH(A, B, C, D, E, F, H,t)
    h_CO_1 = dH_CO_1 + Hf_29815_CO
    s_CO_1 = s(A, B, C, D, E, G,t)
    # ds_CO_1 = s_CO_1 - s_29815_CO

    T = np.linspace(1400, 3000, 17)  # degrees K
    t = T / 1000

    # 1300-6000 K valid temperature range
    A = 35.15070
    B = 1.300095
    C = -0.205921
    D = 0.013550
    E = -3.282780
    F = -127.8375
    G = 231.7120
    H = -110.5271

    dH_CO_2 = dH(A, B, C, D, E, F, H, t)
    h_CO_2 = dH_CO_2 + Hf_29815_CO
    s_CO_2 = s(A, B, C, D, E, G, t)
    # ds_CO_2 = s_CO_2 - s_29815_CO

    dH_CO = np.array(list(dH_CO_1) + list(dH_CO_2))
    h_CO = np.array(list(h_CO_1) + list(h_CO_2))
    s_CO = np.array(list(s_CO_1) + list(s_CO_2))
    # ds_CO = np.array(list(ds_CO_1) + list(ds_CO_2))

    file_T_H_S_CO = open('T_H_S_CO.dat', 'w')

    file_T_H_S_CO.write("# CARBON MONOXIDE (CO)\n")
    file_T_H_S_CO.write("# Temperature\tdH\tH\tS\n")

    for i in np.arange(26):
        file_T_H_S_CO.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_CO[i]) + "\t" + str(h_CO[i]) + "\t" + str(s_CO[i]) + "\n")

    file_T_H_S_CO.close()

    #       H2


    T = np.linspace(500, 1000, 6)  # degrees K
    t = T / 1000

    # 298-1000 K valid temperature range
    A = 33.066178
    B = -11.363417
    C = 11.432816
    D = -2.772874
    E = -0.158558
    F = -9.980797
    G = 172.707974
    H = 0.0


    dH_H2_1 = dH(A, B, C, D, E, F, H, t)
    h_H2_1 = dH_H2_1 + Hf_29815_H2
    s_H2_1 = s(A, B, C, D, E, G, t)
    # ds_H2_1 = s_H2_1 - s_29815_H2

    T = np.linspace(1100, 2500, 15)  # degrees K
    t = T / 1000

    # 1000-2500 K valid temperature range
    A = 18.563083
    B = 12.257357
    C = -2.859786
    D = 0.268238
    E = 1.977990
    F = -1.147438
    G = 156.288133
    H = 0.0

    dH_H2_2 = dH(A, B, C, D, E, F, H, t)
    h_H2_2 = dH_H2_2 + Hf_29815_H2
    s_H2_2 = s(A, B, C, D, E, G, t)
    # ds_H2_2 = s_H2_2 - s_29815_H2

    T = np.linspace(2600, 3000, 5)  # degrees K
    t = T / 1000

    # 2500-6000 K valid temperature range
    A = 43.413560
    B = -4.293079
    C = 1.272428
    D = -0.096876
    E = -20.533862
    F = -38.515158
    G = 162.081354
    H = 0.0

    dH_H2_3 = dH(A, B, C, D, E, F, H, t)
    h_H2_3 = dH_H2_3 + Hf_29815_H2
    s_H2_3 = s(A, B, C, D, E, G, t)
    # ds_H2_3 = s_H2_3 - s_29815_H2

    dH_H2 = np.array(list(dH_H2_1) + list(dH_H2_2) + list(dH_H2_3))
    h_H2 = np.array(list(h_H2_1) + list(h_H2_2) + list(h_H2_3))
    s_H2 = np.array(list(s_H2_1) + list(s_H2_2) + list(s_H2_3))
    # ds_H2 = np.array(list(ds_H2_1) + list(ds_H2_2) + list(ds_H2_3))

    file_T_H_S_H2 = open('T_H_S_H2.dat', 'w')

    file_T_H_S_H2.write("# HYDROGEN (H2)\n")
    file_T_H_S_H2.write("# Temperature\tdH\tH\tS\n")
    for i in np.arange(26):
        file_T_H_S_H2.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_H2[i]) + "\t" + str(h_H2[i]) + "\t" + str(s_H2[i]) + "\n")

    file_T_H_S_H2.close()

    #       CO2

    T = np.linspace(500, 1200, 8)  # degrees K
    t = T / 1000

    # 298-1200 K valid temperature range
    A = 24.99735
    B = 55.18696
    C = -33.69137
    D = 7.948387
    E = -0.136638
    F = -403.6075
    G = 228.2431
    H = -393.5224


    dH_CO2_1 = dH(A, B, C, D, E, F, H,t)
    h_CO2_1 = dH_CO2_1 + Hf_29815_CO2
    s_CO2_1 = s(A, B, C, D, E, G,t)
    # ds_CO2_1 = s_CO2_1 - s_29815_CO2

    T = np.linspace(1300, 3000, 18)  # degrees K
    t = T / 1000

    # 1200-6000 K valid temperature range
    A = 58.16639
    B = 2.720074
    C = -0.492289
    D = 0.038844
    E = -6.447293
    F = -425.9186
    G = 263.6125
    H = -393.5224

    dH_CO2_2 = dH(A, B, C, D, E, F, H,t)
    h_CO2_2 = dH_CO2_2 + Hf_29815_CO2
    s_CO2_2 = s(A, B, C, D, E, G,t)
    # ds_CO2_2 = s_CO2_2 - s_29815_CO2

    dH_CO2 = np.array(list(dH_CO2_1) + list(dH_CO2_2))
    h_CO2 = np.array(list(h_CO2_1) + list(h_CO2_2))
    s_CO2 = np.array(list(s_CO2_1) + list(s_CO2_2))
    # ds_CO2 = np.array(list(ds_CO2_1) + list(ds_CO2_2))

    file_T_H_S_CO2 = open('T_H_S_CO2.dat', 'w')

    file_T_H_S_CO2.write("# CARBON DIOXIDE (CO2) \n")
    file_T_H_S_CO2.write("# Temperature\tdH\tH\tS\n")

    for i in np.arange(26):
        file_T_H_S_CO2.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_CO2[i]) + "\t" + str(h_CO2[i]) + "\t" + str(s_CO2[i]) + "\n")

    file_T_H_S_CO2.close()

    #       C2H2

    T = np.linspace(500, 1100, 7)  # degrees K
    t = T / 1000

    # 298-1100 K valid temperature range
    A = 40.68697
    B = 40.73279
    C = -16.17840
    D = 3.669741
    E = -0.658411
    F = 210.7067
    G = 235.0052
    H = 226.7314


    dH_C2H2_1 = dH(A, B, C, D, E, F, H,t)
    h_C2H2_1 = dH_C2H2_1 + Hf_29815_C2H2
    s_C2H2_1 = s(A, B, C, D, E, G,t)
    # ds_C2H2_1 = s_C2H2_1 - s_29815_C2H2

    T = np.linspace(1200, 3000, 19)  # degrees K
    t = T / 1000

    # 1100-6000 K valid temperature range
    A = 67.47244
    B = 11.75110
    C = -2.021470
    D = 0.136195
    E = -9.806418
    F = 185.4550
    G = 253.5337
    H = 226.7314

    dH_C2H2_2 = dH(A, B, C, D, E, F, H,t)
    h_C2H2_2 = dH_C2H2_2 + Hf_29815_C2H2
    s_C2H2_2 = s(A, B, C, D, E, G,t)
    # ds_C2H2_2 = s_C2H2_2 - s_29815_C2H2

    dH_C2H2 = np.array(list(dH_C2H2_1) + list(dH_C2H2_2))
    h_C2H2 = np.array(list(h_C2H2_1) + list(h_C2H2_2))
    s_C2H2 = np.array(list(s_C2H2_1) + list(s_C2H2_2))
    # ds_C2H2 = np.array(list(ds_C2H2_1) + list(ds_C2H2_2))

    file_T_H_S_C2H2 = open('T_H_S_C2H2.dat', 'w')

    file_T_H_S_C2H2.write("# ACETYLENE (C2H2) \n")
    file_T_H_S_C2H2.write("# Temperature\tdH\tH\tS\n")

    for i in np.arange(26):
        file_T_H_S_C2H2.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_C2H2[i]) + "\t" + str(h_C2H2[i]) + "\t" + str(s_C2H2[i]) + "\n")

    file_T_H_S_C2H2.close()

    #TiO2

    # 500-3900 K temperature range

    dH_TiO2_1, s_TiO2_1 = np.loadtxt("TiO2_nist.dat", comments= '#', delimiter ='\t', usecols = (1, 2) , unpack = True)

    h_TiO2_1 = dH_TiO2_1 + Hf_29815_TiO2
    # ds_TiO2_1 = s_TiO2_1 - s_29815_TiO2

    '''

    T = np.linspace(4000, 6000, 21)  # degrees K
    t = T / 1000

    # 4000-6000 K valid temperature range
    A = 63.82818
    B = -4.418178
    C = 1.080707
    D = -0.058816
    E = -5.216235
    F = -336.0739
    G = 323.0094
    H = -305.4324

    dH_TiO2_2 = dH(A, B, C, D, E, F, H, t)
    h_TiO2_2 = dH_TiO2_2 + Hf_29815_TiO2
    s_TiO2_2 = s(A, B, C, D, E, G, t)
    # ds_TiO2_2 = s_TiO2_2 - s_29815_TiO2

    dH_TiO2 = np.array(list(dH_TiO2_1) + list(dH_TiO2_2))
    h_TiO2 = np.array(list(h_TiO2_1) + list(h_TiO2_2))
    s_TiO2 = np.array(list(s_TiO2_1) + list(s_TiO2_2))
    # ds_TiO2 = np.array(list(ds_TiO2_1) + list(ds_TiO2_2))

    '''

    dH_TiO2 = np.array(list(dH_TiO2_1))[:26]
    h_TiO2 = np.array(list(h_TiO2_1))[:26]
    s_TiO2 = np.array(list(s_TiO2_1))[:26]
    # ds_TiO2 = np.array(list(ds_TiO2_1))[:26]

    file_TiO2 = open("TiO2.dat", "w")
    file_TiO2.write("# TITANIUM DIOXIDE (TiO2)\n")
    file_TiO2.write("# Temperature\tdH\tH\tS\n")
    for i in np.arange(26):
        file_TiO2.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_TiO2[i]) + "\t" + str(h_TiO2[i]) + "\t" + str(s_TiO2[i]) + "\n")
    file_TiO2.close()

    #TiO

    # 500-4400 K temperature range

    dH_TiO_1, s_TiO_1 = np.loadtxt("TiO_nist.dat", comments='#', delimiter='\t', usecols=(1, 2), unpack=True)


    h_TiO_1 = dH_TiO_1 + Hf_29815_TiO
    # ds_TiO_1 = s_TiO_1 - s_29815_TiO

    '''

    T = np.linspace(4500, 6000, 21)  # degrees K
    t = T / 1000

    # 4500-6000 K valid temperature range
    A = 36.25740
    B = -2.704541
    C = 1.691450
    D = -0.151753
    E = 9.805701
    F = 51.83580
    G = 281.4380
    H = 54.39200

    dH_TiO_2 = dH(A, B, C, D, E, F, H, t)
    h_TiO_2 = dH_TiO_2 + Hf_29815_TiO
    s_TiO_2 = s(A, B, C, D, E, G, t)
    # ds_TiO_2 = s_TiO_2 - s_29815_TiO

    dH_TiO = np.array(list(dH_TiO_1) + list(dH_TiO_2))
    h_TiO = np.array(list(h_TiO_1) + list(h_TiO_2))
    s_TiO = np.array(list(s_TiO_1) + list(s_TiO_2))
    # ds_TiO = np.array(list(ds_TiO_1) + list(ds_TiO_2))

    '''

    dH_TiO = np.array(list(dH_TiO_1))[:26]
    h_TiO = np.array(list(h_TiO_1))[:26]
    s_TiO = np.array(list(s_TiO_1))[:26]
    # ds_TiO = np.array(list(ds_TiO_1))[:26]

    file_TiO = open("TiO.dat", "w")
    file_TiO.write("# TITANIUM OXIDE (TiO)\n")
    file_TiO.write("# Temperature\tdH\tH\tS\n")
    for i in np.arange(26):
        file_TiO.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_TiO[i]) + "\t" + str(h_TiO[i]) + "\t" + str(s_TiO[i]) + "\n")
    file_TiO.close()

    # Ti

    # 500-3600 K temperature range

    dH_Ti_1, s_Ti_1 = np.loadtxt("Ti_nist.dat", comments='#', delimiter='\t', usecols=(1, 2), unpack=True)

    h_Ti_1 = dH_Ti_1 + Hf_29815_Ti
    # ds_Ti_1 = s_Ti_1 - s_29815_Ti

    '''

    T = np.linspace(3700, 6000, 24)  # degrees K
    t = T / 1000

    # 3630-6000 K valid temperature range
    A = 9.274255
    B = 6.092113
    C = 0.577095
    D = -0.110364
    E = 6.504405
    F = 483.0093
    G = 204.1566
    H = 473.6288

    dH_Ti_2 = dH(A, B, C, D, E, F, H, t)
    h_Ti_2 = dH_Ti_2 + Hf_29815_Ti
    s_Ti_2 = s(A, B, C, D, E, G, t)
    # ds_Ti_2 = s_Ti_2 - s_29815_Ti

    dH_Ti = np.array(list(dH_Ti_1) + list(dH_Ti_2))
    h_Ti = np.array(list(h_Ti_1) + list(h_Ti_2))
    s_Ti = np.array(list(s_Ti_1) + list(s_Ti_2))
    # ds_Ti = np.array(list(ds_Ti_1) + list(ds_Ti_2))

    '''

    dH_Ti = np.array(list(dH_Ti_1))[:26]
    h_Ti = np.array(list(h_Ti_1))[:26]
    s_Ti = np.array(list(s_Ti_1))[:26]
    # ds_Ti = np.array(list(ds_Ti_1))[:26]

    file_Ti = open("Ti.dat", "w")
    file_Ti.write("# TITANIUM (Ti)\n")
    file_Ti.write("# Temperature\tdH\tH\tS\n")
    for i in np.arange(26):
        file_Ti.write(
            str(np.linspace(500, 3000, 26)[i]) + "\t" + str(dH_Ti[i]) + "\t" + str(h_Ti[i]) + "\t" + str(s_Ti[i]) + "\n")
    file_Ti.close()

    T = np.linspace(500, 3000, 26)

    plt.figure()
    plt.plot(T,dH_H2, label='$\Delta H_{H_2}$')
    plt.plot(T,dH_CH4, label='$\Delta H_{CH_4}$')
    plt.plot(T,dH_H2O, label='$\Delta H_{H_2O}$')
    plt.plot(T,dH_CO, label='$\Delta H_{CO}$')
    plt.plot(T,dH_CO2, label='$\Delta H_{CO_2}$')
    plt.plot(T,dH_C2H2, label='$\Delta H_{C_2H_2}$')
    plt.plot(T,dH_Ti, label='$\Delta H_{Ti}$')
    plt.plot(T,dH_TiO, label='$\Delta H_{TiO}$')
    plt.plot(T,dH_TiO2, label='$\Delta H_{TiO_2}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('$\Delta H \; (kJ/mol)$') # H-H_298
    plt.legend( loc='best')
    plt.savefig("enthalpies-NIST-molecules.pdf")

    plt.figure()
    plt.plot(T,s_H2, label='$S_{H_2}$')
    plt.plot(T,s_CH4, label='$S_{CH_4}$')
    plt.plot(T,s_H2O, label='$S_{H_2O}$')
    plt.plot(T,s_CO, label='$S_{CO}$')
    plt.plot(T,s_CO2, label='$S_{CO_2}$')
    plt.plot(T,s_C2H2, label='$S_{C_2H_2}$')
    plt.plot(T,s_Ti, label='$S_{Ti}$')
    plt.plot(T,s_TiO, label='$S_{TiO}$')
    plt.plot(T,s_TiO2, label='$S_{TiO_2}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('$S \; (J \; mol^{-1} \; K^{-1})$')
    plt.legend( loc='best')
    plt.savefig("entropies-NIST-molecules.pdf")


    '''
    Reactions
    '''

    T = np.linspace(500, 3000, 26)

    # reaction 1: CH4 + H2O <-> CO + 3 H2

    print("# Reaction 1: CH4 + H2O <-> CO + 3 H2")

    Hrxn1_29815 = Hrxn_29815(Hf_29815_CH4, Hf_29815_H2O, Hf_29815_CO, 3 * Hf_29815_H2)
    Srxn1_29815 = Srxn_29815(s_29815_CH4, s_29815_H2O, s_29815_CO, 3 * s_29815_H2)
    Grxn1_29815 = Grxn_29815(Hrxn1_29815, Srxn1_29815)

    Hrxn1 = Hrxn1_29815 + dH_CO + 3 * dH_H2 - dH_CH4 - dH_H2O
    Grxn1 = Hrxn1 - T * (s_CO + 3 * s_H2 - s_CH4 - s_H2O) / 1000

    #print(Grxn1)

    plt.figure()
    plt.plot(T,Grxn1, label='$\Delta G_{rxn1}$')
    plt.plot(T,Hrxn1, label='$\Delta H_{rxn1}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('(kJ/mol)')
    plt.legend( loc='best')
    plt.savefig("abundances-nist-1.pdf")
    #plt.show()


    # Equilibrium constant calculation (K')

    K1_1bar = (P_0 / P_1) ** 2 * np.exp(-Grxn1 / (R * T))
    K1_P_wanted = (P_0 / P_wanted) ** 2 * np.exp(-Grxn1 / (R * T))

    print("Ready :D! \n")

    '''

    plt.figure()
    plt.plot(T,K1_1bar,label='$K1_{1bar}$')
    plt.plot(T,K1_P_wanted,label='$K1_{P_wanted}$')
    plt.xlim([500, 3000])
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Equilibrium constant rxn1')
    plt.legend( loc='best')
    plt.savefig('abundances-nist-1-K.pdf')
    #plt.show()

    '''

    # reaction 2: CO2 + H2 <-> CO + H2O

    print("#reaction 2: CO2 + H2 <-> CO + H2O")

    Hrxn2_29815 = Hrxn_29815(Hf_29815_CO2, Hf_29815_H2, Hf_29815_CO, Hf_29815_H2O)
    Srxn2_29815 = Srxn_29815(s_29815_CO2, s_29815_H2, s_29815_CO, s_29815_H2O)
    Grxn2_29815 = Grxn_29815(Hrxn2_29815, Srxn2_29815)

    Hrxn2 = Hrxn2_29815 + dH_CO + dH_H2O - dH_CO2 - dH_H2
    Grxn2 = Hrxn2 - T * (s_CO + s_H2O - s_CO2 - s_H2) / 1000

    #print(Grxn2)

    plt.figure()
    plt.plot(T,Grxn2, label='$\Delta G_{rxn2}$')
    plt.plot(T,Hrxn2, label='$\Delta H_{rxn2}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('(kJ/mol)')
    plt.legend( loc='best')
    plt.savefig("abundances-nist-2.pdf")
    #plt.show()


    # Equilibrium constant calculation (K')

    K2_sin_presion = np.exp(-Grxn2 / (R * T))

    print("Ready :D! \n")

    '''

    plt.figure()
    plt.plot(T,K2_1bar,label='$K´_{2}(1bar)$')
    plt.plot(T,K2_P_wanted,label='$K´_{2}(P_wanted)$',ls='--')
    plt.plot(T,K2_sin_presion,label='$K´_{2}$',ls='--')
    plt.xlim([500, 3000])
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Equilibrium constant rxn2')
    plt.legend( loc='best')
    plt.savefig('abundances-nist-2-K.pdf')
    #plt.show()

    plt.figure()
    plt.plot(T,1/K2_sin_presion,label='$1/K´_{2}$',color='r')
    plt.xlim([500, 3000])
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('$1/K´_{2}$')
    plt.legend( loc='best')
    plt.savefig('abundances-nist-fig1.pdf')
    #plt.show()

    '''

    # reaction 3: 2 CH4 <-> C2H2 + 3 H2

    print("#reaction 3: 2 CH4 <-> C2H2 + 3 H2")

    Hrxn3_29815 = Hrxn_29815(2. * Hf_29815_CH4, 0.0, Hf_29815_C2H2, 3. * Hf_29815_H2)
    Srxn3_29815 = Srxn_29815(2. * s_29815_CH4, 0.0, s_29815_C2H2, 3. * s_29815_H2)
    Grxn3_29815 = Grxn_29815(Hrxn3_29815, Srxn3_29815)

    Hrxn3 = Hrxn3_29815 + dH_C2H2 + 3. * dH_H2 - 2. * dH_CH4
    Grxn3 = Hrxn3 - T * (s_C2H2 + 3. * s_H2 - 2. * s_CH4) / 1000.

    #print(Grxn3)

    plt.figure()
    plt.plot(T,Grxn3, label='$\Delta G_{rxn3}$')
    plt.plot(T,Hrxn3, label='$\Delta H_{rxn3}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('(kJ/mol)')
    plt.legend( loc='best')
    plt.savefig("abundances-nist-3.pdf")
    plt.show()


    # Equilibrium constant calculation (K')

    K3_1bar = (P_0 / P_1) ** 2. * np.exp(-Grxn3 / (R * T))
    K3_P_wanted = (P_0 / P_wanted) ** 2. * np.exp(-Grxn3 / (R * T))

    print("Ready :D! \n")

    '''

    plt.figure()
    plt.plot(T,K3_1bar,label='$K3_{1bar}$')
    plt.plot(T,K3_P_wanted,label='$K3_{P_wanted}$')
    plt.xlim([500, 3000])
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Equilibrium constant rxn3')
    plt.legend( loc='best')
    plt.savefig('abundances-nist-3-K.pdf')
    #plt.show()

    '''

    '''

    # This generates a graphic of the constants (K1, K2 and K3) with different pressures

    plt.figure()
    plt.plot(T,K1_1bar,label='$K´_{1}(1bar)$',color='orange',ls='-',linewidth=1)
    plt.plot(T,K1_P_wanted,label='$K´_{1}('+P_wanted_str+'bar)$',color='orange',ls='-',linewidth=3)
    plt.plot(T,K2_sin_presion,label='$K´_{2}$',color='r',ls='-.')
    plt.plot(T,K3_1bar,label='$K´_{3}(1bar)$',color='k',ls='--',linewidth=1)
    plt.plot(T,K3_P_wanted,label='$K´_{3}('+P_wanted_str+'bar)$',color='k',ls='--',linewidth=3)
    plt.xlim([500, 3000])
    plt.ylim([10**-22, 10**14])
    plt.yscale('log')
    plt.xticks(np.linspace(500,3000,6,endpoint=True))
    plt.yticks(10**(np.linspace(-22,14,13)))
    plt.tick_params(direction='in',bottom='on', top='on', left='on', right='on')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Normalised Equilibrium Constants')
    plt.legend( loc='best')
    plt.savefig('abundances-nist_P_'+P_wanted_str+'.pdf')
    plt.show()

    '''

    # reaction 4: TiO2 + H2 <-> TiO + H2O

    print("#reaction 4: TiO2 + H2 <-> TiO + H2O")

    Hrxn4_29815 = Hrxn_29815(Hf_29815_TiO2, Hf_29815_H2, Hf_29815_TiO, Hf_29815_H2O)
    Srxn4_29815 = Srxn_29815(s_29815_TiO2, s_29815_H2, s_29815_TiO, s_29815_H2O)
    Grxn4_29815 = Grxn_29815(Hrxn4_29815, Srxn4_29815)

    Hrxn4 = Hrxn4_29815 + dH_TiO + dH_H2O - dH_TiO2 - dH_H2
    Grxn4 = Hrxn4 - T * (s_TiO + s_H2O - s_TiO2 - s_H2) / 1000.


    #print(Grxn4)

    plt.figure()
    plt.plot(T,Grxn4, label='$\Delta G_{rxn4}$')
    plt.plot(T,Hrxn4, label='$\Delta H_{rxn4}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('(kJ/mol)')
    plt.legend( loc='best')
    plt.savefig("abundances-nist-4.pdf")
    plt.show()



    # Equilibrium constant calculation (K')

    K4_sin_presion = np.exp(-Grxn4 / (R * T))

    print("Ready :D! \n")

    # reaction 5: TiO + H2 <-> Ti + H2O

    print("#reaction 5: TiO + H2 <-> Ti + H2O")

    Hrxn5_29815 = Hrxn_29815(Hf_29815_TiO, Hf_29815_H2, Hf_29815_Ti, Hf_29815_H2O)
    Srxn5_29815 = Srxn_29815(s_29815_TiO, s_29815_H2, s_29815_Ti, s_29815_H2O)
    Grxn5_29815 = Grxn_29815(Hrxn5_29815, Srxn5_29815)

    Hrxn5 = Hrxn5_29815 + dH_Ti + dH_H2O - dH_TiO - dH_H2
    Grxn5 = Hrxn5 - T * (s_Ti + s_H2O - s_TiO - s_H2) / 1000.

    #print(Grxn5)

    plt.figure()
    plt.plot(T,Grxn5, label='$\Delta G_{rxn5}$')
    plt.plot(T,Hrxn5, label='$\Delta H_{rxn5}$')
    plt.xlabel('Temperature (K)')
    plt.ylabel('(kJ/mol)')
    plt.legend( loc='best')
    plt.savefig("abundances-nist-5.pdf")
    plt.show()

    # Equilibrium constant calculation (K')

    K5_sin_presion = np.exp(-Grxn5 / (R * T))

    print("Ready :D! \n")

    # This generates a graphic of the constants (K1, K2, K3, K4 and K5) with different pressures

    plt.figure()
    plt.plot(T,K1_1bar,label='$K´_{1}(1bar)$',color='orange',ls='-',linewidth=1)
    plt.plot(T,K1_P_wanted,label='$K´_{1}('+P_wanted_str+'bar)$',color='orange',ls='-',linewidth=3)
    plt.plot(T,K2_sin_presion,label='$K´_{2}$',color='r',ls='-.')
    plt.plot(T,K3_1bar,label='$K´_{3}(1bar)$',color='k',ls='--',linewidth=1)
    plt.plot(T,K3_P_wanted,label='$K´_{3}('+P_wanted_str+'bar)$',color='k',ls='--',linewidth=3)
    plt.plot(T,K4_sin_presion,label='$K´_{4}$',color='g',ls='-.',linewidth=1)
    plt.plot(T,K5_sin_presion,label='$K´_{5}$',color='c',ls=':',linewidth=1)
    plt.xlim([500, 3000])
    plt.ylim([10**-22, 10**14])
    plt.yscale('log')
    plt.xticks(np.linspace(500,3000,6,endpoint=True))
    plt.yticks(10**(np.linspace(-22,14,13)))
    plt.tick_params(direction='in',bottom='on', top='on', left='on', right='on')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Normalised Equilibrium Constants')
    plt.legend( loc='best', ncol=2)
    plt.savefig('abundances-nist_P_'+P_wanted_str+'_extra.pdf')

    return K1_1bar, K1_P_wanted, K2_sin_presion, K3_1bar, K3_P_wanted, K4_sin_presion, K5_sin_presion


def abundances_norm_H(CO_wanted, P_wanted_str, selection):
    n_C_wanted = (CO_wanted / rate_CO_solar) * n_C
    print("rate_CO_solar = ", rate_CO_solar)
    print("n_C = ", n_C)
    print("n_C_wanted = ", n_C_wanted)

    if selection == '1':
        # METHANE

        T = np.linspace(500, 3000, 26)
        count_1bar = 0
        count_P_wanted = 0

        file_roots_calculation = open('roots_calculation.dat', 'w')

        for j in np.arange(2):
            A = []
            if j == 0:
                print("P = 1 bar")
                K1 = K1_1bar
                K2 = K2_sin_presion
                K3 = K3_1bar
                K4 = K4_sin_presion
                K5 = K5_sin_presion
                n_CH4_1bar = []
            if j == 1:
                print('P=' + P_wanted_str)
                K1 = K1_P_wanted
                K2 = K2_sin_presion
                K3 = K3_P_wanted
                K4 = K4_sin_presion
                K5 = K5_sin_presion
                n_CH4_P_wanted = []

            '''
            # This correspond to the coefficients obtained by Heng et al.:
            A.append(-2.0 * n_C_wanted)
            A.append(8.0 * (K1 / K2) * (n_O - n_C_wanted) ** 2.0 + 1.0 + 2.0 * K1 * (n_O - n_C_wanted))
            A.append(8.0 * (K1 / K2) * (n_O - n_C_wanted) + 2.0 * K3 + K1)
            A.append(2.0 * (K1 / K2) * (1.0 + 8.0 * K3 * (n_O - n_C_wanted)) + 2.0 * K1 * K3)
            A.append(8.0 * (K1 * K3 / K2))
            A.append(8.0 * (K1 * (K3 ** 2.0) / K2))
            '''
            # This correspond to the coefficients obtained by us:
            A.append(-2.0 * n_C_wanted)
            A.append(4.0 * (K1 / K2) * (n_O - n_C_wanted) ** 2.0 + 1.0 + 2.0 * K1 * (n_O - n_C_wanted))
            A.append(4.0 * (K1 / K2) * (n_O - n_C_wanted) + 2.0 * K3 + K1)
            A.append((K1 / K2) * (1.0 + 8.0 * K3 * (n_O - n_C_wanted)) + 2.0 * K1 * K3)
            A.append(4.0 * (K1 * K3 / K2))
            A.append(4.0 * (K1 * (K3 ** 2.0) / K2))

            for i in np.arange(26):
                file_roots_calculation.write("# Coefficients (A)")
                file_roots_calculation.write(
                    '\n' + str(A[0]) + '|' + str(A[1][i]) + '|' + str(A[2][i]) + '|' + str(A[3][i]) + '|' + str(
                        A[4][i]) + '|' + str(A[5][i]))
                roots = poly.polyroots([A[0], A[1][i], A[2][i], A[3][i], A[4][i], A[5][i]])
                file_roots_calculation.write("\n# Roots for T=" + str(T[i]))
                for w in np.arange(len(roots)):
                    two_n_C = float(-A[0])  # We know that n_CH4 should be approx. 2*n_C
                    possible_n_CH4 = roots[w]
                    file_roots_calculation.write("\nPossible ñ_CH4 (root " + str(w + 1) + "): " + str(possible_n_CH4))
                    file_roots_calculation.write("\n2 ñ_C (We know that ñ_CH4 should be approx. this): " + str(two_n_C))
                    if ('{0:.1f}'.format(possible_n_CH4.real) == '{0:.1f}'.format(two_n_C) or (
                                    possible_n_CH4.real <= two_n_C and possible_n_CH4.real > 0.0)) and possible_n_CH4.imag == 0.0:
                        file_roots_calculation.write(
                            "\n~~~~~ MATCH 1313 ~~~~~(T=" + str(T[i]) + ") | Possible_n_CH4.real=" + str(
                                possible_n_CH4.real) + " | Two_n_C=" + '{0:.6f}'.format(two_n_C) + '\n')
                        if j == 0:
                            n_CH4_1bar.append(possible_n_CH4.real)
                            count_1bar += 1  # This number should be 26 at the end of the loops (because we have 26 temperatures)
                        if j == 1:
                            n_CH4_P_wanted.append(possible_n_CH4.real)
                            count_P_wanted += 1

        print("***count_1bar*** = ", count_1bar)
        file_roots_calculation.write("\n***count_1bar*** = " + str(count_1bar))
        print("***count_P_wanted*** = ", count_P_wanted)
        file_roots_calculation.write("\n***count_P_wanted*** = " + str(count_P_wanted))
        print("(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))")
        file_roots_calculation.write(
            "\n(These two numbers should be 26 at the end of the loops (because we have 26 temperatures))\n")

        file_roots_calculation.close()

        # METHANE

        n_CH4_1bar = np.array(n_CH4_1bar)
        n_CH4_P_wanted = np.array(n_CH4_P_wanted)

        # ACETYLENE

        n_C2H2_1bar = K3_1bar * n_CH4_1bar ** 2.0
        n_C2H2_P_wanted = K3_P_wanted * n_CH4_P_wanted ** 2.0

        # WATER

        n_H2O_1bar = (2.0 * K3_1bar * n_CH4_1bar ** 2.0) + n_CH4_1bar + (2.0 * (n_O - n_C_wanted))
        n_H2O_P_wanted = (2.0 * K3_P_wanted * n_CH4_P_wanted ** 2.0) + n_CH4_P_wanted + (2.0 * (n_O - n_C_wanted))

        # CARBON MONOXIDE

        n_CO_1bar = K1_1bar * n_CH4_1bar * n_H2O_1bar
        n_CO_P_wanted = K1_P_wanted * n_CH4_P_wanted * n_H2O_P_wanted

        # CARBON DIOXIDE

        n_CO2_1bar = n_CO_1bar * n_H2O_1bar / K2_sin_presion
        n_CO2_P_wanted = n_CO_P_wanted * n_H2O_P_wanted / K2_sin_presion

        # SAVE DATA

        file_abundances_1bar = open('abu_heng_1bar.dat','w')
        file_abundances_P = open('abu_heng_Pwanted.dat','w')

        file_abundances_1bar.write('# P = '+str(P_1)+'\t C/O = '+CO_wanted_str+'\n')
        file_abundances_1bar.write('# T(K)\tCH4\tC2H2\tH2O\tCO\tCO2\n')

        file_abundances_P.write('# P = '+P_wanted_str+'\t C/O = '+CO_wanted_str+'\n')
        file_abundances_P.write('# T(K)\tCH4\tC2H2\tH2O\tCO\tCO2\n')

        for i in range(len(T)):
            file_abundances_1bar.write(str(T[i])+'\t'+str(n_CH4_1bar[i])+'\t'+str(n_C2H2_1bar[i])+'\t'+str(n_H2O_1bar[i])+'\t'+str(n_CO_1bar[i])+'\t'+str(n_CO2_1bar[i])+'\n')
        for i in range(len(T)):
            file_abundances_P.write(str(T[i]) + '\t' + str(n_CH4_P_wanted[i]) + '\t' + str(n_C2H2_P_wanted[i]) + '\t' + str(
                n_H2O_P_wanted[i]) + '\t' + str(n_CO_P_wanted[i]) + '\t' + str(n_CO2_P_wanted[i]) + '\n')

        file_abundances_1bar.close()
        file_abundances_P.close()


        return n_CH4_1bar, n_CH4_P_wanted, n_C2H2_1bar, n_C2H2_P_wanted, n_H2O_1bar, n_H2O_P_wanted, n_CO_1bar, n_CO_P_wanted, n_CO2_1bar, n_CO2_P_wanted

    if selection == '2':

        file_roots_calculation_selection2 = open('roots_calculation_selection2.dat', 'w')

        # METHANE

        K1 = K1_P_wanted
        K2 = K2_sin_presion
        K3 = K3_P_wanted

        print("K1=", K1)
        print("K2=", K2)
        print("K3=", K3)

        A0 = -2.0 * n_C_wanted
        A1 = 8.0 * (K1 / K2) * (n_O - n_C_wanted) ** 2.0 + 1.0 + 2.0 * K1 * (n_O - n_C_wanted)
        A2 = 8.0 * (K1 / K2) * (n_O - n_C_wanted) + 2.0 * K3 + K1
        A3 = 2.0 * (K1 / K2) * (1.0 + 8.0 * K3 * (n_O - n_C_wanted)) + 2.0 * K1 * K3
        A4 = 8.0 * (K1 * K3 / K2)
        A5 = 8.0 * (K1 * (K3 ** 2.0) / K2)

        file_roots_calculation_selection2.write("# Coefficients (A)")
        file_roots_calculation_selection2.write(
            '\n' + str(A0) + '|' + str(A1) + '|' + str(A2) + '|' + str(A3) + '|' + str(A4) + '|' + str(A5))
        roots = poly.polyroots([A0, A1, A2, A3, A4, A5])
        file_roots_calculation_selection2.write("\n# Roots for T=" + T_wanted_str)

        for w in np.arange(len(roots)):
            two_n_C = float(- A0)  # We know that n_CH4 should be approx. 2*n_C
            possible_n_CH4 = roots[w]
            file_roots_calculation_selection2.write("\nPossible ñ_CH4 (root " + str(w + 1) + "): " + str(possible_n_CH4))
            file_roots_calculation_selection2.write("\n2 ñ_C (We know that ñ_CH4 should be approx. this): " + str(two_n_C))
            if ('{0:.1f}'.format(possible_n_CH4.real) == '{0:.1f}'.format(two_n_C) or (
                    possible_n_CH4.real <= two_n_C and possible_n_CH4.real > 0.0)) and possible_n_CH4.imag == 0.0:
                file_roots_calculation_selection2.write(
                    "\n~~~~~ MATCH 1313 ~~~~~(T=" + T_wanted_str + ") | Possible_n_CH4.real=" + str(
                        possible_n_CH4.real) + " | Two_n_C=" + '{0:.6f}'.format(two_n_C) + '\n')
                n_CH4_P_wanted = possible_n_CH4.real

        file_roots_calculation_selection2.close()

        file_abundances_final_selection2 = open(
            'abundances_final_selection2_CO=' + str(int(CO_wanted)) + '_P_' + P_wanted_str + '_T_' + T_wanted_str + '.dat', "w")

        # METHANE

        file_abundances_final_selection2.write("# METHANE \n")
        file_abundances_final_selection2.write(
            str(n_CH4_P_wanted) + "| T = " + T_wanted_str + "| P = " + P_wanted_str + " | CO = " + CO_wanted_str + "\n")

        # ACETYLENE

        file_abundances_final_selection2.write("# ACETYLENE \n")

        n_C2H2_P_wanted = K3_P_wanted * n_CH4_P_wanted ** 2.0

        file_abundances_final_selection2.write(
            str(n_C2H2_P_wanted) + "| T = " + T_wanted_str + "| P = " + P_wanted_str + " | CO = " + CO_wanted_str + "\n")

        # WATER

        file_abundances_final_selection2.write("# WATER \n")

        n_H2O_P_wanted = (2.0 * K3_P_wanted * n_CH4_P_wanted ** 2.0) + n_CH4_P_wanted + (2.0 * (n_O - n_C_wanted))

        file_abundances_final_selection2.write(
            str(n_H2O_P_wanted) + "| T = " + T_wanted_str + "| P = " + P_wanted_str + " | CO = " + CO_wanted_str + "\n")

        # CARBON MONOXIDE

        file_abundances_final_selection2.write("# CARBON MONOXIDE \n")

        n_CO_P_wanted = K1_P_wanted * n_CH4_P_wanted * n_H2O_P_wanted

        file_abundances_final_selection2.write(
            str(n_CO_P_wanted) + "| T = " + T_wanted_str + "| P = " + P_wanted_str + " | CO = " + CO_wanted_str + "\n")

        # CARBON DIOXIDE

        file_abundances_final_selection2.write("# CARBON DIOXIDE \n")

        n_CO2_P_wanted = n_CO_P_wanted * n_H2O_P_wanted / K2_sin_presion

        file_abundances_final_selection2.write(
            str(n_CO2_P_wanted) + "| T = " + T_wanted_str + "| P = " + P_wanted_str + " | CO = " + CO_wanted_str + "\n")

        file_abundances_final_selection2.close()

        print("\nThe abundances for T="+T_wanted_str+", P="+P_wanted_str+" and C/O="+CO_wanted_str+" are:\n")
        print("ñ_CH4 = ", n_CH4_P_wanted)
        print("ñ_C2H2 = ", n_C2H2_P_wanted)
        print("ñ_H2O = ", n_H2O_P_wanted)
        print("ñ_CO = ", n_CO_P_wanted)
        print("ñ_CO2 = ", n_CO2_P_wanted)
        print("\nYou can find them in the file 'abundances_final_selection2_CO=" + str(float(CO_wanted)) + "_P_" + P_wanted_str + "_T_" + T_wanted_str + ".dat' \n")

        return n_CH4_P_wanted, n_C2H2_P_wanted, n_H2O_P_wanted, n_CO_P_wanted, n_CO2_P_wanted

    if selection == '3':

        # WATER

        T = np.linspace(500, 3000, 26)

        for j in np.arange(2):
            A = []
            if j == 0:
                print("P = 1 bar")
                K1 = K1_1bar
                K2 = K2_sin_presion
                K3 = K3_1bar
                K4 = K4_sin_presion
                K5 = K5_sin_presion
            if j == 1:
                print('P = ' + P_wanted_str + ' bar')
                K1 = K1_P_wanted
                K2 = K2_sin_presion
                K3 = K3_P_wanted
                K4 = K4_sin_presion
                K5 = K5_sin_presion

            C1 = -1.
            C2 = (2. * n_O - K4 - 4. * n_Ti)
            C3 = (2. * n_O * K4 - K4 * K5 - 2. * n_Ti * K4)
            C4 = 2. * n_O * K4 * K5
            C5 = 2. * K1 / K2
            C6 = (K1 + 2. * K1 * K4 / K2)
            C7 = (K1 * K4 + 2. * K1 * K4 * K5 / K2)
            C8 = K1 * K4 * K5

            A.append(2. * K2 * K3 * C4**2.) # A_0
            A.append(4. * K2 * K3 * C3 * C4 + K2 * C4 * C8) # A_1
            A.append(2. * K2 * K3 * (2. * C2 * C4 + C3**2.) + K2 * (C4 * C7 + C3 * C8) + K1 * K2 * C4 * C8 -  2. * K2 * n_C_wanted * C8**2.) # A_2
            A.append(2. * K2 * K3 * (2. * C1 * C4 + 2. * C2 * C3) + K2 * (C4 * C6 + C3 * C7 + C2 * C8) + K1 * K2 * (C4 * C7 + C3 * C8) + K1 * C4 * C8 -  4. * K2 * n_C_wanted * C7 * C8) # A_3
            A.append(2. * K2 * K3 * (2. * C1 * C3 + C2**2.) + K2 * (C4 * C5 + C3 * C6 + C2 * C7 + C1 * C8) + K1 * K2 * (C4 * C6 + C3 * C7 + C2 * C8) + K1 * (C4 * C7 + C3 * C8) - 2. * K2 * n_C_wanted * (2. * C6 * C8 + C7**2.)) # A_4
            A.append(4. * K2 * K3 * C1 * C2 + K2 * (C3 * C5 + C2 * C6 + C1 * C7) + K1 * K2 * (C4 * C5 + C3 * C6 + C2 * C7 + C1 * C8) + K1 * (C4 * C6 + C3 * C7 + C2 * C8) - 2. * K2 * n_C_wanted * (2. * C5 * C8 + 2. * C6 * C7)) # A_5
            A.append(2. * K2 * K3 * C1**2. + K2 * (C2 * C5 + C1 * C6) + K1 * K2 * (C3 * C5 + C2 * C6 + C1 * C7) + K1 * (C4 * C5 + C3 * C6 + C2 * C7 + C1 * C8) - 2. * K2 * n_C_wanted * (2. * C5 * C7 +  C6**2.)) # A_6
            A.append(K2 * C1 * C5 + K1 * K2 * (C2 * C5 + C1 * C6) + K1 * (C3 * C5 + C2 * C6 + C1 * C7) -  4. * K2 * n_C_wanted * C5 * C6) # A_7
            A.append(K1 * K2 * C1 * C5 + K1 * C2 * C5 + K1 * C1 * C6 - 2. * K2 * n_C_wanted * C5**2.) # A_8
            A.append(K1 * C1 * C5) # A_9

            if j == 0:
                n_H2O_1bar = root_searcher_3(A, T, j)
            if j == 1:
                n_H2O_P_wanted = root_searcher_3(A, T, j)


        # WATER

        n_H2O_1bar = np.array(n_H2O_1bar)
        n_H2O_P_wanted = np.array(n_H2O_P_wanted)

        # TITANIUM

        nn_Ti_1bar = (n_Ti * K4 * K5) / (n_H2O_1bar**2. + K4 * n_H2O_1bar + K4 * K5)
        nn_Ti_P_wanted = (n_Ti * K4 * K5) / (n_H2O_P_wanted**2. + K4 * n_H2O_P_wanted + K4 * K5)

        # TITANIUM OXIDE

        n_TiO_1bar = 2. * (nn_Ti_1bar*n_H2O_1bar)/(K5)
        n_TiO_P_wanted = 2. * (nn_Ti_P_wanted*n_H2O_P_wanted)/(K5)

        # TITANIUM DIOXIDE

        n_TiO2_1bar = 2. * (nn_Ti_1bar * n_H2O_1bar**2.)/(K5 * K4)
        n_TiO2_P_wanted = 2. * (nn_Ti_P_wanted * n_H2O_P_wanted**2.)/(K5 * K4)

        # METHANE

        n_CH4_1bar = (- n_H2O_1bar**3. + (2.*n_O - K4 - 4.*n_Ti) * n_H2O_1bar**2. + (2.*n_O*K4 - K4*K5 - 2.*n_Ti*K4) * n_H2O_1bar \
                    + 2.*n_O*K4*K5 )/(2.*K1_1bar/K2 * n_H2O_1bar**4. + (K1_1bar + 2.*K1_1bar*K4/K2) * n_H2O_1bar**3. \
                    + (K1_1bar*K4 + 2.*K1_1bar*K4*K5/K2) * n_H2O_1bar**2. + K1_1bar*K4*K5 * n_H2O_1bar)
        n_CH4_P_wanted = (- n_H2O_P_wanted**3. + (2.*n_O - K4 - 4.*n_Ti) * n_H2O_P_wanted**2. \
                        + (2.*n_O*K4 - K4*K5 - 2.*n_Ti*K4) * n_H2O_P_wanted + 2.*n_O*K4*K5 )/(2.*K1_P_wanted/K2 * n_H2O_P_wanted**4. \
                        + (K1_P_wanted + 2.*K1_P_wanted*K4/K2) * n_H2O_P_wanted**3. \
                        + (K1_P_wanted*K4 + 2.*K1_P_wanted*K4*K5/K2) * n_H2O_P_wanted**2. + K1_P_wanted*K4*K5 * n_H2O_P_wanted)

        # CARBON MONOXIDE

        n_CO_1bar = K1_1bar * n_CH4_1bar * n_H2O_1bar
        n_CO_P_wanted = K1_P_wanted * n_CH4_P_wanted * n_H2O_P_wanted

        # CARBON DIOXIDE

        n_CO2_1bar = n_CO_1bar * n_H2O_1bar / K2
        n_CO2_P_wanted = n_CO_P_wanted * n_H2O_P_wanted / K2

        # ACETYLENE

        n_C2H2_1bar = K3_1bar * n_CH4_1bar**2.
        n_C2H2_P_wanted = K3_P_wanted * n_CH4_P_wanted**2.


        # SAVE DATA

        file_abundances_1bar = open('abu_Heng+Ti_1bar.dat','w')
        file_abundances_P = open('abu_Heng+Ti_P_'+P_wanted_str+'.dat','w')

        file_abundances_1bar.write('# P = '+str(P_1)+'\t C/O = '+CO_wanted_str+'\n')
        file_abundances_1bar.write('# T(K)\tCH4\tC2H2\tH2O\tCO\tCO2\tTiO\tTiO2\tTi\n')

        file_abundances_P.write('# P = '+P_wanted_str+'\t C/O = '+CO_wanted_str+'\n')
        file_abundances_P.write('# T(K)\tCH4\tC2H2\tH2O\tCO\tCO2\tTiO\tTiO2\tTi\n')

        for i in range(len(T)):
            file_abundances_1bar.write(str(T[i])+'\t'+str(n_CH4_1bar[i])+'\t'+str(n_C2H2_1bar[i])+'\t'+str(n_H2O_1bar[i])+'\t'+str(n_CO_1bar[i])+'\t'+str(n_CO2_1bar[i])+'\t'+str(n_TiO_1bar[i])+'\t'+str(n_TiO2_1bar[i])+'\t'+str(nn_Ti_1bar[i])+'\n')
        for i in range(len(T)):
            file_abundances_P.write(str(T[i]) + '\t' + str(n_CH4_P_wanted[i]) + '\t' + str(n_C2H2_P_wanted[i]) + '\t' + str(n_H2O_P_wanted[i]) + '\t' + str(n_CO_P_wanted[i]) + '\t' + str(n_CO2_P_wanted[i]) + '\t' + str(n_TiO_P_wanted[i])+'\t'+str(n_TiO2_P_wanted[i])+'\t'+str(nn_Ti_P_wanted[i])+'\n')

        file_abundances_1bar.close()
        file_abundances_P.close()

        return n_H2O_1bar, n_H2O_P_wanted, nn_Ti_1bar, nn_Ti_P_wanted, n_TiO_1bar, n_TiO_P_wanted, n_TiO2_1bar, n_TiO2_P_wanted, n_CO_1bar, n_CO_P_wanted, n_CH4_1bar, n_CH4_P_wanted, n_CO2_1bar, n_CO2_P_wanted, n_C2H2_1bar, n_C2H2_P_wanted

    if selection == '4':
        # WATER

        T = np.linspace(500, 3000, 26)

        for j in np.arange(2):
            A = []
            if j == 0:
                print("P = 1 bar")
                K1 = K1_1bar
                K2 = K2_sin_presion
                K3 = K3_1bar
                K4 = K4_sin_presion
                K5 = K5_sin_presion
            if j == 1:
                print('P=' + P_wanted_str)
                K1 = K1_P_wanted
                K2 = K2_sin_presion
                K3 = K3_P_wanted
                K4 = K4_sin_presion
                K5 = K5_sin_presion

            A.append(- 2.0 * K4 * K5 * n_O) # A_0
            A.append(K4 * K5 - 2.0 * K4 * K5 * n_O + 2.0 * K4 * (n_Ti - n_O)) # A_1
            A.append(K4 + 2.0 * K4 * (n_Ti - n_O) + 4.0 * n_Ti - 2.0 * n_O) # A_2
            A.append(1.0 + 4.0 * n_Ti - 2.0 * n_O) # A_3

            if j == 0:
                n_H2O_1bar = root_searcher_4(A, T, j)
            if j == 1:
                n_H2O_P_wanted = root_searcher_4(A, T, j)

        # WATER

        n_H2O_1bar = np.array(n_H2O_1bar)
        n_H2O_P_wanted = np.array(n_H2O_P_wanted)

        # TITANIUM

        nn_Ti_1bar = (2.0 * n_Ti * K4 * K5)/(2. * n_H2O_1bar**2. + 2.* K4 * n_H2O_1bar + 2. * K4 * K5 )
        nn_Ti_P_wanted = (2.0 * n_Ti * K4 * K5)/(2. * n_H2O_P_wanted**2. + 2.* K4 * n_H2O_P_wanted + 2. * K4 * K5 )

        # TITANIUM OXIDE

        n_TiO_1bar = 2. * (nn_Ti_1bar * n_H2O_1bar)/(K5)
        n_TiO_P_wanted = 2. * (nn_Ti_P_wanted * n_H2O_P_wanted)/(K5)

        # TITANIUM DIOXIDE

        n_TiO2_1bar = (n_TiO_1bar * n_H2O_1bar)/K4
        n_TiO2_P_wanted = (n_TiO_P_wanted * n_H2O_P_wanted)/K4

        # SAVE DATA

        file_abundances_1bar = open('abu_toymodel_1bar.dat','w')
        file_abundances_P = open('abu_toymodel_P_'+P_wanted_str+'.dat','w')

        file_abundances_1bar.write('# P = '+str(P_1)+'\t C/O = '+CO_wanted_str+'\n')
        file_abundances_1bar.write('# T(K)\tH20\tTi\tTiO\tTiO2\n')

        file_abundances_P.write('# P = '+P_wanted_str+'\t C/O = '+CO_wanted_str+'\n')
        file_abundances_P.write('# T(K)\tH20\tTi\tTiO\tTiO2\n')

        for i in range(len(T)):
            file_abundances_1bar.write(str(T[i])+'\t'+str(n_H2O_1bar[i])+'\t'+str(nn_Ti_1bar[i])+'\t'+str(n_TiO_1bar[i])+'\t'+str(n_TiO2_1bar[i])+'\n')
        for i in range(len(T)):
            file_abundances_P.write(str(T[i])+'\t'+str(n_H2O_P_wanted[i])+'\t' + str(nn_Ti_P_wanted[i]) + '\t' + str(n_TiO_P_wanted[i])+'\t'+str(n_TiO2_P_wanted[i])+'\n')

        file_abundances_1bar.close()
        file_abundances_P.close()


        return n_H2O_1bar, n_H2O_P_wanted, nn_Ti_1bar, nn_Ti_P_wanted, n_TiO_1bar, n_TiO_P_wanted, n_TiO2_1bar, n_TiO2_P_wanted


def graphics_CO(P, figN, CO_wanted):  # P = "1bar" o P_wanted_str o ""

    T = np.linspace(500, 3000, 26)

    plt.figure()
    if P == "1bar":
        plt.plot(T, n_CH4_1bar, label='$CH_{4} (1bar)$', color='black', ls=':', linewidth=3)
        plt.plot(T, n_C2H2_1bar, label='$C_{2}H_{2} (1bar)$', color='y', ls=':', linewidth=3)
        plt.plot(T, n_H2O_1bar, label='$H_{2}O (1bar)$', color='m', ls=':', linewidth=3)
        plt.plot(T, n_CO_1bar, label='$CO (1bar)$', color='c', ls=':', linewidth=3)
        plt.plot(T, n_CO2_1bar, label='$CO_{2} (1bar)$', color='g', ls=':', linewidth=3)
        if selection == "3":
            plt.plot(T, nn_Ti_1bar, label='$Ti (1bar)$', color='purple', ls=':', linewidth=3)
            plt.plot(T, n_TiO_1bar, label='$TiO (1bar)$', color='b', ls=':', linewidth=3)
            plt.plot(T, n_TiO2_1bar, label='$TiO_{2} (1bar)$', color='r', ls=':', linewidth=3)
    elif P == P_wanted_str:
        plt.plot(T, n_CH4_P_wanted, label='$CH_{4} (P_wanted)$', color='black', ls=':', linewidth=1)
        plt.plot(T, n_C2H2_P_wanted, label='$C_{2}H_{2} (P_wanted)$', color='y', ls=':', linewidth=1)
        plt.plot(T, n_H2O_P_wanted, label='$H_{2}O (P_wanted)$', color='m', ls=':', linewidth=1)
        plt.plot(T, n_CO_P_wanted, label='$CO (P_wanted)$', color='c', ls=':', linewidth=1)
        plt.plot(T, n_CO2_P_wanted, label='$CO_{2} (P_wanted)$', color='g', ls=':', linewidth=1)
        if selection == "3":
            plt.plot(T, nn_Ti_P_wanted, label='$Ti (P_wanted)$', color='purple', ls=':', linewidth=1)
            plt.plot(T, n_TiO_P_wanted, label='$TiO (P_wanted)$', color='b', ls=':', linewidth=1)
            plt.plot(T, n_TiO2_P_wanted, label='$TiO_{2} (P_wanted)$', color='r', ls=':', linewidth=1)

    elif P == "":
        plt.plot(T, n_CH4_1bar, label='$CH_{4} (1bar)$', color='black', ls=':', linewidth=3)
        plt.plot(T, n_C2H2_1bar, label='$C_{2}H_{2} (1bar)$', color='y', ls=':', linewidth=3)
        plt.plot(T, n_H2O_1bar, label='$H_{2}O (1bar)$', color='m', ls=':', linewidth=3)
        plt.plot(T, n_CO_1bar, label='$CO (1bar)$', color='c', ls=':', linewidth=3)
        plt.plot(T, n_CO2_1bar, label='$CO_{2} (1bar)$', color='g', ls=':', linewidth=3)
        plt.plot(T, n_CH4_P_wanted, label='$CH_{4} (' + P_wanted_str + 'bar)$', color='black', ls='-', linewidth=1)
        plt.plot(T, n_C2H2_P_wanted, label='$C_{2}H_{2} (' + P_wanted_str + 'bar)$', color='y', ls='-', linewidth=1)
        plt.plot(T, n_H2O_P_wanted, label='$H_{2}O (' + P_wanted_str + 'bar)$', color='m', ls='-', linewidth=1)
        plt.plot(T, n_CO_P_wanted, label='$CO (' + P_wanted_str + 'bar)$', color='c', ls='-', linewidth=1)
        plt.plot(T, n_CO2_P_wanted, label='$CO_{2} (' + P_wanted_str + 'bar)$', color='g', ls='-', linewidth=1)
        if selection == "3":
            plt.plot(T, nn_Ti_1bar, label='$Ti (1bar)$', color='purple', ls=':', linewidth=3)
            plt.plot(T, n_TiO_1bar, label='$TiO (1bar)$', color='b', ls=':', linewidth=3)
            plt.plot(T, n_TiO2_1bar, label='$TiO_{2} (1bar)$', color='r', ls=':', linewidth=3)
            plt.plot(T, nn_Ti_P_wanted, label='$Ti (' + P_wanted_str + 'bar)$', color='purple', ls='-', linewidth=1)
            plt.plot(T, n_TiO_P_wanted, label='$TiO (' + P_wanted_str + 'bar)$', color='b', ls='-', linewidth=1)
            plt.plot(T, n_TiO2_P_wanted, label='$TiO_{2} (' + P_wanted_str + 'bar)$', color='r', ls='-', linewidth=1)

    plt.xlim([500, 3000])
    plt.ylim([10 ** -22, 10 ** 0])
    plt.yscale('log')
    plt.xticks(np.linspace(500, 3000, 6, endpoint=True))
    plt.yticks(10 ** (np.linspace(-22, 0, 12)))
    plt.tick_params(direction='in', bottom='on', top='on', left='on', right='on')
    plt.xlabel('Temperature (K)')
    plt.ylabel('$ñ_{x}$')
    plt.title('C/O = ' + str(CO_wanted))
    plt.legend(loc=4, prop={'size': 7}, ncol=2)
    plt.savefig('ti-abundances-nist-' + str(figN) + '.pdf')

def graphics_ToyModel(P, figN, CO_wanted):  # P = "1bar" o P_wanted_str o ""

    T = np.linspace(500, 3000, 26)

    plt.figure()
    if P == "1bar":
        plt.plot(T, n_H2O_1bar, label='$H_{2}O (1bar)$', color='g', ls=':', linewidth=3)
        plt.plot(T, nn_Ti_1bar, label='$Ti (1bar)$', color='purple', ls=':', linewidth=3)
        plt.plot(T, n_TiO_1bar, label='$TiO (1bar)$', color='b', ls=':', linewidth=3)
        plt.plot(T, n_TiO2_1bar, label='$TiO_{2} (1bar)$', color='r', ls=':', linewidth=3)
    if P == P_wanted_str:
        plt.plot(T, n_H2O_P_wanted, label='$H_{2}O (P_wanted)$', color='g', ls=':', linewidth=1)
        plt.plot(T, nn_Ti_P_wanted, label='$Ti (P_wanted)$', color='purple', ls=':', linewidth=1)
        plt.plot(T, n_TiO_P_wanted, label='$TiO (P_wanted)$', color='b', ls=':', linewidth=1)
        plt.plot(T, n_TiO2_P_wanted, label='$TiO_{2} (P_wanted)$', color='r', ls=':', linewidth=1)
    if P == "":
        plt.plot(T, n_H2O_1bar, label='$H_{2}O (1bar)$', color='g', ls=':', linewidth=3)
        plt.plot(T, nn_Ti_1bar, label='$Ti (1bar)$', color='purple', ls=':', linewidth=3)
        plt.plot(T, n_TiO_1bar, label='$TiO (1bar)$', color='b', ls=':', linewidth=3)
        plt.plot(T, n_TiO2_1bar, label='$TiO_{2} (1bar)$', color='r', ls=':', linewidth=3)
        plt.plot(T, n_H2O_P_wanted, label='$H_{2}O (' + P_wanted_str + 'bar)$', color='g', ls='-', linewidth=1)
        plt.plot(T, nn_Ti_P_wanted, label='$Ti (' + P_wanted_str + 'bar)$', color='purple', ls='-', linewidth=1)
        plt.plot(T, n_TiO_P_wanted, label='$TiO (' + P_wanted_str + 'bar)$', color='b', ls='-', linewidth=1)
        plt.plot(T, n_TiO2_P_wanted, label='$TiO_{2} (' + P_wanted_str + 'bar)$', color='r', ls='-', linewidth=1)
    plt.xlim([500, 3000])
    plt.ylim([10 ** -35, 10 ** 0])
    plt.yscale('log')
    plt.xticks(np.linspace(500, 3000, 6, endpoint=True))
    plt.yticks(10 ** (np.linspace(-35, 0, 6)))
    plt.tick_params(direction='in', bottom='on', top='on', left='on', right='on')
    plt.xlabel('Temperature (K)')
    plt.ylabel('$ñ_{x}$')
    plt.title('C/O = ' + str(CO_wanted))
    plt.legend(loc=4, prop={'size': 7}, ncol=2)
    plt.savefig('toymodel-abundances-' + str(figN) + '.pdf')

# Now inputs

# Enter your Selection

selection = input("What do you want to do? \n [1] Generate a graphic of abundances on a range of temperatures for a specific C/O and pressure \n [2] Obtain the specific abundances of molecules on a specific temperature, C/O and pressure \n [3] Titanium \n [4] Toy Model (H, Ti, O) \n Answer ('1', '2', '3' or '4'): ")

# Enter your Pressure

P_wanted_str = input("Enter the pressure wanted (bar) : ")
P_wanted = float(P_wanted_str)

# Enter your C/O

CO_wanted_str = input("Enter the C/O wanted : ")
CO_wanted = float(CO_wanted_str)

if selection == '1':

    K1_1bar, K1_P_wanted, K2_sin_presion, K3_1bar, K3_P_wanted, K4_sin_presion, K5_sin_presion = classic_calculation()

    # NOW WE GENERATES MORE GRAPHICS

    n_CH4_1bar, n_CH4_P_wanted, n_C2H2_1bar, n_C2H2_P_wanted, n_H2O_1bar, n_H2O_P_wanted, n_CO_1bar, n_CO_P_wanted, n_CO2_1bar, n_CO2_P_wanted = abundances_norm_H(CO_wanted, P_wanted_str, selection)

    graphics_CO("", "fig_CO_" + CO_wanted_str + "_P_" + P_wanted_str, '{0:.2f}'.format(CO_wanted))

    print("The files of abundances and graphics for C/O=" + CO_wanted_str + " and P=" + P_wanted_str + " are ready! :D \n")


elif selection == '2':

    T_wanted_str = input("Enter the temperature wanted (between 500K and 6000K): ")
    T_wanted = float(T_wanted_str)

    K1_P_wanted, K2_sin_presion, K3_P_wanted = insert_one_temp(T_wanted, P_wanted)

    n_CH4_P_wanted, n_C2H2_P_wanted, n_H2O_P_wanted, n_CO_P_wanted, n_CO2_P_wanted = abundances_norm_H(CO_wanted, P_wanted_str, selection)

    print("The files of abundances for C/O=" + CO_wanted_str + " P=" + P_wanted_str + " and T=" + T_wanted_str + " are ready! :D \n")

elif selection == '3':

    K1_1bar, K1_P_wanted, K2_sin_presion, K3_1bar, K3_P_wanted, K4_sin_presion, K5_sin_presion = classic_calculation()

    # NOW WE GENERATES MORE GRAPHICS

    n_H2O_1bar, n_H2O_P_wanted, nn_Ti_1bar, nn_Ti_P_wanted, n_TiO_1bar, n_TiO_P_wanted, n_TiO2_1bar, n_TiO2_P_wanted, n_CO_1bar, n_CO_P_wanted, n_CH4_1bar, n_CH4_P_wanted, n_CO2_1bar, n_CO2_P_wanted, n_C2H2_1bar, n_C2H2_P_wanted = abundances_norm_H(CO_wanted, P_wanted_str, selection)

    graphics_CO("", "fig_CO_" + CO_wanted_str + "_P_" + P_wanted_str, '{0:.2f}'.format(CO_wanted))

    print("The files of abundances and graphics for C/O=" + CO_wanted_str + " and P=" + P_wanted_str + " are ready! :D \n")

elif selection == '4':

    K1_1bar, K1_P_wanted, K2_sin_presion, K3_1bar, K3_P_wanted, K4_sin_presion, K5_sin_presion = classic_calculation()

    # NOW WE GENERATES MORE GRAPHICS

    n_H2O_1bar, n_H2O_P_wanted, nn_Ti_1bar, nn_Ti_P_wanted, n_TiO_1bar, n_TiO_P_wanted, n_TiO2_1bar, n_TiO2_P_wanted = abundances_norm_H(CO_wanted, P_wanted_str, selection)

    graphics_ToyModel("", "fig_CO_" + CO_wanted_str + "_P_" + P_wanted_str, '{0:.2f}'.format(CO_wanted))

    print("The files of abundances and graphics for C/O=" + CO_wanted_str + " and P=" + P_wanted_str + " are ready! :D \n")
