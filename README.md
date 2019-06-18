# jennabu :earth_americas: 

This code have two ways to be initialized are:

- **jennabuNIST:** Using the JANAF-NIST thermochemical potentials coefficients.
- **jennabuCEA:** Using the CEA thermochemical potentials coefficients. 

The difference between these two ways to initialize **jennabu** falls on the values of the standard enthalpy of formation (at the reference temperature of 298.15K and pressure of 1bar), in addition to the coefficients and integrations constants for enthalpy (Hº) and entropy (Sº).

The modes of the program are:

- **[CHO] with range of temperature:** Generate a graphic of mixing ratios of molecules involve in a CHO model for a range of temperatures from 500K to 3000K for a specific C/O and pressure.
- **[CHO] for a specific temperature:** Obtains the specific mixing ratios of molecules on a certain temperature, C/O and pressure.
- **[TiCHO] with range of temperature:** Generate a graphic of mixing ratios of molecules involve in a TiCHO model for a range of temperatures from 500K to 3000K for a specific C/O and pressure.
- **[TiHO] Toy model:** Generate a graphic of mixing ratios of molecules involve in a TiHO model for a range of temperatures from 500K to 3000K for a specific C/O and pressure.

## How to run jennabu?

**jennabu** is a code that runs on Python3. To initialize the program all you have to do is to write in the terminal:
```
python3 jennabu_CEAdata.py
```
to initialize jennabuCEA, or 
```
python3 jennabu_NISTdata.py
```
to initialize jennabuNIST.
