# jennabu
Repository for the python program "jennabu". 

The ways to be initialized are:

\begin{itemize}
	\item \textbf{jennabuNIST:} Using the JANAF-NIST thermochemical potentials coefficients.
	\item \textbf{jennabuCEA:} Using the CEA thermochemical potentials coefficients. 
\end{itemize}

Hence the difference between these two ways to initialize \texttt{jennabu} falls on the values of the standard enthalpy of formation, $H^{\degree}_{298}$ (at the reference temperature of $298.15\,K$ and pressure of $1\,bar$), in addition to the coefficients and integrations constants for enthalpy ($H^{\degree}$) and entropy ($S^{\degree}$).  \\

From this two ways we chose \texttt{jennabuCEA} due that this presents less percentage difference in comparison with the CEA results (percentage difference that we'll explain later). \\

The modes of the program are:

\begin{itemize}
	\item \textbf{[CHO] with range of temperature:} Generate a graphic of mixing ratios of molecules involve in a CHO model for a range of temperatures from $500K$ to $3000$ for a specific C/O and pressure.
	\item \textbf{[CHO] for a specific temperature:} Obtains the specific mixing ratios of molecules on a certain temperature, C/O and pressure.
	\item \textbf{[TiCHO] with range of temperature:} Generate a graphic of mixing ratios of molecules involve in a TiCHO model for a range of temperatures from $500K$ to $3000$ for a specific C/O and pressure.
	\item \textbf{[TiHO] Toy model:} Generate a graphic of mixing ratios of molecules involve in a TiHO model for a range of temperatures from $500K$ to $3000$ for a specific C/O and pressure.
\end{itemize}
