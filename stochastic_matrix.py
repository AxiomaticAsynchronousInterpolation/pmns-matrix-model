"""
Python version: 3.13

To calculate the stochastic matrix the mass eigenstate unitary matrix is
calculated and using that matrix the probability of measuring a neutrino
in state 2 starting from state 1 is calculated for all 3 mass eigenstates
using the ultrarelativistic limit.

Data:
The data is from NuFIT "v6.1: Three-neutrino fit based on data available
in November 2025" (http://www.nu-fit.org/?q=node/309).

General references:
    [1] NuFIT for information on parameter fittings:
        https://www.nu-fit.org/
    [2] Wikipedia for information on the PMNS matrix:
        https://en.wikipedia.org/wiki/Pontecorvo%E2%80%93Maki%E2%80%93Nakagawa%E2%80%93Sakata_matrix
    [3] Wikipedia for information on neutrino oscilations and the used formulas:
        https://en.wikipedia.org/wiki/Neutrino_oscillation
"""
import numpy as np

#### Data sets from [1] ####
normal_ordering_data = [
    np.radians(33.76), np.radians(43.27), np.radians(8.62), # theta angles 12, 23, 13
    np.radians(207),                                        # CP-symmetry violation angle
    7.537e-5, 2.521e-3, 2.521e-3                            # mass square differences 21, 31, 32
]
inverted_ordering_data = [
    np.radians(33.76), np.radians(48.15), np.radians(8.65), # theta angles 12, 23, 13
    np.radians(283),                                        # CP-symmetry violation angle
    7.537e-5, -2.500e-3, -2.500e-3                          # mass square differences 21, 31, 32
]
normal_ordering_data_with_atmo = [
    np.radians(33.76), np.radians(43.29), np.radians(8.62), # theta angles 12, 23, 13
    np.radians(212),                                        # CP-symmetry violation angle
    7.537e-5, 2.511e-3, 2.511e-3                            # mass square differences 21, 31, 32
]
inverted_ordering_data_with_atmo = [
    np.radians(33.76), np.radians(47.90), np.radians(8.65), # theta angles 12, 23, 13
    np.radians(274),                                        # CP-symmetry violation angle
    7.537e-5, -2.483e-3, -2.483e-3                          # mass square differences 21, 31, 32
]

def oscillation_matrix(
        ratio,
        atmospheric_data: bool = False,
        ordering: str = "no"
    ) -> np.ndarray:
    """
    Calculates the 3x3 neutrino oscillation stochastic matrix for a given a
    qoutient of distance and energy (km/Gev) using the 3 neutrino PMNS matrix
    model using the NuFIT v6.1 Best-Fit Parameters.
    
    Parameters
    ----------
    ratio : float or np.ndarray
        The ratio of distance traveld (km) by the neutrino and the energy
        (GeV) of the neutrino.
    atmospheric_data : bool
        If `True` the best fit with SK atmospheric data is used and
        if `False` the best fit without SK atmospheric data is used.
        If nothing is specefied `False` is used.
    ordering : str
        The ordering used for the data. If `"no"` is enterd normal ordering is
        used. If `"io"` is enterd inverted ordering is used.
        If nothing is specefied `"no"` is used.

    Returns
    -------
    np.ndarray
        A 3x3 matrix stochastic matrix representing the probability for a
        flavor to be measured.
        The row and columns are orderd by electron, mu and tau (neutrino) such
        that component (m, n) is the the probability for flavor n to be
        measured if the neutrino started with flavor m.
    """
    #### Select correct data set ####
    if atmospheric_data is False and ordering == "no":
        data = normal_ordering_data
    if atmospheric_data is False and ordering == "io":
        data = inverted_ordering_data
    if atmospheric_data is True and ordering == "no":
        data = normal_ordering_data_with_atmo
    if atmospheric_data is True and ordering == "io":
        data = inverted_ordering_data_with_atmo

    # Asing values from the data set to the parameters
    theta12, theta23, theta13 = data[0], data[1], data[2] # theta angles 12, 23, 13 
    delta_cp = data[3]                                    # CP-symmetry violation angle
    dm2_21, dm2_31, dm2_32 = data[4], data[5], data[6]    # mass square differences 21, 31, 32
    
    # Compute paramters for the mass eigenstate unitary matrix
    c12, s12 = np.cos(theta12), np.sin(theta12)
    c23, s23 = np.cos(theta23), np.sin(theta23)
    c13, s13 = np.cos(theta13), np.sin(theta13)
    exp_i_delta = np.exp(1j * delta_cp)

    #### Constructing PMNS matrix ####
    # the components definition can be found under [2]
    U = np.zeros((3, 3), dtype = complex)
    
    U[0, 0] = c12 * c13
    U[0, 1] = s12 * c13
    U[0, 2] = s13 * np.conj(exp_i_delta)
    
    U[1, 0] = -s12 * c23 - c12 * s23 * s13 * exp_i_delta
    U[1, 1] =  c12 * c23 - s12 * s23 * s13 * exp_i_delta
    U[1, 2] =  s23 * c13
    
    U[2, 0] =  s12 * s23 - c12 * c23 * s13 * exp_i_delta
    U[2, 1] = -c12 * s23 - s12 * c23 * s13 * exp_i_delta
    U[2, 2] =  c23 * c13

    #### Precomputation ####
    # Conversion according to [3]
    conv = 1.2669328 # conversion factor for units
    
    D21 = conv * dm2_21 * ratio # converted argument
    D31 = conv * dm2_31 * ratio # converted argument
    D32 = conv * dm2_32 * ratio # converted argument
    
    # Computing phase grid
    # mapping mass square differences to their phase grids
    sin2_phases = np.array([np.sin(D21)**2, np.sin(D31)**2, np.sin(D32)**2])
    sin_2phases = np.array([np.sin(2*D21), np.sin(2*D31), np.sin(2*D32)])
    
    # mapping index pairs to the corresponding phase index (j, i) -> index
    # (we only care about: (0,1) -> D21, (0,2) -> D31, (1,2) -> D32)
    phase_map = {(0, 1): 0, (0, 2): 1, (1, 2): 2}

    #### Computing the stochastic matrix according to [3] ####
    # If ratio is a scalar, shape is (3, 3). If ratio is an array, shape is (3,3, ration(ratio))
    ratio_shape = np.atratioast_1d(ratio).shape[0] if isinstance(ratio, (list, np.ndarray)) else 1
    
    if ratio_shape > 1:
        P = np.zeros((3, 3, ratio_shape))
    else:
        P = np.zeros((3, 3))
        
    for a in range(3): # flavor 1
        for b in range(3): # flavor 2
            # Kronecker delta term
            if a == b:
                P[a, b] = 1.0
                
            # Summation over mass states (i > j)
            for i in [1, 2]:
                for j in range(i):
                    term = np.conj(U[a, i]) * U[b, i] * U[a, j] * np.conj(U[b, j])
                    p_idx = phase_map[(j, i)]
                    
                    P[a, b] -= 4 * np.real(term) * sin2_phases[p_idx]
                    P[a, b] += 2 * np.imag(term) * sin_2phases[p_idx]
                    
    return P
