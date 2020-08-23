from collections import OrderedDict
import numpy as np

from .power_injectors import Load, Generator
from .constants import DASH_1, DASH_2


class Buses(object):
    """
    A container class for the buses.

    At all times, the dictionary `buses` is sorted by bus ID.
    """
    def __init__(self):
        super().__init__()

        self.buses = OrderedDict()
        self.user2internal = {}
        self.internal2user = {}

    def __len__(self):
        return len(self.buses)

    def __getitem__(self, item):
        return self.buses[item]

    def __setitem__(self, key, value):
        self.buses[key] = value

    def values(self):
        return self.buses.values()

    def keys(self):
        return self.buses.keys()

    def items(self):
        return self.buses.items()

    def __str__(self):
        s = DASH_1 + '\n' + 'BUSES' + '\n'
        s += '{:<3s}{:^7s}{:^7s}{:^12s}{:^12s}{:^12s}'\
                 .format('ID', 'n_gen', 'n_load', 'p', 'q', 'v') + '\n' + DASH_2
        for bus in self.values():
            s += '\n' + bus.__str__()
        s += '\n' + DASH_1
        return s

    def add(self, buses):
        """
        Add one or more buses to the container.

        Parameters
        ----------
        buses : Bus or list of Bus
            The buses(es) to add to the container.
        """
        if isinstance(buses, Bus):
            buses = [buses]
        for bus in buses:
            if bus.bus_id not in self.keys():
                internal_id = len(self.buses)
                self[bus.bus_id] = bus
                self.user2internal[bus.bus_id] = internal_id
                self.internal2user[internal_id] = bus.bus_id
            else:
                msg = 'A buses with ID %d already exists.' % bus.bus_id
                raise ValueError(msg)

        # Re-order the buses by ID.
        self.buses = OrderedDict({i: self[i] for i in sorted(self.keys())})

    def remove(self, ids):
        """
        Remove one or more bus(es) from the network.

        Parameters
        ----------
        ids : int or list of int
            The ID(s) of the buses to remove.
        """
        raise NotImplementedError()

    def set_p(self, ps):
        """
        Set the active power injection at one or more buses.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        ps : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The active power injection time series, indexed by buses ID.
        """
        self._set_var_at_all_buses(ps, 'p')

    def set_q(self, qs):
        """
        Set the reactive power injection at one or more buses.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        qs : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The reactive power injection time series, indexed by buses ID.
        """
        self._set_var_at_all_buses(qs, 'q')

    def set_v(self, vs):
        """
        Set the complex voltage at one or more buses.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        vs : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The complex voltage time series, indexed by buses ID.
        """
        self._set_var_at_all_buses(vs, 'v')

    def set_i(self, iss):
        """
        Set the complex current injection at one or more buses.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        iss : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The complex nodal current injection, indexed by bus ID.
        """
        self._set_var_at_all_buses(iss, 'i')

    def _set_var_at_all_buses(self, values, var):
        """
        Update one nodal variable in {P, Q, I, V} for all buses.

        Parameters
        ----------
        values : list of np.ndarray or dict of {int : np.ndarray} or np.ndarray
            The new nodal time series (P, Q, I, V), indexed by bus ID.
        var : {'p', 'q', 'i', 'v'}
            Which variable to update.
        """

        # Values are provided as a list (assumed ordered by bus IDs).
        if isinstance(values, (list, np.ndarray)):
            if len(values) != len(self):
                msg = 'Number of new values is %d but there are %d buses.' % \
                      (len(values), len(self))
                raise ValueError(msg)
            else:
                values = {i: v for i, v in zip(self.keys(), values)}

        # Update the values at all buses specified.
        if isinstance(values, dict):
            for bus_id, value in values.items():
                value = np.copy(value)
                if bus_id in self.keys():
                    if var == 'p':
                        self[bus_id].p = value
                    elif var == 'q':
                        self[bus_id].q = value
                    elif var == 'v':
                        self[bus_id].v = value
                    elif var == 'i':
                        self[bus_id].i = value
                    else:
                        raise NotImplementedError()
                else:
                    msg = 'Bus %d does not exist.' % bus_id
                    raise ValueError(msg)

    def get_vectors(self, vars='all'):
        """
        Return the P, Q, V, I nodal vectors.

        Parameters
        ----------
        vars : str or list of str, optional
            A list or single element of {'p', 'q', 'v', 'i'}, the vectors to
            construct. Default to all.

        Returns
        -------
        dict of {str : np.ndarray} or np.ndarray
            The nodal vectors with keys {'p', 'q', 'v', 'i'}, or a single array
            if only one type of variable was requested.
        """
        all_vars = ['p', 'q', 'v', 'i']

        # Deal with str arguments.
        if isinstance(vars, str):
            if vars == 'all':
                vars = all_vars
            elif vars in all_vars:
                vars = [vars]
            else:
                raise NotImplementedError()

        vectors = {v: [] for v in vars}
        for bus in self.values():
            if 'p' in vars:
                vectors['p'].append(bus.p)
            if 'q' in vars:
                vectors['q'].append(bus.q)
            if 'v' in vars:
                vectors['v'].append(bus.v)
            if 'i' in vars:
                vectors['i'].append(bus.i)

        # Return a single numpy array if only one vector was requested.
        if len(vectors) == 1:
            vectors = np.array(list(vectors.values())[0])
        else:
            vectors = {k: np.array(v) for k, v in vectors.items()}

        return vectors

    def get_dicts(self, vars):
        """
        Same as `get_vectors` but return dictionaries indexed by bus ID.

        Parameters
        ----------
        vars : str or list of str
            See `get_vectors`.

        Returns
        -------
        dict or dict of dict
            Same as `get_vectors`, but instead of lists the variables are
            returned as dictionaries indexed by bus IDs.
        """

        # Collect the values requested as vectors.
        vectors = self.get_vectors(vars)
        if not isinstance(vectors, dict):
            vectors = {0: vectors}

        # Within the dictionary, transform each list into a dictionary indexed
        # by bus ID.
        for var, vec in vectors.items():
            vectors[var] = {bus_id: v for bus_id, v in zip(self.keys(), vec)}

        # If only one variable was specified, return a single dictionary.
        if len(vars) == 1:
            vectors = vectors[list(vectors.keys())[0]]

        return vectors

    def compute_I_from_YV(self, Y):
        """
        Compute nodal current injections from nodal voltages.

        The nodal current injections are computed in matrix form as:
        .. math:
            I = YV
        where :math:`I, V \in \mathbb R^N` are the nodal complex current
        injection and voltage vectors, and :math:`y \in \mathbb R^{N\times N} is
        the nodal admittance matrix.

        Parameters
        ----------
        Y : np.ndarray or scipy.sparse.spmatrix
            The nodal admittance matrix of the network.
        """
        i = Y.dot(self.get_vectors('v'))
        self.set_i(i)

    def compute_S_from_VI(self):
        """
        Compute nodal power injections from voltages and current injections.

        The real and reactive nodal power injections are computed as:
        .. math:
            P_i + jQ_i = V_i I_i^*
        """
        vectors = self.get_vectors(['v', 'i'])
        s = vectors['v'] * np.conj(vectors['i'])
        self.set_p(s.real)
        self.set_q(s.imag)


class Bus(object):
    """ A class modelling a buses of the power system. """

    def __init__(self, bus_id):
        self.bus_id = bus_id
        self.n_load = 0
        self.n_gen = 0
        self.p = np.empty(0)
        self.q = np.empty(0)
        self.v = np.empty(0)
        self.i = np.empty(0)

    def update_device_count(self, n_dev):
        # Generators.
        try:
            self.n_gen = n_dev[Generator.__name__]
        except KeyError:
            self.n_gen = 0

        # Loads
        try:
            self.n_load = n_dev[Load.__name__]
        except KeyError:
            self.n_load = 0

    def __str__(self):
        s = '{:<3d}{:^7d}{:^7d}{:^12s}{:^12s}{:^12s}'\
            .format(self.bus_id, self.n_gen, self.n_load, str(self.p.shape),
                    str(self.q.shape), str(self.v.shape))
        return s