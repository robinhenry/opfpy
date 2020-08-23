from collections import OrderedDict
import numpy as np

from .constants import DASH_1, DASH_2


class Branches(object):
    """
    A container class for the branches of the network.

    The dictionary `branches` remains sorted by the order in which the branches
    were added.
    """

    def __init__(self):
        super().__init__()
        self.branches = OrderedDict()

    def __len__(self):
        return len(self.branches)

    def __getitem__(self, item):
        return self.branches[item]

    def __setitem__(self, key, value):
        self.branches[key] = value

    def values(self):
        return self.branches.values()

    def keys(self):
        return self.branches.keys()

    def items(self):
        return self.branches.items()

    def __str__(self):
        s = DASH_1 + '\n' + 'BRANCHES' + '\n'
        s += '{:<4s}{:<4s}'.format('i', 'j') + '\n' + DASH_2
        for branch in self.values():
            s += '\n' + branch.__str__()
        s += '\n' + DASH_1
        return s

    def add(self, branches):
        """
        Add one or more branches to connect buses in the network.

        Parameters
        ----------
        branches : Branch or list of Branch
            The branches to add.
        """

        if isinstance(branches, Branch):
            branches = [branches]
        for branch in branches:
            if (branch.bus_i, branch.bus_j) in self.keys():
                msg = 'A branch (%d, %d) already exists' % (branch.bus_i,
                                                            branch.bus_j)
                raise ValueError(msg)
            else:
                self[(branch.bus_i, branch.bus_j)] = branch

    def remove(self, ids):
        """
        Remove one or more branch(es) from the network.

        Parameters
        ----------
        ids : tuple or list of tuple
            The ID(s) of the branches to remove.
        """
        raise NotImplementedError()

    def set_p(self, ps, direction):
        """
        Set the active power flow at the end of one or more branches.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        ps : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The active power flow time series, indexed by branch IDs or order
            of creation (if list).
        direction : {'from', 'to'}
            At which end of the branch to update the variable.
        """
        self._set_var_in_all_branches(ps, 'p', direction)

    def set_q(self, qs, direction):
        """
        Set the reactive power flow at the end of one or more branches.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        qs : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The active power flow time series, indexed by branch IDs or order
            of creation (if list).
        direction : {'from', 'to'}
            At which end of the branch to update the variable.
        """
        self._set_var_in_all_branches(qs, 'q', direction)

    def set_i(self, iss, direction):
        """
        Set the current flow at the end of one or more branches.

        See `_set_var_at_all_buses`.

        Parameters
        ----------
        iss : dict of {int : np.ndarray} or list of np.ndarray or np.ndarray
            The active power flow time series, indexed by branch IDs or order
            of creation (if list).
        direction : {'from', 'to'}
            At which end of the branch to update the variable.
        """
        self._set_var_in_all_branches(iss, 'i', direction)

    def _set_var_in_all_branches(self, values, var, direction):
        """
        Update one branch flow variable in {P, Q, I} for all branches.

        Parameters
        ----------
        values : list of np.ndarray or dict of {int : np.ndarray} or np.ndarray
            The new branch flow time series (P, Q, I, V), indexed by bus ID.
        var : {'p', 'q', 'i', 'v'}
            Which variable to update.
        direction : {'from', 'to'}
            At which end of the branch to update the variable.
        """

        if direction not in ['from', 'to']:
            raise NotImplementedError()

        # Values are provided as a list (assumed ordered by branch IDs).
        if isinstance(values, (list, np.ndarray)):
            if len(values) != len(self):
                msg = 'Number of new values is %d but there are %d branches.' % \
                      (len(values), len(self))
                raise ValueError(msg)
            else:
                values = {i: v for i, v in zip(self.keys(), values)}

        # Update the values at all branches specified.
        if isinstance(values, dict):
            for branch_id, value in values.items():
                if branch_id in self.keys():
                    value = np.copy(value)
                    if var == 'p':
                        if direction == 'from':
                            self[branch_id].p_from = value
                        else:
                            self[branch_id].p_to = value
                    elif var == 'q':
                        if direction == 'from':
                            self[branch_id].q_from = value
                        else:
                            self[branch_id].q_to = value
                    elif var == 'i':
                        if direction == 'from':
                            self[branch_id].i_from = value
                        else:
                            self[branch_id].i_to = value
                    else:
                        raise NotImplementedError()
                else:
                    msg = 'Branch %d does not exist.' % branch_id
                    raise ValueError(msg)

    def get_vectors(self, vars='all'):
        """
        Return the branch I vectors, indexed by order of branch creation.

        Parameters
        ----------
        vars : str or list of str
            Which vectors to return, any combination of {'p_from', 'p_to',
            'q_from', 'q_to', 'i_from', 'i_to'}. Use 'all' for all of them.

        Returns
        -------
        dict of {(int, int) : np.ndarray} or np.ndarray
            The vectors, indexed by the same keys as `vars`. If a single vector
            is to be returned, then return the array itself.
        """

        all_vars = ['p_from', 'p_to', 'q_from', 'q_to', 'i_from', 'i_to']

        # Deal with str arguments.
        if isinstance(vars, str):
            if vars == 'all':
                vars = all_vars
            elif vars in all_vars:
                vars = [vars]
            else:
                raise NotImplementedError()

        vectors = {v: [] for v in vars}

        for branch in self.values():
            if 'p_from' in vars:
                vectors['p_from'] = branch.p_from
            if 'p_to' in vars:
                vectors['p_to'].append(branch.p_to)
            if 'q_from' in vars:
                vectors['q_from'].append(branch.q_from)
            if 'q_to' in vars:
                vectors['q_to'].append(branch.q_to)
            if 'i_from' in vars:
                vectors['i_from'].append(branch.i_from)
            if 'i_to' in vars:
                vectors['i_to'].append(branch.i_to)

        # Return a single numpy array if only one vector was requested.
        if len(vectors) == 1:
            vectors = np.array(list(vectors.values())[0])
        else:
            vectors = {k: np.array(v) for k, v in vectors.items()}

        return vectors

    def compute_I_from_Vbus(self, v_bus):
        """
        Compute branch current injections on each side of the branch.

        The current injections are computed as:
        .. math:
            I_{ij} = y_{ij} (V_i - V_j) + y_{ij}^{shunt} V_i
            I_{ji} = y_{ij} (V_j - V_i) + y_{ji}^{shunt} V_j

        Both :math:`I_{ij}` and :math:`I_{ji}` represent the injection into the
        branch, so they have opposite signs.

        Note that, unless shunt admittances are zero, their are not negative of
        each other. Their sum thus represents the total current loss along the
        line due to shunt admittances:
        .. math:
            I_{ij}^{loss} = I_{ij} + I_{ji}
                          = y_{ij}^{shunt} V_i + y_{ji}^{shunt} V_j \ne 0

        Parameters
        ----------
        v_bus : dict of {int : np.complex}
            The complex nodal voltages, indexed by bus ID.
        """
        for branch in self.values():
            v_i = v_bus[branch.bus_i]
            v_j = v_bus[branch.bus_j]
            branch.i_from = branch.y_ij * (v_i - v_j) + branch.y_shunt_from * v_i
            branch.i_to = branch.y_ij * (v_j - v_i) + branch.y_shunt_to * v_j

    def compute_S_from_IVbus(self, v_bus, direction):
        """
        Compute branch power flows on each side of the branch.

        The real and reactive power flows are computed as:
        .. math:
            P_{km} + jQ_{km} = V_k I_{km}^*

        Parameters
        ----------
        v_bus : dict of {int : np.complex}
            The complex nodal voltages, indexed by bus ID.
        direction : {'from', 'to'}
            At which end of the branch to compute the flows.
        """
        for branch in self.values():
            if direction == 'from':
                s = v_bus[branch.bus_i] * np.conj(branch.i_from)
                branch.p_from = s.real
                branch.q_from = s.imag
            elif direction == 'to':
                s = v_bus[branch.bus_j] * np.conj(branch.i_to)
                branch.p_to = s.real
                branch.q_to = s.imag
            else:
                raise NotImplementedError()

    def compute_bus_SI_from_branch_SI(self, vars):
        """
        Compute bus {P, Q, I} injections as the sum of branch injections.

        Parameters
        ----------
        vars : str or list of str
            Which nodal injections in {'p', 'q', 'i'} to compute. Use 'all' to
            return all.

        Returns
        -------
        dict of {dict of {int : np.ndarray}} or dict of {int : np.ndarray}
            The nodal injections specified in `vars`, each as a dictionary
            indexed by bus ID.
        """
        all_vars = ['p', 'q', 'i']

        # Deal with str argument.
        if isinstance(vars, str):
            if vars == 'all':
                vars = all_vars
            elif vars in all_vars:
                vars = [vars]

        def _get_from_and_to_values(br, v):
            """ Return the `from` and `to` value of `var` in {'p', 'q', 'i'}. """
            if v == 'p':
                return br.p_from, branch.p_to
            elif v == 'q':
                return br.q_from, branch.q_to
            elif v == 'i':
                return br.i_from, branch.i_to
            else:
                raise NotImplementedError()

        # Compute the total injections at each bus.
        injections = {v: {} for v in vars}
        for (bus_i, bus_j), branch in self.items():
            for var in vars:  # var in {'p', 'q', 'i'}
                # Get the flow at each end of the branch.
                val_from, val_to = _get_from_and_to_values(branch, var)

                # Flow leaving the bus is positive.
                if bus_i in injections[var].keys():
                    injections[var][bus_i] += val_from
                else:
                    injections[var][bus_i] = val_from.copy()

                # Flow arriving at the bus is negative.
                if bus_j in injections[var].keys():
                    injections[var][bus_j] += np.copy(val_to)
                else:
                    injections[var][bus_j] = np.copy(val_to)

        # Return a single dictionary if only one variable type was specified.
        if len(vars) == 1:
            injections = injections[vars[0]]

        return injections


class Branch(object):
    """ A branch connecting two buses of the network. """

    def __init__(self, bus_i, bus_j, r=0, x=0, b=0, length=1, amp=None):
        """
        Parameters
        ----------
        bus_i : int
            The sending buses ID.
        bus_j : int
            The receiving buses ID.
        r : float
            The resistance of the branch.
        x : float
            The reactance of the branch.
        b : float
            The total shunt susceptance of the branch, such that
            :math:`y_{ij}^{shunt} = y_{ji}^{shunt} = jb/2`.
        length : float
            The length of the branch.
        amp : float
            The ampacity of the branch.
        """
        self.bus_i = int(bus_i)
        self.bus_j = int(bus_j)
        self.r = r
        self.x = x
        self.b = b
        self.length = length
        self.amp = amp
        self.p_from = np.empty(0)
        self.p_to = np.empty(0)
        self.q_from = np.empty(0)
        self.q_to = np.empty(0)
        self.i_from = np.empty(0)
        self.i_to = np.empty(0)

        # Construct admittance.
        self.y_ij = 1. / (self.r + 1j * self.x)
        self.y_shunt_from = 1j * self.b / 2  # shunt admittance of the line at i
        self.y_shunt_to = 1j * self.b / 2    # shunt admittance of the line at j

    def __str__(self):
        return '{:<4d}{:<4d}'.format(self.bus_i, self.bus_j)
