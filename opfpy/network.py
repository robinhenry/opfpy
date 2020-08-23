import numpy as np
from scipy.sparse import csc_matrix

from .components.branch import Branch, Branches
from .components.bus import Buses
from .components.power_injectors import PowerInjectors, PowerInjector
from .io import save_network


class Network(object):
    """ A class modelling a power system. """

    def __init__(self):
        self.buses = Buses()
        self.devices = PowerInjectors()
        self.branches = Branches()

        # Flag to indicate if the nodal admittance matrix has been built (and is
        # up to date).
        self.Y_built = False

    def __str__(self):
        s = self.buses.__str__() + '\n'
        s += self.devices.__str__() + '\n'
        s += self.branches.__str__()
        return s

    def build_Y_bus(self):
        """
        Build the nodal admittance matrix of the network.

        The nodal admittance matrix is constructed as follows:
        .. math::
            Y_{ij} =
            \begin{cases}
                - y_{ij}, & \text{if } i ~ j, \\
                \sum_{k:i~k} (y_{ik} + y_{ik}^{shunt}), & \text{if } i = j, \\
                0, & \text{otherwise.}
            \end{cases}
        where :math:`y_{ij}` is the line series admittance :math:`y_{ij} =
        y_{ji} = r_{ij} + j x_{ij}` and :math:`y_{ij}^{shunt}` is the shunt
        admittance of the line at bus :math:`i` in the pi-model.

        Returns
        -------
        scipy.sparse.csc_matrix
            The sparse (N, N) nodal admittance matrix of the network.
        """
        n_bus = self.n_buses
        Y_bus = np.zeros((n_bus, n_bus), dtype=np.complex)

        for (i, j), br in self.branches.items():
            i_internal = self.buses.user2internal[i]
            j_internal = self.buses.user2internal[j]

            # Fill an off-diagonal elements of the admittance matrix.
            Y_bus[i_internal, j_internal] = - br.y_ij
            Y_bus[j_internal, i_internal] = - br.y_ij

            # Increment diagonal element of the admittance matrix.
            Y_bus[i_internal, i_internal] += br.y_ij + br.y_shunt_from
            Y_bus[j_internal, j_internal] += br.y_ij + br.y_shunt_to

        self.Y = csc_matrix(Y_bus)
        self.Y_built = True

    def add_buses(self, buses):
        """
        Add one or more buses to the network.

        Parameters
        ----------
        buses : Bus or list of Bus
            The buses(es) to add to the network.
        """
        self.buses.add(buses)
        self.Y_built = False

    def add_devices(self, devices):
        """
        Add one or more devices (power injectors) to the network.

        Parameters
        ----------
        devices : PowerInjector or list of PowerInjector
            The device(s) to add to the network.
        """

        if isinstance(devices, PowerInjector):
            devices = [devices]

        # Check that each buses exists.
        for device in devices:
            if device.bus_id not in self.buses.keys():
                msg = 'Bus with ID %d does not exist.' % device.bus_id
                raise ValueError(msg)

        # Add the devices.
        self.devices.add(devices)

        # Update the number of devices connected to each buses.
        n_devices_per_bus = self.devices.n_devices_per_bus()
        for bus_id, n_dev in n_devices_per_bus.items():
            self.buses[bus_id].update_device_count(n_dev)

    def add_branches(self, branches):
        """
        Add one or more branches to the network.

        Parameters
        ----------
        branches : Branch or list of Branch
            The branch(es) to add to the network.
        """
        if isinstance(branches, Branch):
            branches = [branches]

        # Check that each branch is valid.
        for branch in branches:
            msg = 'Trying to add branch (%d, %d)' % (branch.bus_i, branch.bus_j)\
                  + ' but buses %d does not exist.'
            if branch.bus_i not in self.buses.keys():
                raise ValueError(msg % branch.bus_i)
            elif branch.bus_j not in self.buses.keys():
                raise ValueError(msg % branch.bus_j)

        # Add branches.
        self.branches.add(branches)
        self.Y_built = False

    def update_Ibus_from_YVbus(self):
        if not self.Y_built:
            self.build_Y_bus()
        self.buses.compute_I_from_YV(self.Y)

    def update_Sbus_from_VbusIbus(self):
        self.buses.compute_S_from_VI()

    def update_Ibranch_from_Vbus(self):
        v_bus = self.buses.get_dicts('v')
        self.branches.compute_I_from_Vbus(v_bus)

    def update_Sbranch_from_IbranchVbus(self, v_bus, direction):
        v_bus = self.buses.get_dicts('v')
        self.branches.compute_S_from_IVbus(v_bus, direction)

    def update_Ibus_from_Ibranch(self):
        i_bus = self.branches.compute_bus_SI_from_branch_SI('i')
        i_bus = self._unspecified_bus_to_zero(i_bus)
        self.buses.set_i(i_bus)

    def update_Sbus_from_Sbranch(self):
        pq_bus = self.branches.compute_bus_SI_from_branch_SI(['p', 'q'])
        p_bus = self._unspecified_bus_to_zero(pq_bus['p'])
        q_bus = self._unspecified_bus_to_zero(pq_bus['q'])
        self.buses.set_p(p_bus)
        self.buses.set_q(q_bus)

    def update_Sbus_from_Sdevices(self):
        bus_p, bus_q = self.devices.total_bus_power_injections()
        bus_p = self._unspecified_bus_to_zero(bus_p)
        bus_q = self._unspecified_bus_to_zero(bus_q)
        self.buses.set_p(bus_p)
        self.buses.set_q(bus_q)

    def _unspecified_bus_to_zero(self, dic):
        shape = dic[list(dic.keys())[0]].shape
        for bus_id in self.buses.keys():
            if bus_id not in dic:
                dic[bus_id] = np.zeros(shape)
        return dic

    def save(self, filepath):
        """ Save the network and its state to a file. """
        save_network(self, filepath)

    def update_all(self):
        """ Update the state of the network. """

        # # Build admittance matrix.
        # self.build_Y_bus()
        #
        # # Update nodal power and current injections.
        # bus_p, bus_q = self.devices.total_bus_power_injections()
        # for bus_id in self.buses.keys():
        #     if bus_id not in bus_p.keys():
        #         bus_p[bus_id] = 0
        #     if bus_id not in bus_q.keys():
        #         bus_q[bus_id] = 0
        # self.buses.set_p(bus_p)
        # self.buses.set_q(bus_q)
        #
        #
        # self.buses.compute_I_from_YV(self.Y)
        #
        # # Update branch current injections.
        # self.branches.compute_I_from_Vbus(self.buses)

        raise NotImplementedError()

    @property
    def n_buses(self):
        return len(self.buses)

    @property
    def n_devices(self):
        return len(self.devices)

    @property
    def n_branches(self):
        return len(self.branches)
