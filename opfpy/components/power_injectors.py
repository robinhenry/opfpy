from collections import OrderedDict
import numpy as np

from .constants import DASH_1, DASH_2


class PowerInjectors(OrderedDict):
    """
    A container class for the power injectors connected to the network.
    """
    def __init__(self):
        super().__init__()

        # Keep track of number of generators and loads.
        self.n_per_type = {}

    def add(self, devices):
        """
        Add one or more devices (power injectors) to the container.

        Parameters
        ----------
        devices : PowerInjector or list of PowerInjector
            The device(s) to add.
        """
        if isinstance(devices, PowerInjector):
            devices = [devices]

        for dev in devices:
            if dev.dev_id in self.keys():
                msg = 'A device with ID %d already exists.' % dev.dev_id
                raise ValueError(msg)
            else:
                self[dev.dev_id] = dev

            # Update the counter of devices.
            name = type(dev).__name__
            if name in self.n_per_type.keys():
                self.n_per_type[name] += 1
            else:
                self.n_per_type[name] = 1

    def set_p(self, ps):
        """
        Parameters
        ----------
        ps : dict of {int : np.ndarray}
            The active power injection time series, indexed by device IDs.
        """
        for dev_id, p in ps.items():
            if dev_id in self.keys():
                self[dev_id].p = p
            else:
                msg = 'Device %d does not exist.' % dev_id
                raise ValueError(msg)

    def set_q(self, qs):
        """
        Parameters
        ----------
        qs : dict of {int : np.ndarray}
            The reactive power injection time series, indexed by device IDs.
        """
        for dev_id, q in qs.items():
            if dev_id in self.keys():
                self[dev_id].q = q
            else:
                msg = 'Device %d does not exist.' % dev_id
                raise ValueError(msg)

    def total_bus_power_injections(self):
        """
        Return the total P and Q power injections at each bus.

        Returns
        -------
        ps : dict of {int : np.ndarray}
            The total P injection at each bus, indexed by bus ID.
        qs : dict of {int : np.ndarray}
            The total Q injection at each bus, indexed by bus ID.
        """
        ps, qs = {}, {}
        for dev in self.values():
            if dev.bus_id in ps.keys():
                ps[dev.bus_id] += dev.p
                qs[dev.bus_id] += dev.q
            else:
                ps[dev.bus_id] = np.copy(dev.p)
                qs[dev.bus_id] = np.copy(dev.q)

        return ps, qs

    def n_devices_per_bus(self):
        n_dev = {}
        for dev in self.values():
            name = type(dev).__name__
            if dev.bus_id in n_dev.keys():
                if name in n_dev[dev.bus_id].keys():
                    n_dev[dev.bus_id][name] += 1
                else:
                    n_dev[dev.bus_id][name] = 1
            else:
                n_dev[dev.bus_id] = {name: 1}

        return n_dev

    @property
    def n_devices(self):
        return sum([len(bus) for bus in self])

    @property
    def n_gen(self):
        return self._get_number_device_type(Generator.__name__)

    @property
    def n_load(self):
        return self._get_number_device_type(Load.__name__)

    def _get_number_device_type(self, class_name):
        try:
            return self.n_per_type[class_name]
        except KeyError:
            return 0

    # def __str__(self):
    #     s = DASH_1 + '\n' + 'DEVICES'
    #     heading = '{:<4s}{:^8s}{:^8s}'.format('ID', 'p', 'q')
    #     for bus_id, bus in self.items():
    #         s += '\n' + DASH_2 + '\n' + 'Bus ' + str(bus_id) + '\n' + heading
    #         for dev_id, dev in bus.items():
    #             s += '\n' + dev.__str__()
    #     s += '\n' + DASH_1
    #     return s


class PowerInjector(object):
    """ Base class for any electrical device connected to the network. """

    def __init__(self, bus_id, dev_id, p=0, q=0):
        self.bus_id = bus_id
        self.dev_id = dev_id
        self.p = p
        self.q = q

    def __str__(self):
        return '{:<4d}{:^8.3f}{:^8.3f}'.format(self.bus_id, self.p, self.q)


class Load(PowerInjector):
    """ A load connected to the network. """

    def __init__(self, bus_id, dev_id, p=0, q=0):
        super().__init__(bus_id, dev_id, p, q)


class Generator(PowerInjector):
    """ A generator connected to the network. """

    def __init__(self, bus_id, dev_id, p=0, q=0):
        super().__init__(bus_id, dev_id, p, q)