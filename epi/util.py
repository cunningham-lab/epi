""" General util functions for EPI. """

import numpy as np
import pickle
import os
from epi.error_formatters import format_type_err_msg


def gaussian_backward_mapping(mu, Sigma):
    if type(mu) is not np.ndarray:
        raise TypeError(
            format_type_err_msg(
                "epi.util.gaussian_backward_mapping", "mu", mu, np.ndarray
            )
        )
    elif type(Sigma) is not np.ndarray:
        raise TypeError(
            format_type_err_msg(
                "epi.util.gaussian_backward_mapping", "Sigma", Sigma, np.ndarray
            )
        )

    mu = np_column_vec(mu)
    Sigma_shape = Sigma.shape
    if len(Sigma_shape) != 2:
        raise ValueError("Sigma must be 2D matrix, shape ", Sigma_shape, ".")
    if Sigma_shape[0] != Sigma_shape[1]:
        raise ValueError("Sigma must be square matrix, shape ", Sigma_shape, ".")
    if not np.allclose(Sigma, Sigma.T, atol=1e-10):
        raise ValueError("Sigma must be symmetric. shape.")
    if Sigma_shape[1] != mu.shape[0]:
        raise ValueError("mu and Sigma must have same dimensionality.")

    D = mu.shape[0]
    Sigma_inv = np.linalg.inv(Sigma)
    x = np.dot(Sigma_inv, mu)
    y = np.reshape(-0.5 * Sigma_inv, (D ** 2))
    eta = np.concatenate((x[:, 0], y), axis=0)
    return eta


def np_column_vec(x):
    if type(x) is not np.ndarray:
        raise (
            TypeError(format_type_err_msg("epi.util.np_column_vec", "x", x, np.ndarray))
        )
    x_shape = x.shape
    if len(x_shape) == 1:
        x = np.expand_dims(x, 1)
    elif len(x_shape) == 2:
        if x_shape[1] != 1:
            if x_shape[0] > 1:
                raise ValueError("x is matrix.")
            else:
                x = x.T
    elif len(x_shape) > 2:
        raise ValueError("x dimensions > 2.")
    return x


def array_str(a):
    """Returns a compressed string from a 1-D numpy array.

    :param a: A 1-D numpy array.
    :type a: class`numpy.ndarray`
    :return: A string compressed via scientific notation and repeated elements.
    :rtype: str
    """
    if type(a) is not np.ndarray:
        raise TypeError(format_type_err_msg("epi.util.array_str", "a", a, np.ndarray))

    if len(a.shape) > 1:
        raise ValueError("epi.util.array_str takes 1-D arrays not %d." % len(a.shape))

    def repeats_str(num, mult):
        if mult == 1:
            return "%.2E" % num
        else:
            return "%dx%.2E" % (mult, num)

    d = a.shape[0]
    mults = []
    nums = []
    prev_num = a[0]
    mult = 1
    for i in range(1, d):
        if a[i] == prev_num:
            mult += 1
        else:
            nums.append(prev_num)
            prev_num = a[i]
            mults.append(mult)
            mult = 1

        if i == d - 1:
            nums.append(prev_num)
            mults.append(mult)

    array_str = repeats_str(nums[0], mults[0])
    for i in range(1, len(nums)):
        array_str += "_" + repeats_str(nums[i], mults[i])

    return array_str


def init_path(arch_string, init_type, init_param):
    if type(arch_string) is not str:
        raise TypeError(
            format_type_err_msg("epi.util.init_path", "arch_string", arch_string, str)
        )
    if type(init_type) is not str:
        raise TypeError(
            format_type_err_msg("epi.util.init_path", "init_type", init_type, str)
        )

    path = "./data/" + arch_string + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    if init_type == "iso_gauss":
        if "loc" in init_param:
            loc = init_param["loc"]
        else:
            raise ValueError("'loc' field not in init_param for %s." % init_type)
        if "scale" in init_param:
            scale = init_param["scale"]
        else:
            raise ValueError("'scale' field not in init_param for %s." % init_type)
        path += init_type + "_loc=%.2E_scale=%.2E" % (loc, scale)

    return path


def save_tf_model(path, variables):
    d = {}
    for variable in variables:
        d[variable.name] = variable.numpy()
    return pickle.dump(d, open(path + ".p", "wb"))


def load_tf_model(path, variables):
    d = pickle.load(open(path + ".p", "rb"))
    for variable in variables:
        variable.assign(d[variable.name])
    return None

class SimpleBleDevice(Peripheral):
    """This is a conceptual class representation of a simple BLE device (GATT Server). It is essentially an extended combination of the :class:`bluepy.btle.Peripheral` and :class:`bluepy.btle.ScanEntry` classes

    :param client: A handle to the :class:`simpleble.SimpleBleClient` client object that detected the device
    :type client: class:`simpleble.SimpleBleClient`
    :param addr: Device MAC address, defaults to None
    :type addr: str, optional
    :param addrType: Device address type - one of ADDR_TYPE_PUBLIC or ADDR_TYPE_RANDOM, defaults to ADDR_TYPE_PUBLIC
    :type addrType: str, optional
    :param iface: Bluetooth interface number (0 = /dev/hci0) used for the connection, defaults to 0
    :type iface: int, optional
    :param data: A list of tuples (adtype, description, value) containing the AD type code, human-readable description and value for all available advertising data items, defaults to None
    :type data: list, optional
    :param rssi: Received Signal Strength Indication for the last received broadcast from the device. This is an integer value measured in dB, where 0 dB is the maximum (theoretical) signal strength, and more negative numbers indicate a weaker signal, defaults to 0
    :type rssi: int, optional
    :param connectable: `True` if the device supports connections, and `False` otherwise (typically used for advertising ‘beacons’)., defaults to `False`
    :type connectable: bool, optional
    :param updateCount: Integer count of the number of advertising packets received from the device so far, defaults to 0
    :type updateCount: int, optional
    """

    def __init__(self, client, addr=None, addrType=ADDR_TYPE_PUBLIC, iface=0, data=None, rssi=0, connectable=False, updateCount=0):
        """Constructor method
        """
        super().__init__(deviceAddr=None, addrType=addrType, iface=iface)
        self.addr = addr
        self.addrType = addrType
        self.iface = iface
        self.rssi = rssi
        self.connectable = connectable
        self.updateCount = updateCount
        self.data = data
        self._connected = False
        self._services = []
        self._characteristics = []
        self._client = client

    def getServices(self, uuids=None):
        """Returns a list of :class:`bluepy.blte.Service` objects representing the services offered by the device. This will perform Bluetooth service discovery if this has not already been done; otherwise it will return a cached list of services immediately..

        :param uuids: A list of string service UUIDs to be discovered, defaults to None
        :type uuids: list, optional
        :return: A list of the discovered :class:`bluepy.blte.Service` objects, which match the provided ``uuids``
        :rtype: list On Python 3.x, this returns a dictionary view object, not a list
        """
        self._services = []
        if(uuids is not None):
            for uuid in uuids:
                try:
                    service = self.getServiceByUUID(uuid)
                    self.services.append(service)
                except BTLEException:
                    pass
        else:
            self._services = super().getServices()
        return self._services

    def setNotificationCallback(self, callback):
        """Set the callback function to be executed when the device sends a notification to the client.

        :param callback: A function handle of the form ``callback(client, characteristic, data)``, where ``client`` is a handle to the :class:`simpleble.SimpleBleClient` that invoked the callback, ``characteristic`` is the notified :class:`bluepy.blte.Characteristic` object and data is a `bytearray` containing the updated value. Defaults to None
        :type callback: function, optional
        """
        self.withDelegate(
            SimpleBleNotificationDelegate(
                callback,
                client=self._client
            )
        )

    def getCharacteristics(self, startHnd=1, endHnd=0xFFFF, uuids=None):
        """Returns a list containing :class:`bluepy.btle.Characteristic` objects for the peripheral. If no arguments are given, will return all characteristics. If startHnd and/or endHnd are given, the list is restricted to characteristics whose handles are within the given range.

        :param startHnd: Start index, defaults to 1
        :type startHnd: int, optional
        :param endHnd: End index, defaults to 0xFFFF
        :type endHnd: int, optional
        :param uuids: a list of UUID strings, defaults to None
        :type uuids: list, optional
        :return: List of returned :class:`bluepy.btle.Characteristic` objects
        :rtype: list
        """
        self._characteristics = []
        if(uuids is not None):
            for uuid in uuids:
                try:
                    characteristic = super().getCharacteristics(
                        startHnd, endHnd, uuid)[0]
                    self._characteristics.append(characteristic)
                except BTLEException:
                    pass
        else:
            self._characteristics = super().getCharacteristics(startHnd, endHnd)
        return self._characteristics

    def connect(self):
        """Attempts to initiate a connection with the device.

        :return: `True` if connection was successful, `False` otherwise
        :rtype: bool
        """
        try:
            super().connect(self.addr, addrType=self.addrType, iface=self.iface)
        except BTLEException as ex:
            self._connected = False
            return (False, ex)
        self._connected = True
        return True

    def disconnect(self):
        """Drops existing connection to device
        """
        super().disconnect()
        self._connected = False

    def isConnected(self):
        """Checks to see if device is connected

        :return: `True` if connected, `False` otherwise
        :rtype: bool
        """
        return self._connected
