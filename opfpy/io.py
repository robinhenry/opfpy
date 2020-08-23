import pickle


def save_network(network, filepath):
    """ Save a network and its state to a file. """
    with open(filepath, 'wb') as f:
        pickle.dump(network, f)


def load_network(filepath):
    """ Load a previously saved network from a file. """
    with open(filepath, 'rb') as f:
        network = pickle.load(f)
    return network
