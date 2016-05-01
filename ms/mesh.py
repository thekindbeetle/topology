import numpy as np
import abc


class GridMesh:
    """
    Квадратная сетка (возможно, с граничными условиями)
    """

    # Размеры сетки по X и Y
    sizeX, sizeY = 0, 0

    # Количество вершин
    size = 0

    # Значения сетки
    values = None

    def __init__(self, lx, ly):
        self.sizeX = lx
        self.sizeY = ly
        self.size = lx * ly
        self.values = np.zeros((self.sizeX, self.sizeY))

    def set_values(self, val):
        """
        :param val: NumPy array
        """
        self.values = val

    @abc.abstractmethod
    def value(self, idx):
        return
