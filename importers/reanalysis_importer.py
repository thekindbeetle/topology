import netCDF4
import numpy as np
import os.path
import cv2
from datetime import datetime


class ReanalysisImporter:
    """
    Парсер данных реанализа по температуре.
    """

    def __init__(self, data_folder='D:/data/air_temperature'):
        """
        :param data_file: Папка с наборами данных в формате NetCDF4
        :return:
        """
        self.dset = netCDF4.Dataset(os.path.join(data_folder, 'air.mon.mean.nc'), "r", format="NETCDF4")
        self.dtimes = netCDF4.num2date(self.dset.variables['time'][:], self.dset.variables['time'].units)
        self.levels = self.dset.variables['level'][:].data
        self.data_folder = data_folder

    def get_mean_month_temperature_data(self, year, month, level=1000):
        """
        Поле средней температуры за месяц.
        :param year: год
        :param month: месяц
        :param level: уровень (мБар) - по умолчанию, у поверхности (1000 мБар)
        :return:
        """
        time_idx = np.argwhere(self.dtimes==datetime(year, month, 1))[0][0]
        level_idx = np.argwhere(self.levels==level)[0][0]
        return np.roll(cv2.resize(self.dset.variables['air'][time_idx, level_idx, ::-1, :], dsize=(1440, 720)), 720, axis=1)

    def get_temperature_day_data(self, year, month, day, level=1000):
        """
        Данные по температуре за конкретный день.
        :param level: уровень (давление, мбар). По умолчанию - у поверхности (10000 мбар).
        :param month: месяц
        :param day: год
        :return:
        """
        dfile = netCDF4.Dataset(os.path.join(self.data_folder, 'air.{y}.nc'.format(y=year)), "r", format="NETCDF4")
        dfile_times = netCDF4.num2date(dfile.variables['time'][:], dfile.variables['time'].units)
        time_idx = np.argwhere(self.dtimes == datetime(year, month, day))[0][0]
        level_idx = np.argwhere(self.levels == level)[0][0]
        return np.roll(cv2.resize(dfile.variables['air'][time_idx, level_idx, :, :][::-1, :],
                                  dsize=(1440, 720)), 720, axis=1)

    # def get_temperature_day_data_surface(self, year, month, day):
    #     """
    #     Данные по температуре за конкретный день (у поверхности, уровень 1000 мБар).
    #     :param month: месяц
    #     :param day: год
    #     :return:
    #     """
    #     dfile = netCDF4.Dataset(os.path.join(self.data_folder, 'air.sig995.{y}.nc'.format(y=year)), "r",
    #                             format="NETCDF4")
    #     dfile_times = netCDF4.num2date(dfile.variables['time'][:], dfile.variables['time'].units)
    #     time_idx = np.argwhere(dfile_times==datetime(year, month, day))[0][0]
    #     return np.roll(cv2.resize(dfile.variables['air'][time_idx:time_idx+4, :, :].mean(axis=0)[::-1, :],
    #                               dsize=(1440, 720)), 720, axis=1)


def test():
    import matplotlib.pyplot as plt
    re_importer = ReanalysisImporter()
    data = re_importer.get_temperature_day_data(1949, 10, 1, 30)
    plt.contour(data, levels=np.arange(182.5, 230, 2.5))
    plt.colorbar()
    plt.show()

test()
