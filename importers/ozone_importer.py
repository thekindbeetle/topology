import os
import numpy as np
from pandas import date_range
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


class OzoneImporter:
    data_dir = ''

    longitude_large = np.linspace(-180, 180, 1441, endpoint=True)
    latitude_large = np.linspace(-90, 90, 721, endpoint=True)

    longitude_small = np.linspace(-180, 180, 361, endpoint=True)
    latitude_small = np.linspace(-90, 90, 181, endpoint=True)

    def __init__(self, datadir='/home/alexeev/data_alexeev/ozone/toms.gsfc.nasa.gov'):
        self.data_dir = datadir

    @staticmethod
    def _parse(filename, large=True):
        """
        Парсим файл с данными по озону за день.
        :param filename: 
        :param large: 
        :return: 
        """
        if large:
            # Большое разрешение
            field = np.zeros((720, 1440))
            with open(filename, 'r') as f:
                # Пропускаем заголовок
                for _ in range(3):
                    f.readline()

                for i in range(720):
                    next_line = ''.join([next(f) for x in range(58)])
                    next_line = next_line.replace('\n ', '').replace('\r', '')
                    next_line = next_line[1:-15]
                    field[i] = np.array(list(map(int, [next_line[3 * t: 3 * (t + 1)] for t in range(1440)])))
        else:
            # Маленькое разрешение
            field = np.zeros((180, 360))
            with open(filename, 'r') as f:
                # Пропускаем заголовок
                for _ in range(3):
                    f.readline()

                for i in range(180):
                    next_line = ''.join([next(f) for x in range(15)])
                    next_line = next_line.replace('\n ', '').replace('\r', '')
                    next_line = next_line[1:-15]
                    field[i] = np.array(list(map(int, [next_line[3 * t: 3 * (t + 1)] for t in range(360)])))
        return field

    def get_fields(self, start_date, finish_date, large=True):
        """
        Набор полей для заданного диапазона дат
        :param start_date:
        :param finish_date:
        :param large:
        :return:
        """
        days = date_range(start_date, finish_date, freq='1D')
        day_num = len(days)
        fields = np.zeros((day_num, 720, 1440)) if large else np.zeros((day_num, 180, 360))

        for d in range(day_num):
            if large:
                filename = os.path.join(self.data_dir, 'large/Y{year}/L3e_ozone_omi_{date}.txt'.
                                        format(year=days[d].year, date=days[d].strftime('%Y%m%d')))
            else:
                filename = os.path.join(self.data_dir, 'small/Y{year}/L3_ozone_omi_{date}.txt'.
                                        format(year=days[d].year, date=days[d].strftime('%Y%m%d')))
            try:
                fields[d] = OzoneImporter._parse(filename=filename, large=large)
            except FileNotFoundError:
                print('Файл {filename} не найден!'.format(filename=filename))

        fields[fields == 0] = np.nan

        return fields

    def draw_map(self, field, large=True, vmin=200, vmax=600, cmap='jet', projection='ortho'):
        """
        Нарисовать поле озона в заданной проекции.
        :param field:
        :param large:
        :param vmin:
        :param vmax:
        :param cmap:
        :param projection:
        :return:
        """
        if large:
            longitude, latitude = self.longitude_large, self.latitude_large
        else:
            longitude, latitude = self.longitude_small, self.latitude_small

        if projection == 'ortho':
            m = Basemap(projection=projection, lon_0=0, lat_0=70)
        elif projection == 'mercator':
            m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80,
                        llcrnrlon=-180, urcrnrlon=180, lat_ts=20, resolution='c')
        else:
            raise Exception('Projection {0} is not supported'.format(projection))

        xx, yy = np.meshgrid(longitude, latitude)
        m.pcolormesh(xx, yy, field, latlon=True, cmap=cmap, vmin=vmin, vmax=vmax)
        m.drawcoastlines()
        m.colorbar()
        return plt.gcf()

    def ozone_mean(self, start_date, finish_date, large=True):
        """
        Средняя карта озона для промежутка времени.
        :param start_date:
        :param finish_date:
        :param large:
        :return:
        """
        fields = self.get_fields(start_date, finish_date, large)
        return np.nanmean(fields, axis=0)
