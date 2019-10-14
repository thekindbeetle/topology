import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
import os

from datetime import datetime
from astropy import units as u
from astropy.coordinates import SkyCoord

import sunpy
import sunpy.map
from sunpy.net import hek, attrs as a
from sunpy.coordinates import frames
from sunpy.database import Database

plt.style.use("ggplot")

# 2010/02/10 is the start time of SDO (with HMI device).
START_TIME_HMI = datetime(2010, 2, 10, 0)


def _get_coords_arcsec(coord1, coord2, obstime):
    """
    Convert geliographic coords (in degrees) to helioprojective coords (in arcseconds).
    :param coord1:
    :param coord2:
    :param obstime:
    :return:
    """
    src_coord = SkyCoord(coord1 * u.deg, coord2 * u.deg, frame=frames.HeliographicStonyhurst, obstime=obstime)
    result_coord = src_coord.transform_to(frames.Helioprojective)
    return result_coord.Tx.value, result_coord.Ty.value


def download_ar_hmi_data(ar_number, period=3600, database_path='D:/fits', half_image_size=150,
                         plot_evolution=False, silent=False):
    """
    Download Active Region data (in FITS format)
    :param silent:
        If True, do not show log messages.
    :param plot_evolution:
        Plot every 10th image.
    :param half_image_size:
        Half size of square cutting region.
    :param database_path:
        Path of local database.
    :param ar_number:
        Number of Active Region.
    :param period:
        Frequency of images (in seconds), 600 by default (1 image per 10 minutes).
    :return:
    """
    # Initialize database connection
    os.chdir(database_path)
    database = Database('sqlite:///sunpydata.sqlite')
    database.default_waveunit = 'angstrom'

    client = hek.HEKClient()

    result = client.search(hek.attrs.Time(START_TIME_HMI, datetime.now()), hek.attrs.OBS.Observatory == 'SDO',
                           hek.attrs.AR.NOAANum == ar_number)
    result = [e for e in result if e['search_instrument'] == 'HMI']

    print('{num} events in AR {ar} found'.format(num=len(result), ar=ar_number))

    if not result:
        print('AR {ar}: No events found!'.format(ar=ar_number))
        return

    file_names = []
    centers_x = []
    centers_y = []
    for event in result:
        start_time = datetime.strptime(event['event_starttime'], '%Y-%m-%dT%H:%M:%S')
        end_time = datetime.strptime(event['event_endtime'], '%Y-%m-%dT%H:%M:%S')
        query = a.Time(start_time, end_time) & \
                a.Instrument('HMI') & \
                a.vso.Provider('JSOC') & \
                a.vso.Physobs('LOS_magnetic_field')
        database.fetch(query & a.vso.Sample(period * u.s), path="./data")
        database.commit()
        entries = database.search(query)
        for e in entries:
            # В базе есть поле hdu_index, но селектора для него не предусмотрено
            # Сраные индусы
            # Приходится делать дополнительный отбор, чтобы не дублировать файлы.
            # Плюс, выбираем только события, близкие к центру (+- 45 градусов широта / долгота)
            # Потому что нам нафиг не нужны события на краю диска (там ни хрена не видно).
            if (e.hdu_index == 0) and (np.abs(event['event_coord1']) < 45) and (np.abs(event['event_coord2']) < 45):
                file_names.append(e.path)
                centers_x.append(event['event_coord1'])
                centers_y.append(event['event_coord2'])
        if not silent:
            print('Events from {start} to {end} processed'.format(start=start_time, end=end_time))

    # Сглаживаем список координат центров AR по долготе (чтобы не было скачков).
    # centers_x.index(centers_x[-1]) — индекс первого события в последней серии с несмещённым центром AR.
    params = np.polyfit([0, centers_x.index(centers_x[-1])], [centers_x[0], centers_x[-1]], deg=1)
    x = list(range(len(centers_x)))
    centers_x_fit = np.polyval(params, x)  # Новый список координаты X центров.

    for file_num in list(range(len(file_names))):
        fname = file_names[file_num]
        try:
            mp = sunpy.map.Map(fname)
        except TypeError:
            print('File {f} is corrupted'.format(f=fname))
            continue

        # Переводим координаты в гелиопроективные
        center_point = _get_coords_arcsec(centers_x_fit[file_num], centers_y[file_num], mp.date)

        # Считаем координаты углов в пикселях
        lower_left = SkyCoord((center_point[0] - half_image_size) * u.arcsec,
                              (center_point[1] - half_image_size) * u.arcsec, frame=mp.coordinate_frame)

        upper_right = SkyCoord((center_point[0] + half_image_size) * u.arcsec,
                               (center_point[1] + half_image_size) * u.arcsec, frame=mp.coordinate_frame)

        # Вырезаем кусок из карты
        cut_map = mp.submap(lower_left, upper_right)

        if not os.path.exists('./processed/AR{ar_number}'.format(ar_number=ar_number)):
            os.makedirs('./processed/AR{ar_number}'.format(ar_number=ar_number))

        # save_path = './processed/AR{ar_number}/{fnum}.fits'.format(ar_number=ar_number,fnum=file_num)
        short_name = os.path.basename(fname)
        save_path = 'C:/data/hmi/processed/AR{ar_number}/{fname}'.format(ar_number=ar_number, fname=short_name)

        if os.path.exists(save_path):
            os.remove(save_path)
        cut_map.save(save_path)
        print('File {f} saved'.format(f=save_path))

        # If needed plot every 10th submap.
        if plot_evolution and (file_num % 10 == 1):
            plt.matshow(cut_map.data, cmap='gray', norm=colors.Normalize(vmin=-1000, vmax=1000))
            plt.colorbar()

    if plot_evolution:
        plt.show()
