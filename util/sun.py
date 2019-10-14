from sunpy.net import hek
from datetime import datetime

hek_client = hek.HEKClient()
start_time_hmi = datetime(2010, 2, 10, 0)   # Start of the HMI device flight


def is_ao_flare(ar_number):
    """
    Check if active region was a solar flare.
    If true, print information about the time when event occured.
    """
    result_fl = hek_client.search(hek.attrs.Time(start_time_hmi, datetime.now()),
                                  hek.attrs.EventType('FL'),
                                  hek.attrs.AR.NOAANum==ar_number)
    if result_fl:
        print('FLARE {fl_no}: \nSTART: {start_time} \nPEAK: {peak_time} \nEND: {end_time}'.format(
                fl_no = ar_number,
                start_time = result_fl[0]['event_starttime'],
                peak_time = result_fl[0]['event_peaktime'],
                end_time = result_fl[0]['event_endtime']))
        return True
    else:
        return False


def get_ar_flares(ar_number, verbose=True):
    """
    Get flares list for active region.
    """
    hek_client = hek.HEKClient()
    result_fl = hek_client.search(hek.attrs.Time(start_time_hmi, datetime.now()), hek.attrs.EventType('FL'), hek.attrs.AR.NOAANum==ar_number)
    result = [e for e in result_fl if e['search_instrument']=='GOES']
    if verbose:
        for e in result:
            print(e['fl_goescls'], ' at ', e['event_peaktime'])
    return result
