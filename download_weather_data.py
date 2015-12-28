"""Retrieve the gfs analysis dataset between 2 points in time."""

# TODO(johmathe): it seems that every night NCEP republishes their
# dataset, cutting the download and making this script fail miserably.
# We should be robust to that and resume the whole thing.

import os
import logging
import urllib
import datetime

YEAR_START = 2010
YEAR_END = 2015


def get_analysis_data():
    pass


class Error404(Exception):
    pass


class BetterURLOpener(urllib.FancyURLopener):
    def http_error_default(self, url, fp, errcode, errmsg, headers):
        raise Error404


url_retriever = BetterURLOpener()


def fetch_noaa_data(start, end):

    path = '/spindisk/weather_data/gfs_anl/'
    url_prefix = 'http://nomads.ncdc.noaa.gov/data/gfsanl/'
    delta = end - start

    urls = []
    for i in range(delta.days + 1):
        date = start + datetime.timedelta(days=i)
        for hour in [0, 6, 12, 18]:
            for suffix in [0]:
                url = (
                    url_prefix + '/%04d%02d' %
                    (date.year, date.month) + '/%04d%02d%02d/' %
                    (date.year, date.month,
                     date.day) + 'gfsanl_4_%04d%02d%02d_%02d00_%03d.grb2' %
                    (date.year, date.month, date.day, hour, suffix))
                urls.append(url)
    for url in urls:
        try:
            logging.info('retrieving %s -> %s', url, path)
            target = path + '/' + url.split('/')[-1]
            if os.path.exists(target):
                logging.info('file %s already exists', target)
                continue
            url_retriever.retrieve(url, path + '/' + url.split('/')[-1])
        except Error404:
            logging.warning('404 when retrieving %s' % url)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fetch_noaa_data(datetime.date(YEAR_START, 1, 1), datetime.date(YEAR_END,
                                                                   12, 31))
