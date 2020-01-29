import math

import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np
import geopy.distance
import imageio
from timeit import default_timer as timer
import pandas as pd
import seaborn as sns
import scipy
from scipy.stats import norm
import requests
import json
import os
from os.path import join, dirname, abspath
from glob import glob
import io
import pathlib
from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from shapely import geometry
import random
import shapely.geometry as ge

import itertools
import networkx as nx

import shapely
import random

from shapely.geometry import LineString, Point
from datetime import datetime, timezone, timedelta


from sshtunnel import SSHTunnelForwarder
import os.path


class HostnameManager:
    @staticmethod
    def get_host_name(host_type):
        hostname='localhost'
        if host_type in 'prod':
            hostname='automotive.vizible.zone'
        elif host_type in 'test':
            hostname='dev.vizible.zone'
        return hostname

    @staticmethod
    def get_pem_file_name(host_type):
        pem_file_name= ''
        if host_type in 'prod':
            pem_file_name= 'viziblezone-prod.pem'
        elif host_type in 'test':
            pem_file_name= 'automotive-dev.pem'
        return pem_file_name


class MongoConnection:

    def __init__(self):
        self.client=None
        self.server=None
        self.db=None

    def connect(self, connection_type):

        MONGO_HOST = HostnameManager.get_host_name(connection_type)
        MONGO_DB = "VizibleZone"
        MONGO_USER = "ubuntu"
        if (connection_type == 'prod'):
            REMOTE_ADDRESS = ('docdb-2019-06-13-11-43-18.cluster-cybs9fpwjg54.eu-west-1.docdb.amazonaws.com', 27017)
        else:
            REMOTE_ADDRESS = ('vz-dev-docdb-2019-11-10-13-24-25.cluster-cybs9fpwjg54.eu-west-1.docdb.amazonaws.com',27017)

        pem_ca_file = 'rds-combined-ca-bundle (1).pem'
        pem_server_file = HostnameManager.get_pem_file_name(connection_type)

        pem_path = '../pems/'
        if not os.path.exists(pem_path + pem_server_file):
            pem_path = pem_path[1:]

        self.server = SSHTunnelForwarder(
            MONGO_HOST,
            ssh_pkey=pem_path + pem_server_file,
            ssh_username=MONGO_USER,
            remote_bind_address=REMOTE_ADDRESS
        )

        self.server.start()

        if (connection_type == 'prod'):
            self.client = MongoClient('127.0.0.1',
                                 self.server.local_bind_port,
                                 username='viziblezone',
                                 password='vz123456',
                                 ssl=True,
                                 ssl_match_hostname=False,
                                 ssl_ca_certs=(pem_path + pem_ca_file),
                                 authMechanism='SCRAM-SHA-1')  # server.local_bind_port is assigned local port
        else:
            self.client = MongoClient('127.0.0.1',
                                  self.server.local_bind_port,
                                  username='dev',
                                  password='protectingpedestrians',
                                  ssl=True,
                                  ssl_match_hostname=False,
                                  ssl_ca_certs=(pem_path + pem_ca_file),
                                  authMechanism='SCRAM-SHA-1')  # server.local_bind_port is assigned local port

        self.db = self.client[MONGO_DB]
        print('db',  self.db)
        print('\nYou are connected to ' + connection_type + ' server\n')
        print(self.db.collection_names())
        return True

    def log_session(self, session):
        self.db.walking_session.insert_one(session)

    def get_sessions_by_date(self, start_date, end_date):

        agg_code = [
            {"$match": {"start_time": {"$gt": start_date, "$lt": end_date}}}
        ]

        agg = self.db.walking_session.aggregate(agg_code)
        return pd.DataFrame(agg)


    def dispose(self):
        print("Closing connection to DB")

        self.client.close()
        self.server.stop()

    @staticmethod
    def convert_to_unix_time(date):
        t0 = datetime(1970, 1, 1, tzinfo=timezone(timedelta(seconds=0)))
        try:  # check date string format
            date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f%z")
            return (date - t0).total_seconds() * 1000
        except:
            raise ValueError("Time string should be of format: 2019-07-28 00:00:00.000+0300")



import math


# {['latitude':1]},'gps_longitude':1 ,'gps_speed':1

def read_VZ_from_mongo(mc,_id):
    dfjson = pd.DataFrame(mc.db.sensors.find({"_id": ObjectId(_id)}, {"_id": 1, 'gps': 1, 'user_id': 1, 'device_type': 1, "timestamp_local": 1, "createdAt": 1}))
    if len(dfjson) == 0:
        print("_id {} is empty".format(_id))
        return dfjson
    # find number_of_samples
    vecs = ['gps']
    singles = ['_id', 'user_id', 'device_type', "timestamp_local", "createdAt"]
    vecs_dfs = []
    min_ts = np.inf
    max_ts = 0
    for column in vecs:
        if column in dfjson.columns:
            t = pd.DataFrame(dfjson[column][0])
            if len(t) > 0:
                t.columns = map(str.lower, t.columns)
                min_ts = min(min_ts, t.timestamp.min())
                max_ts = max(max_ts, t.timestamp.max())
                merge_on = round(t.timestamp / 50)  # time resolution 50ms
                t = t.drop(["timestamp"], axis=1)
                if "_id" in t.columns:
                    t = t.drop(["_id"], axis=1)
                t = t.add_prefix(column + "_")
                t["merge_on"] = merge_on
                t = t.drop_duplicates(subset=["merge_on"])
                vecs_dfs.append(t)
        else:
            print("{} is missing from _id {}".format(column, _id))
    df_tmp = pd.DataFrame()
    df_tmp["merge_on"] = np.arange(round(min_ts / 50), round(max_ts / 50))
    df_tmp["timestamps_utc"] = pd.to_datetime(np.array(df_tmp.merge_on) * 50, unit='ms')

    for df_i in vecs_dfs:
        df_tmp = pd.merge(left=df_tmp, right=df_i, on="merge_on", how="left")
    df_tmp = df_tmp.fillna(method="ffill")
    df_tmp = df_tmp.iloc[np.arange(1, len(df_tmp), 2)]  # take only 100ms
    df_tmp = df_tmp.reset_index(drop=True)

    for column in singles:
        if column in dfjson.columns:
            df_tmp[column] = dfjson[column][0]
        else:
            print("{} is missing from _id {}".format(column, _id))
    df_tmp = df_tmp.rename(columns={"gps_bearing": "gps_azimuth",
                                    "gps_bearing_accuracy": "gps_azimuth_accuracy", 'testing_mode_value': 'testing_mode'})

    # correct and add columns

    # create timestamps_value (local)
    s = df_tmp.timestamp_local.iloc[0]
    seconds_tz = int(s[-5:-3]) * 3600 + int(s[-2:]) * 60
    df_tmp["timestamp"] = df_tmp.timestamps_utc.dt.tz_localize('UTC').dt.tz_convert(seconds_tz)
    df_tmp["timestamps_value"] = df_tmp["timestamp"]
    # clean zeros in the lat/long reading
    df_tmp = df_tmp[df_tmp["gps_latitude"] < df_tmp["gps_latitude"].median() + 1]
    df_tmp = df_tmp[df_tmp["gps_latitude"] > df_tmp["gps_latitude"].median() - 1]


    # def calc_tot_acceleration(row):
    #     r = row['linear_acceleration_x_axis'] ** 2 + row['linear_acceleration_y_axis'] ** 2 + row[
    #         'linear_acceleration_z_axis'] ** 2
    #     return r ** 0.5
    #
    #
    # def calc_tot_gyro(row):
    #     r = row['gyroscope_x_axis'] ** 2 + row['gyroscope_y_axis'] ** 2 + row['gyroscope_z_axis'] ** 2
    #     return r ** 0.5
    #
    # df_tmp['linear_acceleration'] = df_tmp.apply(calc_tot_acceleration, axis=1)
    # df_tmp['gyroscope_tot'] = df_tmp.apply(calc_tot_gyro, axis=1)

    return df_tmp







def get_df_for_ids(mc,ids):

    print(len(ids), ' ids')
    print(ids)
    # list_ids=list(df_walk._id)
    df_vz = pd.DataFrame()
    for _id in ids:
        try:
            df_tmp = read_VZ_from_mongo(mc,_id)
            df_vz = pd.concat([df_vz, df_tmp], axis=0)
        except:
            print('problem with id {}'.format(_id))

    #    df_vz['timestamp']=df_vz.apply(convert_str_to_datetime, axis=1)
    df_vz = df_vz.sort_values(['timestamp'])
    return df_vz.reset_index(drop=True)

