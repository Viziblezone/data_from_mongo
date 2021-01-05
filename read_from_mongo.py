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
from itertools import chain
from functools import reduce


class HostnameManager:
    @staticmethod
    def get_host_name(env_type, ssh_only):
        hostname='localhost'
        if env_type=='prod':
            if ssh_only:
                hostname='stats.vizible.zone'
            else:
                hostname='api.vizible.zone'
        elif env_type in 'test':
            if ssh_only:
                hostname='statsdev.vizible.zone'
            else:
                hostname='apidev.vizible.zone'

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
        self.db_write=None

    def connect(self, connection_type, read_only=False):

        MONGO_HOST = HostnameManager.get_host_name(connection_type, True)
        print('\nHostname is: ' + MONGO_HOST)

        MONGO_DB = "VizibleZone"
        MONGO_USER = "ubuntu"
        if (connection_type == 'prod'):
            REMOTE_ADDRESS = ('docdb-2019-06-13-11-43-18.cluster-cybs9fpwjg54.eu-west-1.docdb.amazonaws.com', 27017)
        else:
            REMOTE_ADDRESS = ('vz-dev-docdb-2019-11-10-13-24-25.cluster-cybs9fpwjg54.eu-west-1.docdb.amazonaws.com',27017)

        pem_ca_file = 'rds-combined-ca-bundle.pem'
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
        if (not read_only):
            self.db_write = self.db
        print('db',  self.db)
        print('\nYou are connected to ' + connection_type + ' server\n')
        print(self.db.collection_names())
        return True
    '''
    def log_session(self, session):
        self.db_write.walking_session.insert_one(session)

    def get_sessions_by_date(self, start_date, end_date):

        agg_code = [
            {"$match": {"start_time": {"$gt": start_date, "$lt": end_date}}}
        ]

        agg = self.db.walking_session.aggregate(agg_code)
        return pd.DataFrame(agg)



    '''
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
from collections import defaultdict



# {['latitude':1]},'gps_longitude':1 ,'gps_speed':1


def merge_dicts(dicts):
    mergedict = defaultdict(list)
    for k, v in chain(*[d.items() for d in dicts]):
        mergedict[k].append(v)

    return mergedict



def read_vz_to_dfs(mc, _id):
    dfjson = pd.DataFrame(mc.db.sensors.find({"_id": ObjectId(_id)}))
    if len(dfjson) == 0:
        print("_id {} is empty".format(_id))

    vecs=['gps', 'linear_acceleration', 'gyroscope', 'orientation', 'steps','testing_mode', 'acceleration',
          'gravity', 'magnetometer', 'rotation_matrix']
    #    vecs=['ble_proximity','testing_mode']
    singles = ['_id', 'status', 'user_id', 'user_type', 'device_type',"deltaSteps", "distance", 'sample_period',
               'timestamp_local', 'createdAt', 'updatedAt', '__v']
    singles_df = pd.DataFrame.from_dict({column: [dfjson[column][0]] for column in singles if column in dfjson.columns},
                                        orient='columns', dtype=None, columns=None)
    vecs_dic = {column: pd.DataFrame(dfjson[column][0]).drop(["_id"], axis=1, errors='ignore').add_prefix(column + "_")
                for column in vecs if column in dfjson.columns}
    vecs_dic['singles_df'] = singles_df
    return vecs_dic



def get_dfs_for_ids(mc, ids):
    md = merge_dicts([read_vz_to_dfs(mc, _id) for _id in ids])
    return {k: pd.concat(md[k]) for k in md}


def get_timestamp_local(mc, _id):
    agg = mc.db.sensors.aggregate(
        [{"$match": {"_id": ObjectId(_id)}}, {"$project": {"timestamp_local": "$timestamp_local"}}])
    print('in get_timestamp_local')
    return pd.DataFrame(agg)['ftimestamp_local'][0]

def get_user_id(mc, _id):
    agg = mc.db.sensors.aggregate(
        [{"$match": {"_id": ObjectId(_id)}}, {"$project": {"user_id": "$user_id"}}])
    return pd.DataFrame(agg)['user_id'][0]




def set_ts(df):
    tscol = [col for col in df.columns if '_timestamp' in col][0]
    df = df.rename(columns={tscol: "timestamp"}).sort_values('timestamp')
    df = df[df.timestamp > 0]  # ignore rows with blank time

    return df


def get_df_for_ids(mc, ids):

    print(len(ids), ' ids')
    print(ids)
    # list_ids=list(df_walk._id)

    dfs_dic = get_dfs_for_ids(mc, ids)

    dfs_dic_with_ts = {k: set_ts(dfs_dic[k]) for k in dfs_dic if
                       any([col for col in dfs_dic[k].columns if '_timestamp' in col])}

    min_ts = min([dfs_dic_with_ts[k]['timestamp'].min() for k in dfs_dic_with_ts.keys()])
    max_ts = max([dfs_dic_with_ts[k]['timestamp'].max() for k in dfs_dic_with_ts.keys()])

    timestamp_df = pd.DataFrame(data={'timestamp': np.linspace(min_ts, max_ts, int((max_ts - min_ts) / 100))})

    gps_df = dfs_dic_with_ts.pop('gps')

    gps_df = pd.merge_asof(timestamp_df, gps_df, on='timestamp', direction='nearest', tolerance=2000)

    df_AS = reduce(lambda left, right: pd.merge_asof(left,
                                                     right.drop([c for c in left.columns if c != 'timestamp'], axis=1,
                                                                errors='ignore'), on='timestamp', direction='nearest',
                                                     tolerance=100),
                   dict(list({'time': timestamp_df}.items()) + list(dfs_dic_with_ts.items())).values())
    gps_df = gps_df[['timestamp', 'gps_accuracy', 'gps_altitude', 'gps_bearing', 'gps_bearing_accuracy', 'gps_latitude',
                     'gps_longitude', 'gps_speed']]

    df_AS = df_AS.merge(gps_df, on='timestamp')
    singles_df=dfs_dic['singles_df']
    df_AS["timestamps_utc"] = pd.to_datetime(np.array(df_AS.timestamp), unit='ms')

    df_AS["timestamp"] = df_AS.timestamps_utc.dt.tz_localize('UTC').dt.tz_convert('Asia/Hebron')
    return pd.concat([df_AS, singles_df.append([singles_df] * (len(df_AS) - 1), ignore_index=True)], axis=1)

