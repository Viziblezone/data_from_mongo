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
from pyqtree import Index
from shapely import geometry
import random
import shapely.geometry as ge

import itertools
import networkx as nx

import shapely
import random

from shapely.geometry import LineString, Point
import connect_to_db as mongoConnection


# {['latitude':1]},'gps_longitude':1 ,'gps_speed':1



def convert_str_to_datetime(row):
    t = row['timestamp_local']
    return datetime.strptime(t[:-3] + t[-2:], '%Y-%m-%dT%H:%M:%S.%f%z')


# In[77]:


import math


# {['latitude':1]},'gps_longitude':1 ,'gps_speed':1

def read_VZ_from_mongo(_id):
    dfjson = pd.DataFrame(mongoConnection.db.sensors.find({"_id": ObjectId(_id)}, {"_id": 1, 'gps': 1, 'user_id': 1, 'device_type': 1, "timestamp_local": 1}))
    if len(dfjson) == 0:
        print("_id {} is empty".format(_id))
        return dfjson
    # find number_of_samples
    vecs = ['gps']
    singles = ['_id', 'user_id', 'device_type', "timestamp_local"]
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




# In[67]:


# server.stop()


# In[68]:


# pd.DataFrame(db.sensors.find({ 'user_id': "rZBbw12NtJYWsDqgomNJSg9z7ii1" }))


# In[78]:


def get_df_for_ids(ids):

    print(len(ids), ' ids')
    print(ids)
    # list_ids=list(df_walk._id)
    df_vz = pd.DataFrame()
    for _id in ids:
        try:
            df_tmp = read_VZ_from_mongo(_id)
            df_vz = pd.concat([df_vz, df_tmp], axis=0)
        except:
            print('problem with id {}'.format(_id))

    #    df_vz['timestamp']=df_vz.apply(convert_str_to_datetime, axis=1)
    df_vz = df_vz.sort_values(['timestamp'])
    return df_vz.reset_index(drop=True)

