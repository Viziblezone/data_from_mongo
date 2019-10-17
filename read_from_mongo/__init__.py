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

    df_tmp = pd.DataFrame()
    dfjson = pd.DataFrame(mongoConnection.db.sensors.find({"_id": ObjectId(_id)}, {"_id": 1, 'gps': 1, 'timestamps': 1}))
    # find number_of_samples
    vecs_dct = {}
    singles_dct = {}
    for column in dfjson.columns:
        #        print(column)
        try:
            if dfjson[column][0]["number_of_samples"] > 0:
                vecs_dct[column] = dfjson[column][0]["number_of_samples"]
        except:
            singles_dct[column] = dfjson[column][0]
    nos = min(vecs_dct.values())  # number_of_samples
    if max(vecs_dct.values()) > min(vecs_dct.values()):
        print('NOTE: number_of_samples varies by {}'.format(max(vecs_dct.values()) - min(vecs_dct.values())))
    for column in vecs_dct.keys():
        for key in dfjson[column][0].keys():
            if key != "number_of_samples" and len(dfjson[column][0][key]) >= nos:
                col_name = column + '_' + key
                col_name = col_name.replace('bearing', 'azimuth')
                col_name = col_name.replace('testing_mode_value', 'testing_mode')
                df_tmp[col_name] = dfjson[column][0][key][:nos]
    for column in singles_dct.keys():
        df_tmp[column] = singles_dct[column]
    # correct and add columns

    # clean zeros in the lat/long reading
    df_tmp = df_tmp[df_tmp["gps_latitude"] < df_tmp["gps_latitude"].median() + 1]
    df_tmp = df_tmp[df_tmp["gps_latitude"] > df_tmp["gps_latitude"].median() - 1]

    def convert_str_to_datetime(row):
        #    print(row)
        try:
            return datetime.strptime(row['timestamps_value'], '%Y-%m-%dT%H:%M:%S.%f')
        except:
            t = row['timestamps_value']
            return datetime.strptime(t[:-3] + t[-2:], '%Y-%m-%dT%H:%M:%S.%f%z')

    df_tmp['timestamp'] = df_tmp.apply(convert_str_to_datetime, axis=1)

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

