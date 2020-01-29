import time
from datetime import datetime, timezone, timedelta
import pandas as pd

import numpy as np





def get_id_list_from_user(mc,user_id, start_date, end_date, device_type=0, params={"max_speed": 0.5, "min_speed": 8,
                                                                                "min_accuracy": 15,
                                                                                "steps_delta": -1,
                                                                                "acc_std": 1.0,
                                                                                "max_dist_km2min": 0.16,
                                                                                "min_dist_km2min": 0.015,
                                                                                "gap_time": 5 * 60}):
    # Query mongo DB for "walking" sessions
    # returns a list of IDs
    # Version 2.21, 18-11-19
    #
    # changed: quicker query, using external params
    #
    # params:
    # max_speed: 0.5 m/s, the lower threshold of max speed (less than that is no movement)
    # min_speed: 6 m/s, the upper threshold of max speed (more than that is probably driving)
    # min_accuracy: 15 m, the lower threshold of min_accuracy (more than that is bad GPS reception)
    # delta_lat: 0.0001, minimum latitude change (about 10 m North/South)
    # steps_delta: 10, minimum number of steps required
    # max_dist_km2min: 0.16 km, maximum allowed distance to be travelled in 1 minute
    # gap_time: 10*60 sec, gap between sessions to be considered a new session

    try:  # check date string format
        start_date_long = mc.convert_to_unix_time(start_date)
        end_date_long = mc.convert_to_unix_time(end_date)
    except:
        raise ValueError("Time string should be of format: 2019-07-28 00:00:00.000+0300")

    agg_code = [
        {"$match": {"user_id": user_id,
                    "timestamp": {"$gt": start_date_long, "$lt": end_date_long},
                    "gpsMaxSpeed": {"$lt": params["min_speed"],"$gt": params["max_speed"]},
                    "distance": {"$gt": params["min_dist_km2min"], "$lt": params["max_dist_km2min"]},
                    "gpsMinAccuracy": {"$lt": params["min_accuracy"]},
                    "accelerationStd": {"$gt": params["acc_std"]},
                    "deltaSteps": {"$gt": params["steps_delta"]}
                    }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "gps0": {"$arrayElemAt": ["$gps", 0]},
                      "gps1": {"$arrayElemAt": ["$gps", -1]},
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "elapsed_time": {"$multiply": [
                          {"$subtract": ["$gps1.timestamp", "$gps0.timestamp"]},
                          0.001]},
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      }}
    ]
    if device_type != 0:
        agg_code[0] = {"$match": {
            "user_id": user_id,
            "device_type": device_type,
            "timestamp": {"$gt": start_date_long, "$lt": end_date_long},
            "gpsMaxSpeed": {"$lt": params["min_speed"]},
            "gpsMinSpeed": {"$gt": params["max_speed"]},
            "distance": {"$gt": params["min_dist_km2min"], "$lt": params["max_dist_km2min"]},
            "gpsMinAccuracy": {"$lt": params["min_accuracy"]},
            "accelerationStd": {"$lt": params["acc_std"]},
            "deltaSteps": {"$gt": params["steps_delta"]}
        }}

    agg = mc.db.sensors.aggregate(agg_code)
    df = pd.DataFrame(agg)
    print(len(df))
    if len(df) == 0:
        print("No data was retrieved for this user")
        return []

    df["_id"] = df._id.apply(str)

    # sort by time
    df = df.sort_values(['timestamp'])
    df = df.reset_index(drop=True)

    df['time_since_start_in_sec'] = (df.timestamp - df.timestamp.min()) / 1000

    # finding where gap between sessions is high -> a new list
    t = np.array(df.time_since_start_in_sec)
    t2 = np.array(df.time_since_start_in_sec.shift())
    try:
        dt = (t - t2)
    except:
        print("The format of timestamp_local is deprecated for this user")
        return []
    dt[0] = 0
    dt = np.array([float(i) for i in dt])
    et = np.array(df.elapsed_time.shift())
    et[0] = 0
    tot_dt = dt - et
    session_idx = np.concatenate([[0], np.where(tot_dt > params["gap_time"])[0], [len(t)]], axis=0)

    list_ids = list(df._id)
    list_of_lists = []
    for i in range(len(session_idx) - 1):
        list_i = list_ids[session_idx[i]:session_idx[i + 1]]
        if len(list_i) > 0:
            list_of_lists.append(list_i)
    return list_of_lists


def get_id_list_from_user_by_createdAt(mc,user_id, start_date, end_date, device_type=0, params={"max_speed": 0.5, "min_speed": 8,
                                                                                "min_accuracy": 15,
                                                                                "steps_delta": -1,
                                                                                "acc_std": 1.0,
                                                                                "max_dist_km2min": 0.16,
                                                                                "min_dist_km2min": 0.015,
                                                                                "gap_time": 5 * 60}):
    # Query mongo DB for "walking" sessions
    # returns a list of IDs
    # Version 3, 29-01-20
    #
    # changed: query by createdAt, returns all the sessions since the timestamp of last data uploaded
    #
    # params:
    # max_speed: 0.5 m/s, the lower threshold of max speed (less than that is no movement)
    # min_speed: 6 m/s, the upper threshold of max speed (more than that is probably driving)
    # min_accuracy: 15 m, the lower threshold of min_accuracy (more than that is bad GPS reception)
    # delta_lat: 0.0001, minimum latitude change (about 10 m North/South)
    # steps_delta: 10, minimum number of steps required
    # max_dist_km2min: 0.16 km, maximum allowed distance to be travelled in 1 minute
    # gap_time: 10*60 sec, gap between sessions to be considered a new session

    try:  # check date string format
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S.%f%z")
        end_date_long = mc.convert_to_unix_time(end_date)
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S.%f%z")

    except:
        raise ValueError("Time string should be of format: 2019-07-28 00:00:00.000+0300")

    agg_code = [
        {"$match": {"user_id": user_id,
                    "createdAt": {"$gt": start_date, "$lt": end_date},
                    "gpsMaxSpeed": {"$lt": params["min_speed"], "$gt": params["max_speed"]},
                    "distance": {"$gt": params["min_dist_km2min"], "$lt": params["max_dist_km2min"]},
                    "gpsMinAccuracy": {"$lt": params["min_accuracy"]},
                    "accelerationStd": {"$gt": params["acc_std"]},
                    "deltaSteps": {"$gt": params["steps_delta"]}
                    }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      }},
    ]
    agg = mc.db.sensors.aggregate(agg_code)
    last_uploaded_data = pd.DataFrame(agg)
    print("last uploaded data size: {}".format(len(last_uploaded_data)))
    if len(last_uploaded_data)==0:
        return [], datetime.utcnow()
    min_timestamp=last_uploaded_data.timestamp.min()
    _id_min=str(last_uploaded_data[last_uploaded_data.timestamp==min_timestamp]._id.iloc[0])

    agg_code = [
        {"$match": {"user_id": user_id,
                    "timestamp": {"$gt": min_timestamp-5*60*60*1000, "$lt": end_date_long},
                    "gpsMaxSpeed": {"$lt": params["min_speed"],"$gt": params["max_speed"]},
                    "distance": {"$gt": params["min_dist_km2min"], "$lt": params["max_dist_km2min"]},
                    "gpsMinAccuracy": {"$lt": params["min_accuracy"]},
                    "accelerationStd": {"$gt": params["acc_std"]},
                    "deltaSteps": {"$gt": params["steps_delta"]}
                    }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "gps0": {"$arrayElemAt": ["$gps", 0]},
                      "gps1": {"$arrayElemAt": ["$gps", -1]},
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "elapsed_time": {"$multiply": [
                          {"$subtract": ["$gps1.timestamp", "$gps0.timestamp"]},
                          0.001]},
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      }}
    ]
    if device_type != 0:
        agg_code[0] = {"$match": {
            "user_id": user_id,
            "device_type": device_type,
            "createdAt": {"$gt": start_date, "$lt": end_date},
            "gpsMaxSpeed": {"$lt": params["min_speed"]},
            "gpsMinSpeed": {"$gt": params["max_speed"]},
            "distance": {"$gt": params["min_dist_km2min"], "$lt": params["max_dist_km2min"]},
            "gpsMinAccuracy": {"$lt": params["min_accuracy"]},
            "accelerationStd": {"$lt": params["acc_std"]},
            "deltaSteps": {"$gt": params["steps_delta"]}
        }}

    agg = mc.db.sensors.aggregate(agg_code)
    df = pd.DataFrame(agg)
    print(len(df))
    if len(df) == 0:
        print("No data was retrieved for this user")
        return [], datetime.utcnow()

    df["_id"] = df._id.apply(str)

    # sort by time
    df = df.sort_values(['timestamp'])
    df = df.reset_index(drop=True)

    df['time_since_start_in_sec'] = (df.timestamp - df.timestamp.min()) / 1000

    # finding where gap between sessions is high -> a new list
    t = np.array(df.time_since_start_in_sec)
    t2 = np.array(df.time_since_start_in_sec.shift())
    try:
        dt = (t - t2)
    except:
        print("The format of timestamp_local is deprecated for this user")
        return [], datetime.utcnow()
    dt[0] = 0
    dt = np.array([float(i) for i in dt])
    et = np.array(df.elapsed_time.shift())
    et[0] = 0
    tot_dt = dt - et
    session_idx = np.concatenate([[0], np.where(tot_dt > params["gap_time"])[0], [len(t)]], axis=0)

    list_ids = list(df._id)
    list_timestamps=list(df.timestamp)
    ids_start_time=list_timestamps[0]
    list_of_lists = []
    to_include=False
    for i in range(len(session_idx) - 1):
        list_i = list_ids[session_idx[i]:session_idx[i + 1]]
        if (not to_include) and (_id_min in list_i):
            to_include=True
            ids_start_time=pd.to_datetime(list_timestamps[session_idx[i]],unit="ms")
        if len(list_i) > 0 and to_include:
            list_of_lists.append(list_i)
    return list_of_lists, ids_start_time
