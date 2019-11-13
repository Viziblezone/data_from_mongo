import time
from datetime import datetime, timezone, timedelta
import connect_to_db as mongoConnection
import pandas as pd
import numpy as np
import read_from_mongo


def convert_str_to_datetime(row):
    t = row['timestamp_local']
    return datetime.strptime(t[:-3] + t[-2:], '%Y-%m-%dT%H:%M:%S.%f%z')


def get_id_list_from_user(user_id, start_date, end_date, device_type=0, params={"max_speed": 0.5, "min_speed": 8.5,
                                                                                "min_accuracy": 15, "delta_lat": 0.0001,
                                                                                "steps_delta": -1,
                                                                                "acc_std": 1.0,
                                                                                "max_dist_km2min": 0.16,
                                                                                "gap_time": 5 * 60}):
    # Query mongo DB for "walking" sessions
    # returns a list of IDs
    # Version 2.14, 12-11-19
    #
    # changed: compatible to iPhone
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
        start_date = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S%z")
        end_date = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S%z")
        t0 = datetime(1970, 1, 1, tzinfo=timezone(timedelta(seconds=0)))
        start_date_long = (start_date - t0).total_seconds() * 1000
        end_date_long = (end_date - t0).total_seconds() * 1000
    except:
        raise ValueError("Time string should be of format: 2019-07-28 00:00:00+0300")

    agg_code = [
        {"$match": {"user_id": user_id,
                    "timestamp": {"$gt": start_date_long, "$lt": end_date_long},
                    "gps.speed": {"$gt": params["max_speed"], "$lt": params["min_speed"]},
                    "gps.accuracy": {"$lt": params["min_accuracy"]}
                    }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "gps": "$gps",
                      "gps0": {"$arrayElemAt": ["$gps", 0]},
                      "gps1": {"$arrayElemAt": ["$gps", -1]},
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      "linear_acceleration": "$linear_acceleration",
                      "acceleration": "$acceleration",
                      "steps": "$steps",
                      "steps0": {"$arrayElemAt": ["$steps", 0]},
                      "steps1": {"$arrayElemAt": ["$steps", -1]}
                      }},
        {"$project": {"device_type": "$device_type", "createdAt": "$createdAt",
                      "gps": "$gps",
                      "elapsed_time": {"$multiply": [
                          {"$subtract": ["$gps1.timestamp", "$gps0.timestamp"]},
                          0.001]},
                      "_id": "$_id",
                      "timestamp_local": "$timestamp_local",
                      "timestamp": "$timestamp",
                      "linear_acceleration": "$linear_acceleration",
                      "acceleration": "$acceleration",
                      "steps": "$steps",
                      "delta_steps": {"$subtract": ["$steps1.value", "$steps0.value"]}
                      }},
        {"$match": {"delta_steps": {"$gt": params['steps_delta']}
                    }},
    ]
    if device_type != 0:
        agg_code[0] = {"$match": {"user_id": user_id,
                                  "device_type": device_type,
                                  "timestamp": {"$gt": start_date_long, "$lt": end_date_long},
                                  "gps.speed": {"$gt": params["max_speed"], "$lt": params["min_speed"]},
                                  "gps.accuracy": {"$lt": params["min_accuracy"]}
                                  }}

    agg = mongoConnection.db.sensors.aggregate(agg_code)
    df = pd.DataFrame(agg)
    print(len(df))
    if len(df) == 0:
        print("No data was retrieved for this user")
        return []

    df["_id"] = df["_id"].apply(str)

    # sort by time
    df = df.sort_values(['timestamp'])
    df = df.reset_index(drop=True)

    df['time_since_start_in_sec'] = (df.timestamp - df.timestamp.min()) / 1000

    # acceleration filter
    def calc_std_acceleration(row):
        la=pd.DataFrame(row.linear_acceleration)
        acc=pd.DataFrame(row.acceleration)
        if len(la)>0:
            la["tot"]=(la.x_axis**2+la.y_axis**2+la.z_axis**2)**0.5
            return la["tot"].std()
        elif len(acc)>0:
            la=acc
            la["tot"]=(la.x_axis**2+la.y_axis**2+la.z_axis**2)**0.5
            return la["tot"].std()
        return np.nan

    df['acc_std'] = df.apply(calc_std_acceleration, axis=1)
    # print(df['acc_std'])

    df = df[np.array(df.acc_std >= params["acc_std"]) + np.array(df.acc_std == 0)]
    if len(df) == 0:
        print("All sessions are less then acc_std")
        return []

    # dropping sessions with total steps < steps_delta
    def calc_steps_delta(row):
        steps = pd.DataFrame(row.steps)
        steps = steps[steps.value > 0]
        if len(steps) == 0:
            return 0
        return steps.value.max() - steps.value.min()

    df['calc_steps_delta'] = df.apply(calc_steps_delta, axis=1)
    df = df[df.calc_steps_delta >= params["steps_delta"]]
    if len(df) == 0:
        print("All sessions are less then step_threshold")
        return []

    # dropping sessions with too large distance (probably driving)
    def calc_dist_km(row):
        gps = pd.DataFrame(row.gps)
        dlat_km = (gps.latitude.max() - gps.latitude.min()) * 111.7
        dlong_km = (gps.longitude.max() - gps.longitude.min()) * 40075 / 360 * np.cos(gps.latitude.mean() / 180 * np.pi)
        dist_km = np.sqrt(dlat_km ** 2 + dlong_km ** 2)
        return dist_km

    df['dist_km'] = df.apply(calc_dist_km, axis=1)
    dist_km2min = np.array(df.dist_km) / np.array(df.elapsed_time) * 60
    df = df.iloc[dist_km2min < params["max_dist_km2min"]]
    if len(df) == 0:
        print("All data for this user is too fast for walking!")
        return []

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
