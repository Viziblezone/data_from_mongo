import time
from datetime import datetime
import connect_to_db as mongoConnection
import pandas as pd
import numpy as np
import read_from_mongo


def convert_str_to_datetime(row):
    t = row['timestamp_local']
    return datetime.strptime(t[:-3] + t[-2:], '%Y-%m-%dT%H:%M:%S.%f%z')


def get_id_list_from_user(user_id, start_date, end_date, device_type=0, params={"max_speed": 0.5, "min_speed": 6,
                                                                                "min_accuracy": 15, "delta_lat": 0.0001,
                                                                                "steps_delta": -1,
                                                                                "acc_std": 2.0,
                                                                                "max_dist_km2min": 0.16,
                                                                                "gap_time": 5 * 60}):
    # Query mongo DB for "walking" sessions
    # returns a list of IDs
    # Version 1.07, 16-09-19
    #
    # added: filter based on accelerometer
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
    except:
        raise ValueError("Time string should be of format: 2019-07-28 00:00:00+0300")
    agg_code = [
            {"$match": {"user_id": user_id, "createdAt": {"$gt": start_date}}},
            {"$project": {"speed": "$gps.speed",
                          "_id": "$_id",
                          "timestamp_local": "$timestamp_local",
                          "gps_accuracy": "$gps.accuracy",
                          "latitude": "$gps.latitude",
                          "longitude": "$gps.longitude",
                          "acc_x": "$accelerometer.x_axis",
                          "acc_y": "$accelerometer.y_axis",
                          "acc_z": "$accelerometer.z_axis",
                          "max_gravity_x": {"$max": "$gravity.x_axis"},"max_gravity_y": {"$max": "$gravity.y_axis"},"max_gravity_z": {"$max": "$gravity.z_axis"},
                          "elapsed_time": {"$multiply": ["$gps.number_of_samples", "$sample_period"]},
                          "steps_value": "$steps.value"
                          }},
            {"$project": {"_id": "$_id",
                          "max_speed": {"$max": "$speed"},
                          "min_speed": {"$min": "$speed"},
                          "delta_lat": {"$subtract": [{"$max": "$latitude"}, {"$min": "$latitude"}]},
                          "delta_long": {"$subtract": [{"$max": "$longitude"}, {"$min": "$longitude"}]},
                          "max_lat": {"$max": "$latitude"},
                          "min_accuracy": {"$min": "$gps_accuracy"},
                          "timestamp_local": "$timestamp_local",
                           "steps_value": "$steps_value",
                          "acc_x": "$acc_x","acc_y": "$acc_y","acc_z": "$acc_z",
                          "max_gravity": {"$max": ["$max_gravity_x", "$max_gravity_y", "$max_gravity_z"]},
                          "elapsed_time": "$elapsed_time"}},
            {"$match": {"max_speed": {"$lt": params['min_speed'],"$gt": params['max_speed']},
                        "min_accuracy": {"$lt": params["min_accuracy"]}, "delta_lat": {"$gt": params["delta_lat"]}
                       }},
            {"$project": {"_id": "$_id", "timestamp_local": "$timestamp_local",
                          "elapsed_time": "$elapsed_time", "max_speed": "$max_speed",
                          "delta_lat": "$delta_lat", "delta_long": "$delta_long", "max_lat": "$max_lat",
                          "acc_x": "$acc_x","acc_y": "$acc_y","acc_z": "$acc_z",
                          "max_gravity": "$max_gravity",
                          "steps_value": "$steps_value"}}
             ]
    if device_type != 0:
        agg_code[0]={"$match": {"user_id": user_id, "device_type": device_type, "createdAt": {"$gt": start_date}}}

    agg = mongoConnection.db.sensors.aggregate(agg_code)
    df = pd.DataFrame(agg)
    print(len(df))
    if len(df) == 0:
        print("No data was retrieved for this user")
        return []

    df["_id"] = df._id.apply(str)

    # filtering for the relevant times
    df["timestamp_local_ts"] = df.apply(convert_str_to_datetime, axis=1)
    df = df[df.timestamp_local_ts >= start_date]
    df = df[df.timestamp_local_ts <= end_date]
    if len(df) == 0:
        print("No data was retrieved for this user in the date window")
        return []

    # sort by time
    df = df.sort_values(['timestamp_local_ts'])
    df = df.reset_index(drop=True)

    def make_time_in_sec(row):
        return (row.timestamp_local_ts - start_time).total_seconds()

    start_time = df.timestamp_local_ts.min()
    df['time_since_start_in_sec'] = df.apply(make_time_in_sec, axis=1)

    #acceleration filter
    def calc_std_acceleration(row):
        if row["max_gravity"]>0 and row["max_gravity"]<2:
            factor=9.8
        else:
            factor=1
        l=min(len(row['acc_x']), len(row['acc_y']), len(row['acc_z']))
        r = np.array(row['acc_x'][:l]) ** 2 + np.array(row['acc_y'][:l]) ** 2 + np.array(row['acc_z'][:l]) ** 2
        return factor*np.std(r ** 0.5)
    df['acc_std'] = df.apply(calc_std_acceleration, axis=1)
    df=df[np.array(df.acc_std >= params["acc_std"]) + np.array(df.acc_std==0)]
    if len(df) == 0:
        print("All sessions are less then acc_std")
        return []

    #dropping sessions with total steps < steps_delta
    def calc_steps_delta(row):
        steps=np.array(row["steps_value"])
        steps=np.array([np.float(x) for x in steps])
        steps=steps[steps>0]
        if len(steps)==0:
             return 0
        return max(steps)-min(steps)
    df['steps_delta']=df.apply(calc_steps_delta, axis=1)
    df=df[df.steps_delta>=params["steps_delta"]]
    if len(df) == 0:
        print("All sessions are less then step_threshold")
        return []

    # dropping sessions with too large distance (probably driving)
    dlat_km = np.array(df.delta_lat) * 111.7
    dlong_km = np.array(df.delta_lat) * 40075 / 360 * np.cos(np.array(df.max_lat) / 180 * np.pi)
    dist_km = np.sqrt(dlat_km ** 2 + dlong_km ** 2)
    dist_km2min = dist_km / np.array(df.elapsed_time) * 60
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

