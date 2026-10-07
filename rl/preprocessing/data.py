import datetime

import pandas as pd


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    _data = pd.read_csv(file_name, index_col=False)
    # Files written with DataFrame.to_csv() carry an unnamed index column.
    _data = _data.drop(columns=[c for c in _data.columns if str(c).startswith("Unnamed:")])
    if "date" in _data.columns:
        _data["date"] = _data["date"].astype(str)
    return _data


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start (inclusive), end (exclusive)
    :return: (df) pandas dataframe indexed by day number
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)
    return time
