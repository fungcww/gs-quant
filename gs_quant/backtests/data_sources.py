"""
Copyright 2019 Goldman Sachs.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
"""

from dataclasses import dataclass, field
from dataclasses_json import dataclass_json, config
import datetime as dt
from enum import Enum
import sqlite3
import numpy as np
import pandas as pd
from typing import Union, Iterable, ClassVar, List, Optional

from gs_quant.backtests.core import ValuationFixingType
from gs_quant.base import field_metadata, static_field
from gs_quant.data import DataFrequency, Dataset
from gs_quant.instrument import Instrument
from gs_quant.json_convertors import decode_pandas_series, encode_pandas_series


class MissingDataStrategy(Enum):
    fill_forward = 'fill_forward'
    interpolate = 'interpolate'
    fail = 'fail'


@dataclass_json
@dataclass
class DataSource:
    __sub_classes: ClassVar[List[type]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DataSource.__sub_classes.append(cls)

    @staticmethod
    def sub_classes():
        return tuple(DataSource.__sub_classes)

    def get_data(self, state, **kwargs):
        raise RuntimeError("Implemented by subclass")

    def get_data_range(self, start: Union[dt.date, dt.datetime], end: Union[dt.date, dt.datetime, int], **kwargs):
        raise RuntimeError("Implemented by subclass")


@dataclass_json
@dataclass
class GsDataSource(DataSource):
    data_set: str
    asset_id: str
    min_date: dt.date = None
    max_date: dt.date = None
    value_header: str = 'rate'
    class_type: str = static_field('gs_data_source')

    def __post_init__(self):
        self.loaded_data = None

    def get_data(self, state: Union[dt.date, dt.datetime] = None, **kwargs):
        if self.loaded_data is None:
            ds = Dataset(self.data_set)
            if self.min_date:
                self.loaded_data = ds.get_data(self.min_date, self.max_date, assetId=(self.asset_id,))
            elif state is not None:
                return ds.get_data(state, state, assetId=(self.asset_id,))[self.value_header]
            else:
                return ds.get_data(dt.datetime(2000, 1, 1), **kwargs)[self.value_header]
        return self.loaded_data[self.value_header].at[pd.to_datetime(state)]

    def get_data_range(self, start: Union[dt.date, dt.datetime], end: Union[dt.date, dt.datetime, int], **kwargs):
        if self.loaded_data is None:
            ds = Dataset(self.data_set)
            if self.asset_id is not None:
                kwargs['assetId'] = (self.asset_id,)
            if self.min_date:
                self.loaded_data = ds.get_data(self.min_date, self.max_date, **kwargs)
            else:
                self.loaded_data = ds.get_data(start, self.max_date, **kwargs)
        if isinstance(end, int):
            return self.loaded_data.loc[self.loaded_data.index < start].tail(end)
        return self.loaded_data.loc[(start < self.loaded_data.index) & (self.loaded_data.index <= end)]


@dataclass_json
@dataclass
class GenericDataSource(DataSource):
    """
    A data source which holds a pandas series indexed by date or datetime
    :param data_set: a pandas dataframe indexed by date or datetime
    :param missing_data_strategy: MissingDataStrategy which defines behaviour if data is missing, will only take
                                  effect if using get_data, get_data_range has no expectations of the number of
                                  expected data points.
    """

    data_set: pd.Series = field(
        default=None, metadata=config(decoder=decode_pandas_series, encoder=encode_pandas_series)
    )

    missing_data_strategy: MissingDataStrategy = field(default=MissingDataStrategy.fail, metadata=field_metadata)
    class_type: str = static_field('generic_data_source')

    def __eq__(self, other):
        if not isinstance(other, GenericDataSource):
            return False
        return self.missing_data_strategy == other.missing_data_strategy and self.data_set.equals(other.data_set)

    def __post_init__(self):
        self._tz_aware = isinstance(self.data_set.index[0], dt.datetime) and self.data_set.index[0].tzinfo is not None

    def get_data(self, state: Union[dt.date, dt.datetime, Iterable]):
        """
        Get the value of the dataset at a time or date.  If a list of dates or times is provided return list of values
        :param state: a date, datetime or a list of dates or datetimes
        :return: float value
        """

        if state is None:
            return self.data_set

        if isinstance(state, Iterable):
            return [self.get_data(i) for i in state]

        if self._tz_aware and (state.tzinfo is None or state.tzinfo.utcoffset(state) is None):
            state = state.replace(tzinfo=dt.timezone.utc)
        if pd.Timestamp(state) in self.data_set:
            return self.data_set[pd.Timestamp(state)]
        elif state in self.data_set or self.missing_data_strategy == MissingDataStrategy.fail:
            return self.data_set[state]
        else:
            if isinstance(self.data_set.index, pd.DatetimeIndex):
                self.data_set.at[pd.to_datetime(state)] = np.nan
                self.data_set = self.data_set.sort_index()
            else:
                self.data_set.at[state] = np.nan
            self.data_set.sort_index()
            if self.missing_data_strategy == MissingDataStrategy.interpolate:
                self.data_set = self.data_set.interpolate()
            elif self.missing_data_strategy == MissingDataStrategy.fill_forward:
                self.data_set = self.data_set.ffill()
            else:
                raise RuntimeError(f'unrecognised missing data strategy: {str(self.missing_data_strategy)}')
            return (
                self.data_set[pd.to_datetime(state)]
                if isinstance(self.data_set.index, pd.DatetimeIndex)
                else self.data_set[state]
            )

    def get_data_range(self, start: Union[dt.date, dt.datetime], end: Union[dt.date, dt.datetime, int]):
        """
        get a range of values from the dataset.
        :param start: a date or datetime
        :param end: a date, datetime or an int.  If an int is provided we return that many data points back from the
                    start date
        :return: pd.Series
        """

        if isinstance(end, int):
            return self.data_set.loc[self.data_set.index < start].tail(end)
        return self.data_set.loc[(start < self.data_set.index) & (self.data_set.index <= end)]

# customization
@dataclass_json
@dataclass
class SQLiteDataSource(DataSource):
    """
    Loads a time series from a SQLite database using pandas (`read_sql_query`) and exposes it as a
    :class:`DataSource` for triggers and :class:`DataManager` (same contract as :class:`GenericDataSource`).

    The SQL should return at least two columns: a date/datetime column and a numeric value column
    (defaults: first column = dates, first remaining column = values). Duplicate timestamps keep the
    last row.

    :param db_path: Path to the SQLite database file (for example ``market.db``).
    :param sql: SQL string passed to :func:`pandas.read_sql_query`.
    :param date_column: Name of the date/datetime column; if omitted, the first column is used.
    :param value_column: Name of the value column; if omitted, the first column after the date column is used.
    :param index_at_time: If set, each calendar date from the query is combined with this clock time (for example
        EOD :class:`datetime.time` matching :class:`PredefinedAssetEngine` valuation time) so that
        :meth:`get_data` matches :class:`~datetime.datetime` states from the engine.
    :param missing_data_strategy: Same semantics as :class:`GenericDataSource` for :meth:`get_data`.
    """

    db_path: str
    sql: str
    date_column: Optional[str] = None
    value_column: Optional[str] = None
    index_at_time: Optional[dt.time] = field(default=None, metadata=field_metadata)
    missing_data_strategy: MissingDataStrategy = field(default=MissingDataStrategy.fail, metadata=field_metadata)
    class_type: str = static_field('sqlite_data_source')

    def __post_init__(self):
        self._wrapped: Optional[GenericDataSource] = None

    def _ensure_wrapped(self) -> GenericDataSource:
        if self._wrapped is not None:
            return self._wrapped
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(self.sql, conn)
        if df.empty:
            raise RuntimeError('SQLite query returned no rows; cannot build time series.')
        date_col = self.date_column or df.columns[0]
        if date_col not in df.columns:
            raise ValueError(f'date_column {date_col!r} not in query columns {list(df.columns)}')
        value_candidates = [c for c in df.columns if c != date_col]
        if self.value_column is not None:
            if self.value_column not in df.columns:
                raise ValueError(f'value_column {self.value_column!r} not in query columns {list(df.columns)}')
            value_col = self.value_column
        elif value_candidates:
            value_col = value_candidates[0]
        else:
            raise ValueError('Could not infer value column; specify value_column or return a value column in SQL.')
        idx = pd.to_datetime(df[date_col])
        if self.index_at_time is not None:
            idx = pd.DatetimeIndex(
                [dt.datetime.combine(pd.Timestamp(ts).date(), self.index_at_time) for ts in idx]
            )
        else:
            idx = pd.DatetimeIndex(idx)
        series = pd.Series(df[value_col].to_numpy(), index=idx, name=value_col)
        series = series[~series.index.duplicated(keep='last')].sort_index()
        self._wrapped = GenericDataSource(series, self.missing_data_strategy)
        return self._wrapped

    def get_data(self, state: Union[dt.date, dt.datetime, Iterable] = None, **kwargs):
        return self._ensure_wrapped().get_data(state)

    def get_data_range(
        self,
        start: Union[dt.date, dt.datetime],
        end: Union[dt.date, dt.datetime, int],
        **kwargs,
    ):
        wrapped = self._ensure_wrapped()
        start_ts = pd.Timestamp(start)
        if isinstance(end, int):
            return wrapped.get_data_range(start_ts, end)
        return wrapped.get_data_range(start_ts, pd.Timestamp(end))


@dataclass_json
@dataclass
class DataManager:
    def __post_init__(self):
        self._data_sources = {}

    def add_data_source(
        self,
        series: Union[pd.Series, DataSource],
        data_freq: DataFrequency,
        instrument: Instrument,
        valuation_type: ValuationFixingType,
    ):
        if not isinstance(series, DataSource) and not len(series):
            return
        if instrument.name is None:
            raise RuntimeError('Please add a name identify your instrument')
        key = (data_freq, instrument.name, valuation_type)
        if key in self._data_sources:
            raise RuntimeError(
                'A dataset with this frequency instrument name and valuation type already added to Data Manager'
            )
        self._data_sources[key] = GenericDataSource(series) if isinstance(series, pd.Series) else series

    def get_data(self, state: Union[dt.date, dt.datetime], instrument: Instrument, valuation_type: ValuationFixingType):
        key = (
            DataFrequency.REAL_TIME if isinstance(state, dt.datetime) else DataFrequency.DAILY,
            instrument.name.split('_')[-1],
            valuation_type,
        )
        return self._data_sources[key].get_data(state)

    def get_data_range(
        self,
        start: Union[dt.date, dt.datetime],
        end: Union[dt.date, dt.datetime],
        instrument: Instrument,
        valuation_type: ValuationFixingType,
    ):
        key = (
            DataFrequency.REAL_TIME if isinstance(start, dt.datetime) else DataFrequency.DAILY,
            instrument.name.split('_')[-1],
            valuation_type,
        )
        return self._data_sources[key].get_data_range(start, end)
