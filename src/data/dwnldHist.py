from tardis_dev import datasets, get_exchange_details
# from tardis_client import datasets, get_exchange_details
import logging
import pandas as pd
# comment out to disable debug logs
logging.basicConfig(level=logging.DEBUG)


# function used by default if not provided via options
def default_file_name(exchange, data_type, date, symbol, format):
    return f"{exchange}_{data_type}_{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# customized get filename function - saves data in nested directory structure
def file_name_nested(exchange, data_type, date, symbol, format):
    return f"{exchange}/{data_type}/{date.strftime('%Y-%m-%d')}_{symbol}.{format}.gz"


# returns data available at https://api.tardis.dev/v1/exchanges/deribit#
# deribit_details = get_exchange_details("deribit")
# print(deribit_details)


d = pd.date_range(start='11/1/2020', end='03/1/2021', freq='MS')
for dt in d.strftime("%Y-%m-%d"):
    datasets.download(
        # one of https://api.tardis.dev/v1/exchanges with supportsDatasets:true - use 'id' value
        exchange="bitstamp",
        # accepted data types - 'datasets.symbols[].dataTypes' field in https://api.tardis.dev/v1/exchanges/deribit,
        # or get those values from 'deribit_details["datasets"]["symbols][]["dataTypes"] dict above
        # data_types=["incremental_book_L2", "trades", "quotes", "derivative_ticker", "book_snapshot_25", "book_snapshot_5"],
         #data_types=["incremental_book_L2", "trades"],
        # data_types=["derivative_ticker"],
        data_types=["book_snapshot_25"],
        # change date ranges as needed to fetch full month or year for example
        from_date=dt,
        # to date is non inclusive
        to_date=dt,
        # accepted values: 'datasets.symbols[].id' field in https://api.tardis.dev/v1/exchanges/deribit
        symbols=["btcgbp", "ethgbp", "xrpgbp", "ltcgbp", "gbpusd", "gbpeur", "linkgbp", "omggbp"],
        # (optional) your API key to get access to non sample data as well
        # api_key="YOUR API KEY",
        # (optional) path where data will be downloaded into, default dir is './datasets'
        # download_dir="./datasets",
        # (optional) - one can customize downloaded file name/path (flat dir strucure, or nested etc) - by default function 'default_file_name' is used
        # get_filename=default_file_name,
        # (optional) file_name_nested will download data to nested directory structure (split by exchange and data type)
        # get_filename=file_name_nested,
    )