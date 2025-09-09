import requests
from typing import Optional
from bs4 import BeautifulSoup
import logging

import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    DateTime,
    Text,
    insert,
    select,
    update,
)
from datetime import datetime
import time
from random import randint
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils.db_utils import get_engine
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert

# read data from db
def read_data_from_db():
    engine = get_engine()

    df_raw = pd.read_sql_table('apartment_data', engine)
