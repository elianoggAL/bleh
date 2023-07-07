#!/usr/bin/env python
# coding: utf-8

# #### This notebook:
# - pulls Thunder features and predictions for the last 90+ days
# - gets residuals and saves to another Snowflake table including weekly refreshed Thunder residuals

# In[ ]:


import pandas as pd
import numpy as np

### Azure SQL (conn) and Snowflake (engine) Connections
from build_db_conn import conn, engine

### Table name to refresh weekly with historical thunder predictions
from set_historical_table_name import (
    historical_table_name,
    get_historical_predictions,
)

pd.set_option("display.max_columns", 100)


# ### Load Updated Historical Thunder Predictions

# In[ ]:


print(
    f"Snowflake table name for historical Thunder predictions:\n{historical_table_name}"
)


# In[ ]:


final_hist_df = get_historical_predictions(tbl_name=historical_table_name)

final_hist_df.columns = final_hist_df.columns.str.upper()

print(final_hist_df.shape)
final_hist_df.head()


# ### Get Thunder Residuals

# In[ ]:


final_hist_df.columns = final_hist_df.columns.str.lower()

print(final_hist_df.shape)
final_hist_df.head()


# In[ ]:


### Pull pickupapptlatest from lod__loadboard_base

qry = """
    select
        lb.loadnumber,
        lb.pickupapptlatest,
        current_date() as current_date
    from
        dapl_raw.accelerateprod.lod__loadboard_base lb
    where
        lb.pickupapptlatest >= '2023-03-01'
    order by pickupapptlatest asc;
"""

date_df = pd.read_sql_query(
    qry,
    engine,
)

date_df["current_date"] = pd.to_datetime(date_df.current_date)

print(date_df.shape)
date_df.head()


# In[ ]:


df = (
    final_hist_df.merge(date_df, on="loadnumber", how="left")
    .sort_values(by=["pickupapptlatest"], ascending=True)
    .reset_index(drop=True)
)

print(df.shape)


# In[ ]:


### Replace negative values for avail2cutoff_hours with zero
print(
    f"# rows with negative avail2cutoff_hours (replaced with zero): {df[df.avail2cutoff_hours < 0].shape[0]}"
)
df.avail2cutoff_hours = np.where(df.avail2cutoff_hours < 0, 0, df.avail2cutoff_hours)

### Get number of days back column
if max(df.pickupapptlatest) < max(df.current_date):
    df["current_date"] = max(df.pickupapptlatest)

df["days_back"] = (df["current_date"] - df["pickupapptlatest"]).dt.days

### Get residuals
df["residual"] = df["actualcost"] - df["predcost"]
df["residual_nv"] = df["actualnv"] - df["predictednv"]

df.head()


# In[ ]:


### Get flags for number of days back

days = [30, 45, 60, 75, 90]
for d in days:
    df[f"days_back_{d}"] = np.where(df.days_back <= d, 1, 0)

df.head()


# ### Sub-group by number of days back

# In[ ]:


quantiles = np.linspace(0.55, 0.95, 9).round(3).tolist()

selected_days = 75
min_loadcount = 300

coarse_groups = ["isreefer", "lead_time_group", "miles_group"]
bin_cols = ["lead_time_group", "miles_group"]


# In[ ]:


def subset_by_days(
    df,
    ndays: int = 60,
    min_loadcount: int = 300,
    groupbys: list = ["isreefer", "lead_time_group", "miles_group", "rateapiflag"],
):
    """
    subsets dataframe by days back,
    creates lead time and miles bins,
    groups by groupbys list to get load counts,
    returns subset df and grouped by df
    """
    print(f"{ndays} days back:")
    sub_df = df[(df[f"days_back_{ndays}"] == 1)].reset_index(drop=True)
    sub_df["lead_time_group"] = pd.qcut(sub_df["avail2cutoff_hours"], 3, precision=0)
    sub_df["miles_group"] = pd.qcut(sub_df["loadmiles"], 3, precision=0)
    print(sub_df.shape)

    tmp = (
        sub_df.groupby(groupbys)
        .size()
        .reset_index()
        .rename(columns={0: "group_load_count"})
        .rename_axis("group_index")
        .reset_index()
    )
    tmp["group_index"] = tmp["group_index"] + 1
    # tmp["days_back"] = ndays

    group_index = list(tmp.group_index.unique())
    print(f"# groups: {len(group_index)}")
    print(
        f"{round((len(tmp[tmp.group_load_count < min_loadcount]) / len(tmp))*100)}% of groups have < 300 loads\n"
    )

    return sub_df, tmp


# ### Residuals for coarse groups (for selected days back)

# In[ ]:


sub_df, coarse_groups_df = subset_by_days(
    df, ndays=selected_days, min_loadcount=min_loadcount, groupbys=coarse_groups
)


# In[ ]:


coarse_groups_df


# In[ ]:


print(sub_df.shape)
sub_df = sub_df.merge(coarse_groups_df, on=coarse_groups)
print(sub_df.shape)

sub_df.head()


# In[ ]:


coarse_groups_resid = coarse_groups_df.copy()

tmp = (
    sub_df.groupby("group_index")[["residual", "residual_nv"]]
    .quantile(quantiles)
    .reset_index()
)
tmp = tmp.rename(columns={"level_1": "quantile"})

coarse_groups_resid = coarse_groups_resid.merge(tmp, on=["group_index"])
coarse_groups_resid


# ### Save coarse groups (larger population only)

# In[ ]:


groups_resid = coarse_groups_resid.copy()

# 1) create left and right interval values for groups
for c in bin_cols:
    groups_resid[f"{c}_left"] = groups_resid.apply(lambda row: row[c].left, axis=1)
    groups_resid[f"{c}_right_closed"] = groups_resid.apply(
        lambda row: row[c].right, axis=1
    )
    groups_resid[c] = groups_resid[c].astype("string")

# 2) drop bin_cols
groups_resid = groups_resid.drop(columns=["lead_time_group", "miles_group"])

print(groups_resid.shape)
groups_resid


# In[ ]:


groups_resid.columns = groups_resid.columns.str.upper()
groups_resid


# In[ ]:


# groups_resid.dtypes


# In[ ]:


# file_name = f"test_thunder_residuals.csv"
# groups_resid.to_csv(file_name, index=False)


# ### Truncate Snowflake Table and Reload with Thunder Residuals!
# 
# ### groups_resid -> Table 2

# In[ ]:

import os
import pandas as pd
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine
from PyInstaller.utils.hooks import copy_metadata
from snowflake.connector.pandas_tools import write_pandas, pd_writer
from datetime import date

#Dummay data frame
#dummy = pd.DataFrame([[1,2,3], [4,5,6], [7,8,9]])

table_name="thunder_residuals"
schema_name="RATE_API"

#integrate data into snowflake table
url = URL(
    user="XXXXX",
    password="XXXXX",
    account="arrive.east-us-2.azure",
    warehouse="XXXXX",
    database="XXXXX",
    schema=schema_name
    )    

engine1 = create_engine(url)


def create_table(out_df, table_name, idx=False):
    connection = engine1.connect()
    connection.execute('truncate table ' +table_name + ';')  
    try:
        out_df.to_sql(
            table_name, connection, if_exists="append", index=idx, method=pd_writer
        )

    except ConnectionError:
        print("Unable to connect to database!")

    finally:
        connection.close()
        engine1.dispose()

    return True

df = pd.DataFrame(groups_resid)
df['DATE_SYNCHED'] = date.today() 


df.columns = map(lambda x: str(x).upper(), df.columns)
print(df.head)

create_table(df, table_name)

