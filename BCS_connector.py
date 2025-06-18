import pyodbc
import pandas as pd
import warnings
import dask.dataframe as dd
import csv


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)



def connect_db(query):

    server = "10.240.1.129"
    database = "asp_BUILDCONT"
    username = "buildcont_reports"
    password = "ASP4664bu"


    # connect with credentials
    connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}"
    connection = pyodbc.connect(connection_string)

    print("Connected to the BCS SSMS database!!")

    # read data into DataFrame
    df = pd.read_sql_query(query, connection)

        
    # return df
    return df, connection



def reader_df():
      
	query = f"""
	select * from bcs_view_master_data_contacts
	"""
      
	df, connection = connect_db(query)

	connection.close()

	return df  
	