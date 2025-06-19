import pandas as pd
import BCS_connector


df = BCS_connector.reader_df()

df.to_excel("Contacts_data.xlsx")