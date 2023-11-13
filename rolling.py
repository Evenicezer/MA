import pandas as pd
from plot_save import add_date_name_column, plot_data_dcr
import math


# Assuming you have a DataFrame 'df' with 'Date' and columns 'n_confirmed', 'n_recovered', 'n_death'
df = pd.read_csv(r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\German_case_.csv')
df_ = pd.read_csv(r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\German_case_.csv')
#adding name date column
df = add_date_name_column(df)
df_=add_date_name_column(df_)
#print(df.head(35))
#'n_confirmed', 'n_recovered', 'n_death'
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


# reduce the data frame by two rows from the top of df
indices_to_remove = [1,2]
df = df.drop(indices_to_remove)
#---   df is dataframe that starts from monday; the first two rows were removed
#----- df_ is dataframe that starts from saturday..

def smooth_sundays(df):


        # Calculate the rolling mean for columns 'n_confirmed', 'n_recovered', and 'n_death'
        avg_c = (previous_rows['n_confirmed'].rolling(window=2, center=True,closed='left',step=7).mean() +
                 next_rows['n_confirmed'].rolling(window=2, center=True,closed='right',step=7).mean())
        avg_r = (previous_rows['n_recovered'].rolling(window=2, center=True,closed='left',step=7).mean() +
                 next_rows['n_recovered'].rolling(window=2, center=True,closed='right',step=7).mean())
        avg_d = (previous_rows['n_death'].rolling(window=2, center=True,closed='left',step=7).mean() +
                 next_rows['n_death'].rolling(window=2, center=True,closed='right',step=7).mean())

        # Replace the values in the current row

        df.at[index, 'n_confirmed'] = avg_c
        df.at[index, 'n_recovered'] = avg_r
        df.at[index, 'n_death'] = avg_d

    return df
print(smooth_sundays(df))