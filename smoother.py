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
    # Find rows with 'Sunday' in the 'date_name' column
    sunday_rows = df[df['date_name'] == 'Sunday']
    print(sunday_rows)

    for index, sunday_row in sunday_rows.iloc[2:-2].iterrows():
        previous_rows = df.loc[index - 2: index - 1]  # Take two rows up
        next_rows = df.loc[index : index + 2]  # Take two rows down
        print(previous_rows['n_recovered'], previous_rows['n_confirmed'],previous_rows['n_death'] , sep='/n')


        # Calculate the average values for columns 'n_confirmed', 'n_recovered', and 'n_death'
        avg_c = (previous_rows['n_confirmed'] + next_rows['n_confirmed']).mean()
        #print(f'avg_c_before: {avg_c}')
        avg_r = sum(previous_rows['n_recovered'] + next_rows['n_recovered']) / len(previous_rows['n_recovered'] + next_rows['n_recovered'])
        avg_d = sum(previous_rows['n_death'] + next_rows['n_death']) / len(previous_rows['n_death'] + next_rows['n_death'])

        # Replace the values in the current row
        df.at[index, 'n_confirmed'] = avg_c
        df.at[index, 'n_recovered'] = avg_r
        df.at[index, 'n_death'] = avg_d
        #print(f'avg_c: {avg_d}')
    return df

#df_smoothed = smooth_sundays(df)
# Print the modified DataFrame
#print(df_smoothed.head(35))

print(smooth_sundays(df))
# plotting
#plot_data_dcr(smooth_sundays(df),'n_death','n_confirmed','n_recovered','smoothed_m_a_2_2')
#plot_data_dcr(df_,'n_death','n_confirmed','n_recovered',)

