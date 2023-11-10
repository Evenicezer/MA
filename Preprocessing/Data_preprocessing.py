import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

#--------------------------------
json_file=r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\cases_all_germany.json'
df = pd.read_json(json_file)
df = df.sort_values(by='Date', ascending=True)
df

#------------------------------------
start_date = '2020-10-10'
end_date = '2021-07-10'
df_by_date=df.loc[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
#--------------------------------------------------------------------------------
def plot_data_dcr(df, column_d: str, column_c: str, column_r: str, output_filename=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['Date'], df[column_d], label='Daily Deaths', color='red')
    ax.plot(df['Date'], df[column_c], label='Daily Confirmed', color='blue')
    ax.plot(df['Date'], df[column_r], label='Daily Recovered', color='orange')

    ax.set_xlabel('Date')
    ax.set_ylabel('Count')
    ax.set_title('COVID-19 Deaths and Cases Over Time')

    plt.xticks(rotation=45)

    ax.legend()

    plt.tight_layout()
    if output_filename:
        # Save the image to the specified output filename
        plt.savefig(output_filename)
    else:
        # Display the plot
        plt.show()
#----------------------------------------------------------------------------------
plot_data_dcr(df_by_date,'Deaths','Confirmed','Recovered')
#---------------------------------------------------------------------------------
def solve_comulative_data(df,former_c1,new_c1,former_c2,new_c2,former_c3,new_c3):
    df[new_c1] = df[former_c1].diff().fillna(df[former_c1][0]).astype(int)
    df[new_c2] = df[former_c2].diff().fillna(df[former_c2][0]).astype(int)
    df[new_c3] = df[former_c3].diff().fillna(df[former_c3][0]).astype(int)

    return df
#--------------------------------------------------------------------------------------
def s_comulative_data(df,former_c1,new_c1,former_c2,new_c2,former_c3,new_c3):
    df[new_c1] = df[former_c1].diff().fillna(0).astype(int).copy()
    df[new_c2] = df[former_c2].diff().fillna(0).astype(int).copy()
    df[new_c3] = df[former_c3].diff().fillna(0).astype(int).copy()
    df.to_csv('new_german_rdc_3_8.csv', index=False)

    return df
#-------------------------------------------------------------------------------
w_df=s_comulative_data(df_by_date,'Confirmed','n_confirmed','Deaths','n_death','Recovered','n_recovered')
w_df.tail()
#------------------------------------------------------------------------------
plot_data_dcr(w_df,'n_death','n_confirmed','n_recovered')
#----------------------------------------------------------------------------
w_df['Infection_case']=np.nan# creating new column
n_confirmed_len=len(w_df['n_confirmed'])
infection_case_len=len(w_df['Infection_case'])
start_row=5
num_values_to_copy = min(n_confirmed_len - start_row, infection_case_len)
w_df['Infection_case'].iloc[:num_values_to_copy] = w_df['n_confirmed'].iloc[start_row:start_row + num_values_to_copy]

f_w_df = w_df.iloc[:-5]  #filtered dataframe; since the last 5 rows of the

f_w_df['Infection_case'] = f_w_df['Infection_case'].astype(int)
f_w_df.to_csv('German_case_.csv', index=False)
f_w_df.tail(20)
#--------------------------------------------------------------------------------
nan_count =f_w_df['Infection_case'].isna().sum()
print(f"Number of NaN values in 'Infection_case' column: {nan_count}")
nan_indices = f_w_df[w_df['Infection_case'].isna()].index
print("Indices of rows with NaN values in 'Infection_case' column:")
print(nan_indices)
#---------------------------------------------------------------------------------