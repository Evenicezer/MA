import pandas as pd
import json
#from Data_preprocessing import solve_comulative_data
import matplotlib.pyplot as plt


json_file = r'C:\Users\kida_ev\PycharmProjects\pythonProject1\venv\Lib\site-packages\memilio\data\pydata\Germany\cases_all_germany.json'

def json_to_dataframe(json_file, selected_column="Confirmed"):

    df = pd.read_json(json_file)

    # Create desired structure of df
    new_df = df.pivot(index='Date', columns='ID_County', values=selected_column)

    new_df = new_df.fillna(0)  # Replace NaN with 0

    new_df.rename_axis(None, axis=1, inplace=True)

    # Add a 'Total' column
    new_df['Total'] = new_df.sum(axis=1)

    new_df.reset_index(inplace=True)
    last_df = new_df[['Date', 'Total']]

    return last_df

def json_check_missing_data_csv(json_file, path: str,selected_column):
    df = pd.read_json(json_file)

    #new_df = df.pivot(index='Date', columns='ID_County', values=selected_column)

    #new_df = new_df.fillna(0)  # Replace NaN with 0

    #new_df.rename_axis(None, axis=1, inplace=True)

    # Add a 'Total' column
    #new_df['Total'] = new_df.sum(axis=1)

    #new_df.reset_index(inplace=True)
    df.to_csv(path, index=True)

    return df
save_path=r"C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\Preprocessing\confirmed_germany_csv.csv"
confirmed_csv_data_frame=json_check_missing_data_csv(json_file,save_path,'Confirmed')

#-----------------------
#selected_column = 'Deaths'  # You can change this to 'Deaths' or 'Recovered'
#result_df = json_to_dataframe(json_file, selected_column)
#print(result_df.loc['2020-03-10'].unique()) # for checking in crossreferencing if specific column value may differ

#-----------------------------------------------------------------------
#save them here
#result_df_death=json_to_dataframe(json_file,'Deaths')
#result_df_confirmed=json_to_dataframe(json_file,'Confirmed')
#result_df_recovered=json_to_dataframe(json_file, 'Recovered')
# converting into csv file
csv_death_path = r"C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\Preprocessing\death_data.csv"
csv_recovery_path = r"C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\Preprocessing\recovery_data.csv"
csv_confirmed_path = r"C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\Preprocessing\confirmed_data.csv"
result_df_death.to_csv(csv_death_path, index=False)
result_df_recovered.to_csv(csv_recovery_path, index=False)
result_df_confirmed.to_csv(csv_confirmed_path, index=False)


#column_name=result_df_recovered.columns.tolist()
#print(column_name)
#any_duplicate=result_df_recovered.columns[result_df_recovered.columns.duplicated()]
#print(any_duplicate)

def solve_comulative_single_df(df,column_name:str):
    df[column_name] = df[column_name].diff().fillna(0).astype(int)

    return df
#------------------------------------







if __name__=='__main__':
    json_to_dataframe(json_file)
    print(result_df_recovered.head(20))