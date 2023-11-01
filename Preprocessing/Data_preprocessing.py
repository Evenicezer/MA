import pandas as pd
import numpy as np
import matplotlib.pyplot as plt





data_on_Case='cases-rki-by-ags.csv'
data_on_Death='deaths-rki-by-ags.csv'
column_name_date='time_iso8601'
start_date = '2020-03-02'
end_date = '2020-04-30'
def csv_to_df(data:str,column_name_date:str, start_date:str,end_date:str,sum_cases_or_deaths:str):

    data = pd.read_csv(data, delimiter=',')

    #  datetime format
    data[column_name_date] = pd.to_datetime(data[column_name_date])



    date_range_df = data[(data[column_name_date] >= start_date) & (data[column_name_date] <= end_date)][[column_name_date, sum_cases_or_deaths]]

    #
    return(date_range_df)
date_case_df=csv_to_df(data_on_Case,'time_iso8601',start_date,end_date,'sum_cases')
date_death_df=csv_to_df(data_on_Death,'time_iso8601',start_date,end_date,'sum_deaths')
#print(date_case_df)
#print(date_death_df)

def merge_df(df1,df2,column_name_date:str):
    merged_df=pd.merge(df1, df2, on=column_name_date, how='inner')
    return merged_df
# merged df for case and death along with date
case_death_df=merge_df(date_death_df,date_case_df,column_name_date)
#print(case_death_df)
def solve_comulative_data(df,column_name1:str,column_name2:str):
    df['daily_deaths'] = df[column_name1].diff().fillna(df[column_name1][0]).astype(int)
    df['daily_cases'] = df[column_name2].diff().fillna(df[column_name2][0]).astype(int)

    return df
print(solve_comulative_data(case_death_df,'sum_deaths','sum_cases'))
#-------------------------------------------------------------------------------------

#case_death_df['']
#deaths_column = case_death_df['sum_deaths']
#cases_column = case_death_df['sum_cases']
#------------------------------------------------------------------------------------
case_death_df['time_iso8601'] = pd.to_datetime(case_death_df['time_iso8601'])




fig, ax = plt.subplots(figsize=(10, 6))


ax.plot(case_death_df['time_iso8601'], solve_comulative_data(case_death_df,'sum_deaths','sum_cases')['daily_deaths'], label='Daily Deaths', color='red')


ax.plot(case_death_df['time_iso8601'], solve_comulative_data(case_death_df,'sum_deaths','sum_cases')['daily_cases'], label='Daily Cases', color='blue')

#ax.plot(case_death_df['time_iso8601'], case_death_df['sum_deaths'], label='Sum Deaths', color='orange')


#ax.plot(case_death_df['time_iso8601'], case_death_df['sum_cases'], label='Sum Cases', color='cyan')
#
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title('COVID-19 Deaths and Cases Over Time')

plt.xticks(rotation=45)

ax.legend()

plt.tight_layout()
plt.show()




