import matplotlib.pyplot as plt
import pandas as pd
from Read_J_df import json_to_dataframe, solve_comulative_single_df

json_file = r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\cases_all_germany.json'

start_date = '2020-03-10'
end_date = '2020-07-10'

result_df_death=json_to_dataframe(json_file,'Deaths')
result_df_confirmed=json_to_dataframe(json_file,'Confirmed')
result_df_recovered=json_to_dataframe(json_file, 'Recovered')


#solving for data filter
selected_result_df_death = result_df_death.loc[(result_df_death['Date'] >= start_date) & (result_df_death['Date'] <= end_date)]
selected_result_df_confirmed = result_df_confirmed.loc[(result_df_confirmed['Date'] >= start_date) & (result_df_confirmed['Date'] <= end_date)]
selected_result_df_recovered = result_df_recovered.loc[(result_df_recovered['Date'] >= start_date) & (result_df_recovered['Date'] <= end_date)]





#solving for cumulative data
#df_recovered= solve_comulative_single_df(selected_result_df_recovered,'Total')
#df_confirmed= solve_comulative_single_df(selected_result_df_confirmed,'Total')
#df_death= solve_comulative_single_df(selected_result_df_death,'Total')
#print(df_confirmed)

# death
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(selected_result_df_death['Date'], selected_result_df_death['Total'], label='Daily Deaths', color='red')
# confirmed
ax.plot(selected_result_df_confirmed['Date'], selected_result_df_confirmed['Total'], label='Daily Confirmed', color='blue')
# recovered
ax.plot(selected_result_df_recovered['Date'], selected_result_df_recovered['Total'], label='Daily Recovered', color='orange')


#
ax.set_xlabel('Date')
ax.set_ylabel('Count')
ax.set_title('COVID-19 Deaths and Cases Over Time')

plt.xticks(rotation=45)

ax.legend()

plt.tight_layout()
plt.show()
