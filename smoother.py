import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator, NullFormatter
from iprocessor import add_day_name_column, plot_data_dcr, add_date_name_column, smooth_sundays_ssmt,smooth_sundays_ssm,\
    smooth_sundays_rolling_ssm_w3,smooth_sundays_rolling_ssm_w5,smooth_sundays_rolling_w5_r,smooth_sundays_rolling_w5_l,smooth_sundays_rolling_w7_l,\
        smooth_sundays_rolling_w7_r,smooth_sundays_rolling_ssm_w3_smt,plot_data_dcr_multi
import math


# ---------------------------------------------------------------------------- Importing data and reading in df
df = pd.read_csv(r'C:\Users\kida_ev\PycharmProjects\pythonProject1\MA_EVNZR\German_case_.csv')

# -------------------------------------------------------------------------------Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')


#------------------------------------------------------------------------------  Modification
# Add the 'days' column
df= add_day_name_column(df)
df = add_date_name_column(df)

# Drop the first two rows
#df = df.iloc[2:]
#-------------------------------------------------------------------------------  Processing row data'assuming we have run the add_day_name_column function

#print(smooth_sundays_ssmt(df)[['Date','n_recovered','n_confirmed','n_confirmed']].head(20))

#--------------------------------------------------------------------------------


#------------------------------------------------------------------------------- plotting
#plot_data_dcr(df,'n_death','n_confirmed','n_recovered','original')
#plot_data_dcr(smooth_sundays_ssmt(df),'n_death','n_confirmed','n_recovered','smoothed_ssmt')
#plot_data_dcr(smooth_sundays_ssm(df),'n_death','n_confirmed','n_recovered','smoothed_ssm')
#plot_data_dcr(smooth_sundays_rolling_ssm_w3(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_ssm_w3')
#plot_data_dcr(smooth_sundays_rolling_ssm_w5(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_fssmt_w5')
#plot_data_dcr(smooth_sundays_rolling_ssm_w3_smt(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_smt_w3')
#plot_data_dcr(smooth_sundays_rolling_w7_r(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w7_r')
#plot_data_dcr(smooth_sundays_rolling_w5_r(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w5_r')
#plot_data_dcr(smooth_sundays_rolling_w7_l(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w7_l')
#plot_data_dcr(smooth_sundays_rolling_w5_l(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w5_l')
#plot_data_dcr(smooth_sundays_rolling_w5_l(df),'n_death','n_confirmed','n_recovered','s')#moothed_rolling_w5_l')
#plot_data_dcr(df_,'n_death','n_confirmed','n_recovered',)

#(smooth_sundays(df))['n_recovered'].plot()
#df_smoothed_rolling_w7_r = smooth_sundays_rolling_w7_r(df)
#df_smoothed_rolling_w5_r = smooth_sundays_rolling_w5_r(df)
#df_origin = df
#df_smoothed_rolling_w3_r = smooth_sundays_rolling_ssm_w3(df)

#plot_data_dcr_multi(df_origin, df_smoothed_rolling_w3_r, df_smoothed_rolling_w5_r,df_smoothed_rolling_w7_r,'n_death' )

#----------------------------------------------------------------------------------------------

# Use Seaborn color palette for better color choices
sns.set_palette("husl")

# Create a figure and axis with a larger size
fig, ax = plt.subplots(figsize=(15, 9))

# Plot the data with different line styles and colors
ax.plot(df['days'], df['n_confirmed'], label='Original', linestyle=':', linewidth=2, color='blue')
ax.plot(df['days'], smooth_sundays_rolling_ssm_w3(df)['n_confirmed'], label='Rolling W3', linestyle='-', linewidth=1, color='orange')
ax.plot(df['days'], smooth_sundays_rolling_w5_l(df)['n_confirmed'], label='Rolling W5', linestyle='-', linewidth=1, color= 'red')
ax.plot(df['days'], smooth_sundays_rolling_w7_l(df)['n_confirmed'], label='Rolling W7', linestyle='-', linewidth=2, color='black')

# Set axis labels and title
ax.set_xlabel("Days")
ax.set_ylabel("Confirmed Cases")
ax.set_title("Comparative Analysis of Confirmed Cases")

# Add gridlines
ax.grid(True, linestyle='--', alpha=0.7)

# Adding legend with a slightly transparent background
ax.legend(framealpha=0.8)

# Save the figure
plt.savefig('o_w3-7_l_15_9_confirmed')

# Display the plot
plt.show()
#---------------
