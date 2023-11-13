import pandas as pd
from iprocessor import add_day_name_column, plot_data_dcr, add_date_name_column, smooth_sundays_ssmt,smooth_sundays_ssm,\
    smooth_sundays_rolling_ssm_w3,smooth_sundays_rolling_ssm_w5,smooth_sundays_rolling_w5_r,smooth_sundays_rolling_w5_l,smooth_sundays_rolling_w7_l,\
        smooth_sundays_rolling_w7_r,smooth_sundays_rolling_ssm_w3_smt
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
plot_data_dcr(df,'n_death','n_confirmed','n_recovered','original')
plot_data_dcr(smooth_sundays_ssmt(df),'n_death','n_confirmed','n_recovered','smoothed_ssmt')
plot_data_dcr(smooth_sundays_ssm(df),'n_death','n_confirmed','n_recovered','smoothed_ssm')
plot_data_dcr(smooth_sundays_rolling_ssm_w3(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_ssm_w3')
plot_data_dcr(smooth_sundays_rolling_ssm_w5(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_fssmt_w5')
plot_data_dcr(smooth_sundays_rolling_ssm_w3_smt(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_smt_w3')
plot_data_dcr(smooth_sundays_rolling_w7_r(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w7_r')
plot_data_dcr(smooth_sundays_rolling_w5_r(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w5_r')
plot_data_dcr(smooth_sundays_rolling_w7_l(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w7_l')
plot_data_dcr(smooth_sundays_rolling_w5_l(df),'n_death','n_confirmed','n_recovered','smoothed_rolling_w5_l')
#
#plot_data_dcr(df_,'n_death','n_confirmed','n_recovered',)

#(smooth_sundays(df))['n_recovered'].plot()