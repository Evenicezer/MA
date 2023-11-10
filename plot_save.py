import pandas as pd
import matplotlib.pyplot as plt




def add_date_name_column(df):

    df['Date'] = pd.to_datetime(df['Date'])
    def get_day_name(date):
        return date.strftime('%A')

    df['date_name'] = df['Date'].apply(get_day_name)

    return df


#---------------------------------------
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

if __name__=='__main__':
    plot_data_dcr()
    add_date_name_column()