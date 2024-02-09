def get_value_by_date(input_date, input_dataframe, column_name):
    """
    Get the value from the specified column in the DataFrame based on the provided date.

    Parameters:
    - input_date (str): The date to look for in the DataFrame.
    - input_dataframe (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The column from which to extract the value.

    Returns:
    - value: The value from the specified column corresponding to the given date.
    """
    try:

        filtered_df = input_dataframe[input_dataframe['Date'] == input_date]

        # Extract the value from the specified column
        value = filtered_df[column_name].iloc[0]

        return value

    except IndexError:
        print(f"No data available for the date {input_date} in the DataFrame.")
        return None


