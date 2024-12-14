import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import sqlite3

# Database setup
def init_db():
    conn = sqlite3.connect('forecasting.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS time_series_data (
                    id INTEGER PRIMARY KEY,
                    date TEXT UNIQUE,
                    value REAL
                )''')
    conn.commit()
    conn.close()

init_db()

# Add data to the database
def add_data_to_db(date, value):
    conn = sqlite3.connect('forecasting.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO time_series_data (date, value) VALUES (?, ?)", (date, value))
        conn.commit()
    except sqlite3.IntegrityError:
        st.error("Date already exists in the database. Please use a unique date.")
    conn.close()

# Update data in the database
def update_data_in_db(date, value):
    conn = sqlite3.connect('forecasting.db')
    c = conn.cursor()
    c.execute("UPDATE time_series_data SET value = ? WHERE date = ?", (value, date))
    conn.commit()
    conn.close()

# Delete the last data entry in the database
def delete_last_data_in_db():
    conn = sqlite3.connect('forecasting.db')
    c = conn.cursor()
    c.execute("DELETE FROM time_series_data WHERE id = (SELECT MAX(id) FROM time_series_data)")
    conn.commit()
    conn.close()

# Retrieve data from the database
def get_data_from_db():
    conn = sqlite3.connect('forecasting.db')
    df = pd.read_sql_query("SELECT date, value FROM time_series_data ORDER BY date", conn)
    conn.close()
    return df

# Forecasting function
def forecast_arima(data, steps):
    model = ARIMA(data, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# Streamlit app
def main():
    st.title("Time-Series Forecasting with ARIMA")

    # Input form for new data
    st.sidebar.header("Input New Data")
    with st.sidebar.form(key="input_form"):
        date = st.text_input("Date (YYYY-MM)")
        value = st.number_input("Value", step=1.0, format="%.2f")
        submitted = st.form_submit_button("Add Data")

        if submitted:
            if date and value:
                add_data_to_db(date, value)
                st.sidebar.success("Data added successfully!")
            else:
                st.sidebar.error("Please provide both date and value.")

    # Retrieve and display the data
    data_df = get_data_from_db()
    if not data_df.empty:
        st.subheader("Data Overview")
        st.write(data_df)

        # Edit data functionality
        st.subheader("Edit Data")
        edit_date = st.selectbox("Select Date to Edit", data_df['date'])
        new_value = st.number_input("New Value", step=1.0, format="%.2f")
        if st.button("Update Data"):
            update_data_in_db(edit_date, new_value)
            st.success(f"Data for {edit_date} updated successfully!")

        # Delete last data functionality
        if st.button("Delete Last Data"):
            delete_last_data_in_db()
            st.success("Last data entry deleted successfully!")

        # Convert date column to datetime
        data_df['date'] = pd.to_datetime(data_df['date'], format='%Y-%m')
        data_df.set_index('date', inplace=True)

        # Plot the actual data
        st.subheader("Actual Data")
        st.line_chart(data_df['value'])

        # Forecasting
        st.subheader("Forecast")
        steps = st.number_input("Number of Steps to Forecast", min_value=1, step=1, value=12)
        forecast_values = forecast_arima(data_df['value'], steps=steps)

        # Create forecast DataFrame
        forecast_index = pd.date_range(start=data_df.index[-1] + pd.DateOffset(months=1), periods=steps, freq='MS')
        forecast_df = pd.DataFrame({'Forecast': forecast_values}, index=forecast_index)

        # Display forecast data as a table
        st.write("### Forecast Data")
        st.write(forecast_df)
        st.line_chart(forecast_df)

        # Plot actual and forecast data
        st.write("### Actual vs Forecast")
        plt.figure(figsize=(10, 6))
        plt.plot(data_df.index, data_df['value'], label='Actual', marker='o')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', marker='x')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Actual vs Forecast')
        st.pyplot(plt)
    else:
        st.warning("No data available. Please add data to start forecasting.")

if __name__ == "__main__":
    main()
