import yfinance as yf
import pandas as pd
import ta


def download_market_data(ticker, start_date, end_date):
    """Descarga datos históricos de Yahoo Finance y aplana las columnas MultiIndex."""
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns] 

    return data


def calculate_indicators(df):
    """Calcula indicadores técnicos básicos usando la librería 'ta'."""    
    df['SMA_50'] = df['Close'] / ta.trend.sma_indicator(df['Close'], window=50) # Normalizar Precio/Sma50
    df['SMA_200'] = df['Close'] / ta.trend.sma_indicator(df['Close'], window=200) # Normalizar Precio/Sma50
    df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)/100
    df['MACD'] = ta.trend.macd(df['Close'])
    df['MACD_signal'] = ta.trend.macd_signal(df['Close'])
    df['MACD_hist'] = ta.trend.macd_diff(df['Close'])
    df.dropna(inplace=True)  # Eliminar filas con valores NaN
    return df

def download_and_prepare_data_by_years(ticker, start_date, end_date):
    """Downloads market data and splits it by years."""
    # Descargar todo el dataset y calcular indicadores
    full_data = download_market_data(ticker, start_date, end_date)
    full_data = calculate_indicators(full_data)
    full_data = full_data.reset_index()
    
    # Validar que Date sea tipo datetime
    if not pd.api.types.is_datetime64_any_dtype(full_data['Date']):
        full_data['Date'] = pd.to_datetime(full_data['Date'])
    
    # Extraer año y sub-dataset
    full_data['Year'] = full_data['Date'].dt.year
    years = full_data['Year'].unique()
    
    # Dividir datos por año
    yearly_data = {}
    for year in years:
        year_data = full_data[full_data['Year'] == year].copy()
        year_data = year_data.reset_index(drop=True)  # Resetear index a 0,1,2,...
        yearly_data[int(year)] = year_data
    
    # Resetear index    
    full_data = full_data.reset_index(drop=True)
        
    return yearly_data, full_data


if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = "2024-01-01"

    market_data = download_market_data(ticker, start_date, end_date)
    print("Columnas después de aplanar:", market_data.columns)  # Debugging

    market_data = calculate_indicators(market_data)

    print(market_data.tail())
    market_data.to_csv(f"{ticker}_market_data.csv", index=False)
    
    