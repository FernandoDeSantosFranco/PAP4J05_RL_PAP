import pandas as pd
import numpy as np
from environment import TradingEnvironment
from deepQLearning import TradingGymEnv, train_deep_q_learning, evaluate_model
import matplotlib.pyplot as plt
from data_base import download_and_prepare_data_by_years

def load_market_data_by_years(ticker="AAPL", start_date="2010-01-01", end_date="2024-01-01"):
    
    # Descargar y preparar los datos del mercado, divididos por año
    yearly_data, full_data = download_and_prepare_data_by_years(ticker, start_date, end_date)
    
    return yearly_data, full_data

def plot_performance(model, env, full_data):
    # Crear un ambiente de prueba separado usando el dataset completo
    test_env = TradingGymEnv({0: full_data}, initial_balance=env.initial_balance, trading_fee=env.trading_fee)

    # Simular episodio con el modelo entrenado
    obs, _ = test_env.reset()
    done = False
    truncated = False
    total_reward = 0
    portfolio_values = [test_env.initial_balance]

    while not (done or truncated):
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = test_env.step(action)
        total_reward += reward
        portfolio_values.append(test_env.balance + (test_env.shares_held * test_env.data.iloc[test_env.current_step]['Close']))

    # Graficar valor del portafolio
    plt.figure(figsize=(10, 6))
    plt.plot(portfolio_values)
    plt.title('Valor del Portafolio Durante Simulación')
    plt.xlabel('Pasos')
    plt.ylabel('Valor del Portafolio')
    plt.tight_layout()
    plt.savefig('portfolio_performance.png')
    plt.close()

    return total_reward, portfolio_values

def main():
    # Cargar datos de mercado
    yearly_data, full_data = load_market_data_by_years()

    # Crear entorno de trading
    env = TradingGymEnv(yearly_data)

    # Entrenar modelo Deep Q-Learning
    model = train_deep_q_learning(env)

    # Guardar modelo
    model.save("trading_dqn_model")

    # Evaluar modelo
    mean_reward, std_reward = evaluate_model(model, env)

    # Simular rendimiento
    total_reward, portfolio_values = plot_performance(model, env, full_data)

    # Imprimir resultados
    print("\nResumen de Rendimiento:")
    print(f"Recompensa Media: {mean_reward}")
    print(f"Desviación Estándar de Recompensa: {std_reward}")
    print(f"Recompensa Total: {total_reward}")

    # Guardar métricas en archivo CSV
    metrics_df = pd.DataFrame({
        'Mean_Reward': [mean_reward],
        'Reward_Std': [std_reward],
        'Total_Reward': [total_reward]
    })
    metrics_df.to_csv('performance_metrics.csv', index=False)

if __name__ == "__main__":
    main()