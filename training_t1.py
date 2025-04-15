import pandas as pd
import numpy as np
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
    
    # After plotting, calculate and print performance metrics
    metrics = test_env.calculate_performance_metrics()
    
    return total_reward, portfolio_values, metrics

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
    
    eval_metrics = evaluate_model(model, env, num_episodes=15)

    # Resultados de la evaluación
    print("\nModel Evaluation Results:")
    print(f"Mean Reward: {eval_metrics['mean_reward']:.4f} ± {eval_metrics['std_reward']:.4f}")
    print(f"Mean Return: {eval_metrics['mean_return']*100:.2f}% ± {eval_metrics['std_return']*100:.2f}%")
    print(f"Action Distribution: Hold={eval_metrics['action_distribution']['Hold']*100:.1f}%, " + 
        f"Buy={eval_metrics['action_distribution']['Buy']*100:.1f}%, " +
      f"Sell={eval_metrics['action_distribution']['Sell']*100:.1f}%")


    # Simular rendimiento
    total_reward, portfolio_values, metrics = plot_performance(model, env, full_data)

    # Metricas de rendimiento
    print("\nFull Dataset Performance:")
    print(f"Total Reward: {total_reward:.4f}")
    print(f"Sharpe Ratio: {metrics['Sharpe_Ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['Sortino_Ratio']:.4f}")
    print(f"Calmar Ratio: {metrics['Calmar_Ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['Max_Drawdown']*100:.2f}%")
    print(f"Win/Loss Ratio: {metrics['Win_Loss_Ratio']:.2f}")
    print(f"Annualized Return: {metrics['Annualized_Return']*100:.2f}%")
    print(f"Total Profit: ${metrics['Total_Profit']:.2f}")
    print(f"Final Portfolio Value: ${metrics['Final_Portfolio_Value']:.2f}")

    # Guardar métricas en archivo CSV
    metrics_df = pd.DataFrame({
        'Mean_Reward': [eval_metrics['mean_reward']],
        'Reward_Std': [eval_metrics['std_reward']],
        'Mean_Return': [eval_metrics['mean_return']],
        'Return_Std': [eval_metrics['std_return']],
        'Total_Reward': [total_reward],
        'Sharpe_Ratio': [metrics['Sharpe_Ratio']],
        'Sortino_Ratio': [metrics['Sortino_Ratio']],
        'Calmar_Ratio': [metrics['Calmar_Ratio']],
        'Max_Drawdown': [metrics['Max_Drawdown']],
        'Win_Loss_Ratio': [metrics['Win_Loss_Ratio']],
        'Annualized_Return': [metrics['Annualized_Return']],
        'Total_Profit': [metrics['Total_Profit']],
        'Final_Portfolio_Value': [metrics['Final_Portfolio_Value']],
        'Hold_Pct': [eval_metrics['action_distribution']['Hold']],
        'Buy_Pct': [eval_metrics['action_distribution']['Buy']],
        'Sell_Pct': [eval_metrics['action_distribution']['Sell']]
    })
    metrics_df.to_csv('performance_metrics.csv', index=False)

if __name__ == "__main__":
    main()