import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

class TradingGymEnv(gym.Env):
    def __init__(self, yearly_data, initial_balance=1000000, trading_fee=0.001):
        super().__init__()
        
        # Configuración del entorno
        self.yearly_data = yearly_data
        self.years = list(yearly_data.keys())
        self.current_year = None
        
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        
        # Definir espacios de acción y observación
        self.action_space = gym.spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(6,), 
            dtype=np.float32
        )
        
        # Estado inicial
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Seleccionar año para el dataset
        self.current_year = np.random.choice(self.years)
        self.data = self.yearly_data[self.current_year]
        
        self.current_step = 0
        self.max_steps = len(self.data) - 1
        self.balance = self.initial_balance
        self.shares_held = 0
        self.operations = []
        self.total_profit = 0
        self.trade_history = []
        self.port_val_history = [self.initial_balance]
        
        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([
            self.data.iloc[self.current_step]['SMA_50'],
            self.data.iloc[self.current_step]['SMA_200'],
            self.data.iloc[self.current_step]['RSI_14'],
            self.data.iloc[self.current_step]['MACD'],
            self.shares_held,
            self.balance
        ], dtype=np.float32)

    def step(self, action):
        # Incrementar el paso actual
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0

        if action == 1:  # Comprar
            if self.balance >= current_price:
                self.operations.append(current_price)
                fee = current_price * self.trading_fee
                self.shares_held += 1
                self.balance -= (current_price + fee)
                reward = 0.01

        elif action == 2:  # Vender
            if self.shares_held > 0:
                fee = current_price * self.trading_fee
                self.shares_held -= 1
                self.balance += (current_price - fee)
                bought_at = self.operations.pop(0) 
                operation_return = (current_price/bought_at)-1
                if operation_return < -0.10:
                    reward = 5 * operation_return
                elif operation_return >= -0.10 and operation_return < -0.05:
                    reward = -0.5
                elif operation_return >= -0.05 and operation_return < 0:
                    reward = -0.1
                elif operation_return >= 0 and operation_return < 0.01:
                    reward = 0.01
                elif operation_return >= 0.01 and operation_return < 0.05:
                    reward = 0.1
                elif operation_return >= 0.05 and operation_return < 0.10:
                    reward = 0.5
                elif operation_return >= 0.10:
                    reward = 5 * operation_return

        elif action == 0:  # Hold
            reward = -0.0001

        # Calcular valor del portafolio
        current_portfolio_value = self.balance + (self.shares_held * current_price)
        self.port_val_history.append(current_portfolio_value)
        
        # Calcular recompensa basada en cambios de valor de portafolio
        reward += (current_portfolio_value - self.port_val_history[-2]) / self.port_val_history[-2]

        return self._get_observation(), reward, done, truncated, {}
    
    def calculate_performance_metrics(self):
        """Calcula métricas de rendimiento."""
        # Convertir retornos a serie de numpy
        returns = np.array(self.returns_history)

        # Sharpe Ratio
        portfolio_returns = np.diff(self.port_val_history) / self.port_val_history[:-1]
        sharpe_ratio = (np.mean(portfolio_returns) - self.risk_free_rate) / np.std(portfolio_returns)

        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.mean(returns) - self.risk_free_rate) / downside_std if downside_std != 0 else 0

        # Calmar Ratio
        max_drawdown = min(self.drawdown_history)
        calmar_ratio = np.mean(returns) / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win/Loss Ratio
        wins = sum(1 for r in self.returns_history if r > 0)
        losses = sum(1 for r in self.returns_history if r <= 0)
        win_loss_ratio = wins / (losses + 1)  # Evitar división por cero

        return {
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Loss_Ratio': win_loss_ratio,
            'Total_Profit': self.total_profit,
            'Final_Portfolio_Value': self.port_val_history[-1]
        }
    
class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards_history = []

    def _on_step(self) -> bool:
        # Puedes agregar lógica de logging o tracking aquí
        return True

def train_deep_q_learning(env, total_timesteps=100000):
    # Envolver el entorno en un DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Crear modelo DQN
    model = DQN(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=500,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )
    
    # Entrenar el modelo
    callback = TrainingCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    return model

def evaluate_model(model, env, num_episodes=100):
    # Evaluar el rendimiento del modelo
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return mean_reward, std_reward