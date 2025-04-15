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
        """Calculate various trading performance metrics."""
 
        portfolio_values = np.array(self.port_val_history)        
        daily_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Calcular metricas
        total_days = len(daily_returns)
        trading_days_per_year = 252  # Standard assumption for trading days in a year
        
        # Rendimiento anualizado
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        holding_period_years = total_days / trading_days_per_year
        annualized_return = (1 + total_return) ** (1 / holding_period_years) - 1 if holding_period_years > 0 else 0
        
        # Radio de sharp (anualizado)
        risk_free_rate = 0.02 / trading_days_per_year  # Assuming 2% annual risk-free rate
        excess_returns = daily_returns - risk_free_rate
        sharpe_ratio = (np.mean(excess_returns) * trading_days_per_year) / (np.std(daily_returns) * np.sqrt(trading_days_per_year)) if np.std(daily_returns) > 0 else 0
        
        # Sortino
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino_ratio = (np.mean(excess_returns) * trading_days_per_year) / (downside_std * np.sqrt(trading_days_per_year)) if downside_std > 0 else 0
        
        # Max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
        
        # Radio de Calmar (rendimiento anualizado / max drawdown)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Relación de ganancia/perdida para trades
        win_count = 0
        loss_count = 0
        
        win_count = sum(1 for r in daily_returns if r > 0)
        loss_count = sum(1 for r in daily_returns if r <= 0)
        
        win_loss_ratio = win_count / loss_count if loss_count > 0 else float('inf')
        
        # Ganancia total
        total_profit = portfolio_values[-1] - portfolio_values[0]
        
        # Valor final del portafolio
        final_portfolio_value = portfolio_values[-1]
        
        # Metrics dictionary
        return {
            'Sharpe_Ratio': sharpe_ratio,
            'Sortino_Ratio': sortino_ratio,
            'Calmar_Ratio': calmar_ratio,
            'Max_Drawdown': max_drawdown,
            'Win_Loss_Ratio': win_loss_ratio,
            'Total_Profit': total_profit,
            'Final_Portfolio_Value': final_portfolio_value,
            'Annualized_Return': annualized_return
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

def evaluate_model(model, env, num_episodes=10, seed=None):
    """
    Custom evaluation function for trading models that calculates more reliable statistics.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Lista para guardar resultados
    episode_rewards = []
    episode_returns = []
    episode_lengths = []
    action_counts = {0: 0, 1: 0, 2: 0}  # Hold, Buy, Sell
    
    # Correr episodios de evaluación
    for i in range(num_episodes):
        # Resetear ambiente con una seed diferente cada vez
        episode_seed = None if seed is None else seed + i
        obs, _ = env.reset(seed=episode_seed)
        
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        initial_value = env.port_val_history[0]
        
        # Correr episodio de evaluación
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            
            action_int = action.item() if hasattr(action, 'item') else int(action)
            action_counts[action_int] += 1
            
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            step_count += 1
        
        # Valor final del portafolio y rendimientos
        final_value = env.port_val_history[-1]
        episode_return = (final_value / initial_value) - 1
        
        episode_rewards.append(total_reward)
        episode_returns.append(episode_return)
        episode_lengths.append(step_count)
        
        print(f"Episode {i+1}/{num_episodes}: Reward={total_reward:.2f}, Return={episode_return*100:.2f}%, Length={step_count}")
    
    # Metricas
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    
    total_actions = sum(action_counts.values())
    action_distribution = {
        'Hold': action_counts[0] / total_actions if total_actions > 0 else 0,
        'Buy': action_counts[1] / total_actions if total_actions > 0 else 0,
        'Sell': action_counts[2] / total_actions if total_actions > 0 else 0
    }
    
    # Diccionario con metricas
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'mean_return': mean_return,
        'std_return': std_return,
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'action_distribution': action_distribution,
        'episode_lengths': np.mean(episode_lengths)
    }