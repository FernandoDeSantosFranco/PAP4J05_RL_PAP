import pandas as pd
import numpy as np
import scipy.stats as stats

class TradingEnvironment:
    def __init__(self, data, initial_balance=1000000, trading_fee=0.001, max_steps=None, risk_free_rate=0.02):
        self.data = data.reset_index()
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.max_steps = max_steps if max_steps else len(data) - 1
        self.risk_free_rate = risk_free_rate  # Tasa libre de riesgo para métricas de rendimiento

        # Reiniciar el estado del entorno
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.total_profit = 0
        self.done = False
        self.trade_history = []
        self.positions = []
        self.port_val_history = [self.initial_balance]
        self.returns_history = []
        self.drawdown_history = []
        self.high_water_mark = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        return np.array([
            self.data.loc[self.current_step, 'Close'],
            self.data.loc[self.current_step, 'SMA_50'],
            self.data.loc[self.current_step, 'SMA_200'],
            self.data.loc[self.current_step, 'RSI_14'],
            self.data.loc[self.current_step, 'MACD'],
            self.shares_held,
            self.balance
        ], dtype=np.float32)

    def step(self, action):
        if self.done:
            raise Exception("El episodio ha terminado, reinicia el entorno.")

        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        current_price = self.data.loc[self.current_step, 'Close']
        reward = 0
        fee = 0

        if action == 1:  # Comprar
            if self.balance >= current_price:
                fee = current_price * self.trading_fee
                self.shares_held += 1
                self.balance -= (current_price + fee)
                self.trade_history.append((self.current_step, 'BUY', current_price, fee))
                self.positions.append(current_price)
                reward = 0.01

        elif action == 2:  # Vender
            if self.shares_held > 0:
                fee = current_price * self.trading_fee
                self.shares_held -= 1
                self.balance += (current_price - fee)
                self.trade_history.append((self.current_step, 'SELL', current_price, fee))
                original_price = self.positions.pop(0)
                reward = (current_price - original_price) / original_price
                self.returns_history.append(reward)

        # Actualizar historial de valor de portafolio
        current_portfolio_value = self.balance + (self.shares_held * current_price)
        self.port_val_history.append(current_portfolio_value)

        # Calcular drawdown
        if current_portfolio_value > self.high_water_mark:
            self.high_water_mark = current_portfolio_value
        drawdown = (current_portfolio_value - self.high_water_mark) / self.high_water_mark
        self.drawdown_history.append(drawdown)

        self.total_profit = current_portfolio_value - self.initial_balance
        return self._get_observation(), reward, self.done

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

    def render(self):
        print(
            f'Step: {self.current_step}, Balance: {self.balance:.2f}, Shares Held: {self.shares_held}, Total Profit: {self.total_profit:.2f}')

    def get_trade_history(self):
        return pd.DataFrame(self.trade_history, columns=['Step', 'Action', 'Price', 'Fee'])