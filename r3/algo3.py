import json
from datamodel import OrderDepth, TradingState, Order, Trade, Symbol, ProsperityEncoder
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
import collections

# Helper function for Normal CDF (used in Black-Scholes)
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

# Black-Scholes Call Option Pricer
def black_scholes_call(S, K, T, sigma, r=0.0):
    """
    Calculates the Black-Scholes price for a European call option.
    S: Current price of the underlying asset
    K: Strike price of the option
    T: Time to expiration in years (or fraction of the relevant period)
    sigma: Volatility of the underlying asset's returns (annualized or matching T's unit)
    r: Risk-free interest rate (annualized or matching T's unit)
    """
    if sigma <= 0 or T <= 0:
        # If no volatility or time left, price is intrinsic value
        return max(0.0, S - K)

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
    return call_price

# Constants
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
ALL_SYMBOLS = [VOLCANIC_ROCK]
VOUCHER_STRIKES = {
    f"{VOUCHER_PREFIX}9500": 9500,
    f"{VOUCHER_PREFIX}9750": 9750,
    f"{VOUCHER_PREFIX}10000": 10000,
    f"{VOUCHER_PREFIX}10250": 10250,
    f"{VOUCHER_PREFIX}10500": 10500,
}
ALL_SYMBOLS.extend(VOUCHER_STRIKES.keys())

POSITION_LIMITS = {
    VOLCANIC_ROCK: 400,
    f"{VOUCHER_PREFIX}9500": 200,
    f"{VOUCHER_PREFIX}9750": 200,
    f"{VOUCHER_PREFIX}10000": 200,
    f"{VOUCHER_PREFIX}10250": 200,
    f"{VOUCHER_PREFIX}10500": 200,
}

# --- Strategy Parameters ---
# Volatility calculation window (number of ticks)
VOLATILITY_WINDOW = 100
# Minimum data points needed to calculate volatility
MIN_VOLATILITY_POINTS = 20
# Option Pricing Parameters
TOTAL_EXPIRATION_DAYS = 7 # Initial days to expiry from Round 1 start
# Risk-free rate (simplified assumption for short duration)
RISK_FREE_RATE = 0.0
# Threshold difference between theoretical price and market price to trigger a trade
THEORETICAL_VS_MARKET_THRESHOLD = 5 # Trade if market price is this much away from theoretical
# Maximum size per order to limit impact
MAX_ORDER_SIZE = 15
# Position buffer percentage (keep this % away from limits)
POSITION_BUFFER_PCT = 0.10


class Trader:

    def __init__(self):
        # Store recent mid-prices of Volcanic Rock for volatility calculation
        self.volcanic_rock_price_history = collections.deque(maxlen=VOLATILITY_WINDOW + 5)
        self.current_day = -1 # Track day changes

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        """Calculates the mid-price from the best bid and ask."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None:
            return (best_bid + best_ask) / 2.0
        elif best_bid is not None:
            return float(best_bid)
        elif best_ask is not None:
            return float(best_ask)
        else:
            return None

    def calculate_historical_volatility(self) -> Optional[float]:
        """
        Calculates the annualized standard deviation of log returns.
        Returns volatility or None if not enough data.
        """
        if len(self.volcanic_rock_price_history) < MIN_VOLATILITY_POINTS:
            return None

        prices = np.array(list(self.volcanic_rock_price_history))
        log_returns = np.log(prices[1:] / prices[:-1])

        # Calculate std dev of log returns.
        # We need to scale it to match the time unit 'T' used in Black-Scholes.
        # If T is fraction of the 7-day period, volatility should reflect that period.
        # std_dev_daily_equivalent = np.std(log_returns) * sqrt(ticks_per_day) # Rough
        # For simplicity, let's use the standard deviation directly as 'sigma'
        # assuming T = days_remaining / TOTAL_EXPIRATION_DAYS represents the correct time fraction.
        # This isn't perfect annualization but keeps units consistent for the BS model here.
        volatility = np.std(log_returns)

        # We need a non-zero volatility for Black-Scholes
        return max(volatility, 1e-6) # Return a small non-zero value if std dev is zero

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        """Extracts the best bid and ask price from the order depth."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def calculate_safe_position_limits(self, symbol: str, current_position: int) -> Tuple[int, int]:
        """
        Calculate safe position limits with a buffer.
        Returns (max_buy_volume, max_sell_volume).
        """
        max_position = POSITION_LIMITS.get(symbol, 0)
        buffer = int(max_position * POSITION_BUFFER_PCT)

        effective_max = max_position - buffer
        effective_min = -max_position + buffer

        max_buy = effective_max - current_position
        max_sell = current_position - effective_min

        return max(0, max_buy), max(0, max_sell) # Ensure non-negative

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {symbol: [] for symbol in ALL_SYMBOLS}
        conversions = 0
        traderData = "" # Used to store state (price history)

        # --- Update State and History ---
        # Determine current day (assuming 1 day = 1,000,000 timestamps)
        # Adjust this based on actual simulation if needed.
        day_from_timestamp = state.timestamp // 1000000
        if day_from_timestamp != self.current_day:
             self.current_day = day_from_timestamp
             print(f"--- Entering Day {self.current_day + 1} (Timestamp: {state.timestamp}) ---")

        # Try to load previous price history
        if state.traderData:
            try:
                loaded_data = json.loads(state.traderData)
                # Load with a maximum size limit to prevent unbounded growth
                self.volcanic_rock_price_history = collections.deque(loaded_data.get("vr_history", []), maxlen=VOLATILITY_WINDOW + 5)
            except json.JSONDecodeError:
                print("Error decoding traderData JSON")


        vr_order_depth = state.order_depths.get(VOLCANIC_ROCK)
        current_vr_price = None
        if vr_order_depth:
            current_vr_price = self.get_mid_price(vr_order_depth)
            if current_vr_price is not None:
                self.volcanic_rock_price_history.append(current_vr_price)
                # print(f"Timestamp {state.timestamp}: Added VR price {current_vr_price:.2f}, History size: {len(self.volcanic_rock_price_history)}")

        # --- Calculate Inputs for Option Pricing ---
        volatility = self.calculate_historical_volatility()
        # Days remaining: Round 1 = 7 days, Round 2 = 6 days, ..., Round 7 = 1 day.
        # Assuming day_from_timestamp starts at 0 for Round 1.
        days_remaining = max(0.001, TOTAL_EXPIRATION_DAYS - self.current_day) # Avoid T=0 before expiry
        # Time to expiration as a fraction of the total period (7 days)
        # Or could use T = days_remaining / 252.0 if sigma is annualized.
        # Let's stick to T relative to the 7 days.
        time_to_expiration = days_remaining / TOTAL_EXPIRATION_DAYS

        # Store updated history for next iteration
        traderData = json.dumps({"vr_history": list(self.volcanic_rock_price_history)})

        # --- Main Logic: Iterate through Vouchers ---
        if current_vr_price is None or volatility is None:
            # Not enough data to price options yet
            print(f"Timestamp {state.timestamp}: Insufficient data (VR Price: {current_vr_price}, Volatility: {volatility})")
            return result, conversions, traderData

        print(f"Timestamp {state.timestamp}: VR Price={current_vr_price:.2f}, Est Volatility={volatility:.4f}, T={time_to_expiration:.3f} ({days_remaining} days left)")

        current_positions = state.position if state.position is not None else {}

        for symbol, strike_price in VOUCHER_STRIKES.items():
            voucher_order_depth = state.order_depths.get(symbol)
            if not voucher_order_depth:
                continue # No market data for this voucher

            voucher_best_bid, voucher_best_ask = self.get_best_bid_ask(voucher_order_depth)
            current_pos = current_positions.get(symbol, 0)
            max_buy_vol, max_sell_vol = self.calculate_safe_position_limits(symbol, current_pos)

            # Calculate theoretical price
            theoretical_price = black_scholes_call(
                S=current_vr_price,
                K=strike_price,
                T=time_to_expiration,
                sigma=volatility,
                r=RISK_FREE_RATE
            )
            intrinsic_value = max(0.0, current_vr_price - strike_price)

            print(f"  {symbol} (K={strike_price}): Market Bid={voucher_best_bid}, Market Ask={voucher_best_ask}, Theoretical={theoretical_price:.2f}, Intrinsic={intrinsic_value:.2f}, Pos={current_pos}")

            # --- Trading Logic ---
            # 1. Buy Opportunity: Market Ask < Theoretical Price - Threshold
            if voucher_best_ask is not None and max_buy_vol > 0:
                if voucher_best_ask < theoretical_price - THEORETICAL_VS_MARKET_THRESHOLD:
                    # Volume is minimum of position limit, max order size, and available ask liquidity
                    ask_volume = abs(voucher_order_depth.sell_orders.get(voucher_best_ask, 0))
                    trade_volume = min(max_buy_vol, MAX_ORDER_SIZE, ask_volume)

                    if trade_volume > 0:
                        print(f"    BUYING {trade_volume} {symbol} @ {voucher_best_ask} (Theoretical: {theoretical_price:.2f})")
                        result[symbol].append(Order(symbol, voucher_best_ask, trade_volume))
                        max_buy_vol -= trade_volume # Update remaining capacity locally

                # Add a simple check for intrinsic value arbitrage (less common but safer)
                elif voucher_best_ask < intrinsic_value - 1: # Buy if ask is clearly below intrinsic
                     ask_volume = abs(voucher_order_depth.sell_orders.get(voucher_best_ask, 0))
                     trade_volume = min(max_buy_vol, MAX_ORDER_SIZE, ask_volume)
                     if trade_volume > 0:
                         print(f"    BUYING (Intrinsic Arb) {trade_volume} {symbol} @ {voucher_best_ask} (Intrinsic: {intrinsic_value:.2f})")
                         result[symbol].append(Order(symbol, voucher_best_ask, trade_volume))
                         max_buy_vol -= trade_volume


            # 2. Sell Opportunity: Market Bid > Theoretical Price + Threshold
            if voucher_best_bid is not None and max_sell_vol > 0:
                if voucher_best_bid > theoretical_price + THEORETICAL_VS_MARKET_THRESHOLD:
                    # Volume is minimum of position limit, max order size, and available bid liquidity
                    bid_volume = abs(voucher_order_depth.buy_orders.get(voucher_best_bid, 0))
                    trade_volume = min(max_sell_vol, MAX_ORDER_SIZE, bid_volume)

                    if trade_volume > 0:
                        print(f"    SELLING {trade_volume} {symbol} @ {voucher_best_bid} (Theoretical: {theoretical_price:.2f})")
                        result[symbol].append(Order(symbol, voucher_best_bid, -trade_volume))
                        max_sell_vol -= trade_volume # Update remaining capacity locally


        # Clean up result: remove products with no orders
        final_result = {symbol: orders for symbol, orders in result.items() if orders}

        return final_result, conversions, traderData