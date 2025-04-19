# Combined Algorithm: algo_combined_safe_arb.py
import json
from datamodel import (
    TradingState, OrderDepth, Order, Trade, Symbol, ProsperityEncoder,
    Listing, ConversionObservation, Observation, Product, Position
)
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
import collections

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------

# --- Product Definitions ---
# (Keep all product definitions: VOLCANIC_ROCK, VOUCHERS, MACARONS, BASKETS, COMPONENTS)
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
CROISSANTS = "CROISSANTS"
JAMS = "JAMS"
DJEMBES = "DJEMBES"
PICNIC_BASKET1 = "PICNIC_BASKET1"
PICNIC_BASKET2 = "PICNIC_BASKET2"

# --- All Symbols Traded ---
VOUCHER_STRIKES = {
    f"{VOUCHER_PREFIX}9500": 9500, f"{VOUCHER_PREFIX}9750": 9750,
    f"{VOUCHER_PREFIX}10000": 10000, f"{VOUCHER_PREFIX}10250": 10250,
    f"{VOUCHER_PREFIX}10500": 10500,
}
OPTIONS_SYMBOLS = [VOLCANIC_ROCK] + list(VOUCHER_STRIKES.keys())
MACARON_SYMBOLS = [MAGNIFICENT_MACARONS]
BASKET_COMPONENTS = {
    PICNIC_BASKET1: {CROISSANTS: 6, JAMS: 3, DJEMBES: 1},
    PICNIC_BASKET2: {CROISSANTS: 4, JAMS: 2},
}
BASKET_SYMBOLS = list(BASKET_COMPONENTS.keys())
COMPONENT_SYMBOLS = [CROISSANTS, JAMS, DJEMBES]
ALL_SYMBOLS = list(set(OPTIONS_SYMBOLS + MACARON_SYMBOLS + BASKET_SYMBOLS + COMPONENT_SYMBOLS))

# --- Position Limits ---
# (Keep POSITION_LIMITS dictionary as before)
POSITION_LIMITS = {
    VOLCANIC_ROCK: 400, f"{VOUCHER_PREFIX}9500": 200, f"{VOUCHER_PREFIX}9750": 200,
    f"{VOUCHER_PREFIX}10000": 200, f"{VOUCHER_PREFIX}10250": 200, f"{VOUCHER_PREFIX}10500": 200,
    MAGNIFICENT_MACARONS: 75, CROISSANTS: 250, JAMS: 350, DJEMBES: 60,
    PICNIC_BASKET1: 60, PICNIC_BASKET2: 100,
}
POSITION_BUFFER_PCT = 0.10 # Keep buffer

# --- Options Strategy Parameters (Safer Arb) ---
VOLATILITY_WINDOW = 100
MIN_VOLATILITY_POINTS = 20
TOTAL_EXPIRATION_DAYS = 7
RISK_FREE_RATE = 0.0
OPTIONS_THEO_VS_MARKET_THRESHOLD = 6 # Slightly increased threshold for safety
OPTIONS_MAX_ORDER_SIZE = 10         # Reduced size

# --- Macaron Strategy Parameters (Safer Conversion Arb ONLY) ---
MM_CONVERSION_LIMIT = 10
# Significantly increased threshold to ensure profit after all fees/spreads
MM_ARB_PROFIT_THRESHOLD = 6.0 # <--- Increased significantly
# CSI_THRESHOLD = 750.0 # Defined but not actively used in this safe version

# --- Basket Strategy Parameters (Safer Arb) ---
BASKET_PROFIT_MARGIN = 6.0 # Increased threshold for safety
BASKET_MAX_ARB_TRADE_SIZE = 8 # Reduced size

# --- Logging ---
class Logger:
    # (Same Logger class as before)
    def __init__(self) -> None:
        self.logs = ""
        self.verbose = True # Set to False to reduce log output size

    def log(self, message: str) -> None:
        if self.verbose:
            self.logs += message + "\n"

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps({"logs": self.logs,}, cls=ProsperityEncoder, indent=2))
        self.logs = ""

logger = Logger()

# ------------------------------------------------------------------------------
# Helper Functions (Keep Math, General Helpers)
# ------------------------------------------------------------------------------
# --- Math Helpers (for Options) ---
def norm_cdf(x):
    # (same as before)
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes_call(S, K, T, sigma, r=0.0):
    # (same as before, including safety checks)
    if sigma <= 0 or T <= 0: return max(0.0, S - K)
    ratio = max(1e-9, S / K)
    d1_num = math.log(ratio) + (r + 0.5 * sigma ** 2) * T
    d1_den = sigma * math.sqrt(T)
    if d1_den == 0 : return max(0.0, S-K)
    d1 = d1_num / d1_den
    d2 = d1 - sigma * math.sqrt(T)
    try:
        call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
    except Exception as e:
        logger.log(f"WARN: Black-Scholes calculation error: {e}. S={S}, K={K}, T={T}, sigma={sigma}")
        return max(0.0, S-K)
    return call_price

# --- General Helpers ---
def get_mid_price(order_depth: OrderDepth) -> Optional[float]:
    # (same as before)
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
    elif best_bid is not None: return float(best_bid)
    elif best_ask is not None: return float(best_ask)
    else: return None

def get_best_bid_ask(order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    # (same as before)
    best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
    best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    return best_bid, best_ask

def calculate_safe_position_limits(symbol: str, current_position: int) -> Tuple[int, int]:
    # (same as before)
    max_pos = POSITION_LIMITS.get(symbol, 0)
    if max_pos == 0: return 0, 0
    buffer = int(max_pos * POSITION_BUFFER_PCT)
    effective_max_limit = max_pos - buffer
    effective_min_limit = -max_pos + buffer
    max_buy = effective_max_limit - current_position
    max_sell = current_position - effective_min_limit
    return max(0, max_buy), max(0, max_sell)

# ------------------------------------------------------------------------------
# Trader Class
# ------------------------------------------------------------------------------
class Trader:
    def __init__(self):
        logger.log("Initializing Combined SAFE ARBITRAGE Trader")
        # State for Options (Volatility)
        self.volcanic_rock_price_history = collections.deque(maxlen=VOLATILITY_WINDOW + 5)
        self.current_day = -1
        self.current_volatility = None
        # NOTE: Macaron MM history is removed as MM is disabled

    def load_state(self, traderData: str):
        # Only load VR history now
        if traderData:
            try:
                loaded_data = json.loads(traderData)
                vr_hist = loaded_data.get("vr_history", [])
                self.volcanic_rock_price_history = collections.deque(vr_hist, maxlen=VOLATILITY_WINDOW + 5)
                logger.log(f"Loaded VR history ({len(self.volcanic_rock_price_history)})")
            except Exception as e:
                logger.log(f"Error loading traderData: {e}. Starting fresh history.")
                self.volcanic_rock_price_history.clear()
        else:
             logger.log("No traderData found, starting fresh history.")

    def save_state(self) -> str:
        # Only save VR history now
        try:
            state_to_save = {"vr_history": list(self.volcanic_rock_price_history)}
            return json.dumps(state_to_save)
        except Exception as e:
            logger.log(f"Error saving state: {e}")
            return ""

    def calculate_historical_volatility(self) -> Optional[float]:
        # (same as before, including safety checks)
        if len(self.volcanic_rock_price_history) < MIN_VOLATILITY_POINTS: return None
        prices = np.array(list(self.volcanic_rock_price_history), dtype=float)
        if np.any(prices <= 0):
            logger.log("Warning: Non-positive prices in VR history.")
            return self.current_volatility
        if len(prices) < 2: return None # Need at least 2 prices for log return
        log_returns = np.log(prices[1:] / prices[:-1])
        volatility = np.std(log_returns)
        if volatility > 0.5:
             logger.log(f"Warning: High volatility calculated ({volatility:.4f}). Using previous: {self.current_volatility}.")
             return self.current_volatility
        self.current_volatility = max(volatility, 1e-6)
        return self.current_volatility

    def run_options_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        """ Executes the Volcanic Rock options ARBITRAGE strategy. """
        logger.log("--- Running Options Strategy (Safe Arb) ---")
        vr_order_depth = state.order_depths.get(VOLCANIC_ROCK)
        current_vr_price = None
        if vr_order_depth:
            current_vr_price = get_mid_price(vr_order_depth)
            if current_vr_price is not None:
                self.volcanic_rock_price_history.append(current_vr_price)

        volatility = self.calculate_historical_volatility()

        day_from_timestamp = state.timestamp // 1000000
        if day_from_timestamp != self.current_day:
            self.current_day = day_from_timestamp
            logger.log(f"--- Entering Day {self.current_day + 1} (Timestamp: {state.timestamp}) ---")
        days_remaining = max(0.001, TOTAL_EXPIRATION_DAYS - self.current_day)
        time_to_expiration = days_remaining / TOTAL_EXPIRATION_DAYS

        if current_vr_price is None or volatility is None:
            logger.log(f"  Insufficient data for Options (VR Price: {current_vr_price}, Vol: {volatility})")
            return

        logger.log(f"  Options Input: VR Price={current_vr_price:.2f}, Vol={volatility:.4f}, T={time_to_expiration:.3f}")

        for symbol, strike_price in VOUCHER_STRIKES.items():
            if symbol not in state.order_depths: continue
            voucher_od = state.order_depths[symbol]
            voucher_bid, voucher_ask = get_best_bid_ask(voucher_od)
            current_pos = current_positions.get(symbol, 0)
            max_buy_vol, max_sell_vol = calculate_safe_position_limits(symbol, current_pos)

            # Skip if no market prices
            if voucher_bid is None and voucher_ask is None: continue

            theoretical_price = black_scholes_call(current_vr_price, strike_price, time_to_expiration, volatility, RISK_FREE_RATE)
            logger.log(f"    {symbol} (K={strike_price}): Mkt Bid={voucher_bid}, Mkt Ask={voucher_ask}, Theo={theoretical_price:.2f}, Pos={current_pos}")

            # Buy Opportunity (Market Ask < Theoretical - Threshold)
            if voucher_ask is not None and max_buy_vol > 0:
                if voucher_ask < theoretical_price - OPTIONS_THEO_VS_MARKET_THRESHOLD:
                    ask_volume = abs(voucher_od.sell_orders.get(voucher_ask, 0))
                    trade_volume = min(max_buy_vol, OPTIONS_MAX_ORDER_SIZE, ask_volume)
                    if trade_volume > 0:
                        logger.log(f"      BUYING Option {trade_volume} @ {voucher_ask} (Theo={theoretical_price:.2f})")
                        result[symbol].append(Order(symbol, voucher_ask, trade_volume))

            # Sell Opportunity (Market Bid > Theoretical + Threshold)
            if voucher_bid is not None and max_sell_vol > 0:
                if voucher_bid > theoretical_price + OPTIONS_THEO_VS_MARKET_THRESHOLD:
                    bid_volume = abs(voucher_od.buy_orders.get(voucher_bid, 0))
                    trade_volume = min(max_sell_vol, OPTIONS_MAX_ORDER_SIZE, bid_volume)
                    if trade_volume > 0:
                        logger.log(f"      SELLING Option {trade_volume} @ {voucher_bid} (Theo={theoretical_price:.2f})")
                        result[symbol].append(Order(symbol, voucher_bid, -trade_volume))

    def calculate_component_value_best_price(self, state: TradingState, basket_symbol: str, use_asks_for_components: bool) -> Optional[float]:
        # (same as before)
        components = BASKET_COMPONENTS.get(basket_symbol, {})
        if not components: return None
        total_value = 0
        for product, quantity in components.items():
            comp_od = state.order_depths.get(product)
            if not comp_od: return None
            comp_bid, comp_ask = get_best_bid_ask(comp_od)
            if use_asks_for_components:
                if comp_ask is None: return None
                total_value += comp_ask * quantity
            else:
                if comp_bid is None: return None
                total_value += comp_bid * quantity
        return total_value

    def run_basket_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        # (same logic as before, using updated safe thresholds/sizes)
        logger.log("--- Running Basket Strategy (Safe Arb) ---")
        for basket_symbol in BASKET_SYMBOLS:
            basket_od = state.order_depths.get(basket_symbol)
            if not basket_od: continue
            components = BASKET_COMPONENTS.get(basket_symbol, {})
            if not components: continue
            basket_pos = current_positions.get(basket_symbol, 0)
            basket_max_buy, basket_max_sell = calculate_safe_position_limits(basket_symbol, basket_pos)
            basket_best_bid, basket_best_ask = get_best_bid_ask(basket_od)

            # Opp 1: Sell Basket, Buy Components
            if basket_best_bid is not None and basket_max_sell > 0:
                comp_buy_cost = self.calculate_component_value_best_price(state, basket_symbol, use_asks_for_components=True)
                if comp_buy_cost is not None and basket_best_bid > comp_buy_cost + BASKET_PROFIT_MARGIN:
                    logger.log(f"  Potential Arb (Sell {basket_symbol}): Basket Bid={basket_best_bid}, Comp Cost={comp_buy_cost:.2f}")
                    vol_limit_basket_bid = abs(basket_od.buy_orders.get(basket_best_bid, 0))
                    vol_limit_comp_ask = []
                    vol_limit_comp_pos = []
                    possible = True
                    for prod, qty in components.items():
                        comp_od = state.order_depths.get(prod)
                        comp_pos = current_positions.get(prod, 0)
                        comp_max_buy, _ = calculate_safe_position_limits(prod, comp_pos)
                        if not comp_od: possible = False; break
                        _, comp_best_ask = get_best_bid_ask(comp_od)
                        if comp_best_ask is None: possible = False; break
                        vol_at_ask = abs(comp_od.sell_orders.get(comp_best_ask, 0))
                        if qty <= 0: continue # Should not happen
                        vol_limit_comp_ask.append(math.floor(vol_at_ask / qty))
                        vol_limit_comp_pos.append(math.floor(comp_max_buy / qty))
                    if not possible: continue
                    max_volume = min(basket_max_sell, BASKET_MAX_ARB_TRADE_SIZE, vol_limit_basket_bid,
                                     min(vol_limit_comp_ask) if vol_limit_comp_ask else 0,
                                     min(vol_limit_comp_pos) if vol_limit_comp_pos else 0)
                    if max_volume > 0:
                        logger.log(f"    EXECUTING Sell Basket {max_volume} @ {basket_best_bid}")
                        result[basket_symbol].append(Order(basket_symbol, basket_best_bid, -max_volume))
                        for prod, qty in components.items():
                             comp_od = state.order_depths.get(prod); _, comp_best_ask = get_best_bid_ask(comp_od)
                             if prod not in result: result[prod] = []
                             if comp_best_ask is not None: # Ensure price exists
                                 result[prod].append(Order(prod, comp_best_ask, max_volume * qty))

            # Opp 2: Buy Basket, Sell Components
            if basket_best_ask is not None and basket_max_buy > 0:
                 comp_sell_rev = self.calculate_component_value_best_price(state, basket_symbol, use_asks_for_components=False)
                 if comp_sell_rev is not None and comp_sell_rev > basket_best_ask + BASKET_PROFIT_MARGIN:
                    logger.log(f"  Potential Arb (Buy {basket_symbol}): Basket Ask={basket_best_ask}, Comp Rev={comp_sell_rev:.2f}")
                    vol_limit_basket_ask = abs(basket_od.sell_orders.get(basket_best_ask, 0))
                    vol_limit_comp_bid = []
                    vol_limit_comp_pos = []
                    possible = True
                    for prod, qty in components.items():
                        comp_od = state.order_depths.get(prod)
                        comp_pos = current_positions.get(prod, 0)
                        _, comp_max_sell = calculate_safe_position_limits(prod, comp_pos)
                        if not comp_od: possible = False; break
                        comp_best_bid, _ = get_best_bid_ask(comp_od)
                        if comp_best_bid is None: possible = False; break
                        vol_at_bid = abs(comp_od.buy_orders.get(comp_best_bid, 0))
                        if qty <= 0: continue # Should not happen
                        vol_limit_comp_bid.append(math.floor(vol_at_bid / qty))
                        vol_limit_comp_pos.append(math.floor(comp_max_sell / qty))
                    if not possible: continue
                    max_volume = min(basket_max_buy, BASKET_MAX_ARB_TRADE_SIZE, vol_limit_basket_ask,
                                     min(vol_limit_comp_bid) if vol_limit_comp_bid else 0,
                                     min(vol_limit_comp_pos) if vol_limit_comp_pos else 0)
                    if max_volume > 0:
                        logger.log(f"    EXECUTING Buy Basket {max_volume} @ {basket_best_ask}")
                        result[basket_symbol].append(Order(basket_symbol, basket_best_ask, max_volume))
                        for prod, qty in components.items():
                             comp_od = state.order_depths.get(prod); comp_best_bid, _ = get_best_bid_ask(comp_od)
                             if prod not in result: result[prod] = []
                             if comp_best_bid is not None: # Ensure price exists
                                 result[prod].append(Order(prod, comp_best_bid, -max_volume * qty))

    def run_macaron_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]) -> int:
        """ Executes the Macaron CONVERSION ARBITRAGE ONLY strategy. """
        logger.log("--- Running Macaron Strategy (Safe Conv Arb) ---")
        conversions = 0
        symbol = MAGNIFICENT_MACARONS
        current_pos = current_positions.get(symbol, 0)
        max_buy, max_sell = calculate_safe_position_limits(symbol, current_pos)

        # Access Observations
        if not state.observations or not state.observations.conversionObservations:
            logger.log("  WARN: No conversion observations.")
            return 0
        conv_obs = state.observations.conversionObservations.get(symbol)
        if not conv_obs:
            logger.log(f"  WARN: No conversion observation for {symbol}")
            return 0

        transport_fees = conv_obs.transportFees
        import_tariff = conv_obs.importTariff
        export_tariff = conv_obs.exportTariff
        pc_bid = conv_obs.bidPrice
        pc_ask = conv_obs.askPrice
        # sunlight_index = conv_obs.sunlightIndex # Available but not used in safe arb

        # Calculate Effective PC Prices
        effective_pc_buy_price = pc_ask + transport_fees + import_tariff
        effective_pc_sell_price = pc_bid - transport_fees - export_tariff

        # Access Exchange Order Book
        order_depth = state.order_depths.get(symbol)
        if not order_depth or (not order_depth.buy_orders and not order_depth.sell_orders):
            logger.log(f"  WARN: No order depth for {symbol}")
            return 0

        best_bid, best_ask = get_best_bid_ask(order_depth)
        logger.log(f"  MM Market: Bid={best_bid}, Ask={best_ask}")
        logger.log(f"  MM PC Prices: Eff Buy={effective_pc_buy_price:.2f}, Eff Sell={effective_pc_sell_price:.2f}")

        # --- Conversion Arbitrage (Higher Threshold) ---
        # Opp 1: Buy PC -> Sell Exch
        if best_bid is not None:
            profit = best_bid - effective_pc_buy_price
            required_profit = MM_ARB_PROFIT_THRESHOLD # Use the new higher threshold
            if profit > required_profit:
                vol_avail = abs(order_depth.buy_orders.get(best_bid, 0))
                # Limit volume by CONVERSION limit, position limit, and liquidity
                vol = min(MM_CONVERSION_LIMIT, max_buy, vol_avail)
                if vol > 0:
                    logger.log(f"    ACTION: MM Arb 1 Executing - Conv BUY: {vol}, Sell Exch @ {best_bid} (Profit={profit:.2f})")
                    conversions = vol
                    if symbol not in result: result[symbol] = []
                    result[symbol].append(Order(symbol, best_bid, -vol))
                    return conversions # Exit after action

        # Opp 2: Buy Exch -> Sell PC
        if best_ask is not None and conversions == 0: # Only if Arb 1 didn't run
            profit = effective_pc_sell_price - best_ask
            required_profit = MM_ARB_PROFIT_THRESHOLD # Use the new higher threshold
            # Condition: Must have positive position to sell to PC
            if profit > required_profit and current_pos > 0:
                vol_avail = abs(order_depth.sell_orders.get(best_ask, 0))
                # Limit volume by CONVERSION limit, current pos, position limit, and liquidity
                vol = min(MM_CONVERSION_LIMIT, current_pos, max_buy, vol_avail) # Note: max_sell implicitly covered by current_pos check here
                if vol > 0:
                    logger.log(f"    ACTION: MM Arb 2 Executing - Conv SELL: {vol}, Buy Exch @ {best_ask} (Profit={profit:.2f})")
                    conversions = -vol
                    if symbol not in result: result[symbol] = []
                    result[symbol].append(Order(symbol, best_ask, vol))
                    return conversions # Exit after action

        # --- Market Making Disabled ---
        logger.log("  Macaron Market Making is DISABLED in safe mode.")

        return conversions # Return conversions count (likely 0 if no arb)

    # --------------------------------------------------------------------------
    # Main Execution Method
    # --------------------------------------------------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        logger.log(f"\n===== Timestamp: {state.timestamp} (Safe Arb Mode) =====")
        self.load_state(state.traderData)
        current_positions = state.position if state.position is not None else {}
        logger.log(f"  Current Positions: {current_positions}")
        result: Dict[str, List[Order]] = {symbol: [] for symbol in ALL_SYMBOLS}
        total_conversions = 0

        # Execute Strategies by Group
        try: self.run_options_strategy(state, current_positions, result)
        except Exception as e: logger.log(f"!!! ERROR in Options Strategy: {e}"); import traceback; traceback.print_exc()
        try: self.run_basket_strategy(state, current_positions, result)
        except Exception as e: logger.log(f"!!! ERROR in Basket Strategy: {e}"); import traceback; traceback.print_exc()
        try:
             macaron_conversions = self.run_macaron_strategy(state, current_positions, result)
             total_conversions += macaron_conversions
        except Exception as e: logger.log(f"!!! ERROR in Macaron Strategy: {e}"); import traceback; traceback.print_exc()

        # Clean up and Save State
        final_result = {symbol: orders for symbol, orders in result.items() if orders}
        logger.log(f"  Final Orders Sent: {final_result}")
        logger.log(f"  Total Conversions Sent: {total_conversions}")
        traderData = self.save_state()
        logger.log(f"===== End Timestamp: {state.timestamp} =====")
        logger.flush(state, final_result, total_conversions, traderData)
        return final_result, total_conversions, traderData