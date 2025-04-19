# Combined Algorithm: algo_combined_safer_mm.py
import json
from datamodel import (
    TradingState, OrderDepth, Order, Trade, Symbol, ProsperityEncoder,
    Listing, ConversionObservation, Observation, Product, Position
)
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
import collections
import statistics # For MAs in original products

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------

# --- Product Definitions ---
VOLCANIC_ROCK = "VOLCANIC_ROCK"
VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"
SQUID_INK = "SQUID_INK"
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
ORIGINAL_SYMBOLS = [RAINFOREST_RESIN, KELP, SQUID_INK]
BASKET_COMPONENTS = {
    PICNIC_BASKET1: {CROISSANTS: 6, JAMS: 3, DJEMBES: 1},
    PICNIC_BASKET2: {CROISSANTS: 4, JAMS: 2},
}
BASKET_SYMBOLS = list(BASKET_COMPONENTS.keys())
COMPONENT_SYMBOLS = [CROISSANTS, JAMS, DJEMBES]
ALL_SYMBOLS = list(set(OPTIONS_SYMBOLS + MACARON_SYMBOLS + ORIGINAL_SYMBOLS + BASKET_SYMBOLS + COMPONENT_SYMBOLS))

# --- Position Limits ---
POSITION_LIMITS = {
    VOLCANIC_ROCK: 400, f"{VOUCHER_PREFIX}9500": 200, f"{VOUCHER_PREFIX}9750": 200,
    f"{VOUCHER_PREFIX}10000": 200, f"{VOUCHER_PREFIX}10250": 200, f"{VOUCHER_PREFIX}10500": 200,
    MAGNIFICENT_MACARONS: 75, RAINFOREST_RESIN: 50, KELP: 50, SQUID_INK: 50,
    CROISSANTS: 250, JAMS: 350, DJEMBES: 60, PICNIC_BASKET1: 60, PICNIC_BASKET2: 100,
}
POSITION_BUFFER_PCT = 0.15 # Keep 15% buffer

# --- Options Strategy Parameters ---
VOLATILITY_WINDOW = 100
MIN_VOLATILITY_POINTS = 20
TOTAL_EXPIRATION_DAYS = 7
RISK_FREE_RATE = 0.0
OPTIONS_ARB_BASE_THRESHOLD = 6.0     # Base threshold for arb
OPTIONS_MONEYNESS_THRESHOLD_FACTOR = 3.0 # How much threshold increases with moneyness
OPTIONS_MONEYNESS_LIMIT = 1.5        # Abs(m_t) value beyond which threshold multiplier maxes out
OPTIONS_MAX_ARB_SIZE = 10
OPTIONS_MM_ENABLE = True
OPTIONS_MM_SPREAD = 8.0              # Wide spread around theoretical
OPTIONS_MM_SIZE = 3                  # Small size
OPTIONS_MM_SKEW_FACTOR = 0.05        # Inventory skew intensity

# --- Macaron Strategy Parameters ---
MM_CONVERSION_LIMIT = 10
MM_ARB_PROFIT_THRESHOLD = 6.0 # High threshold, no MM

# --- Basket Strategy Parameters ---
BASKET_PROFIT_MARGIN = 6.0
BASKET_MAX_ARB_TRADE_SIZE = 8

# --- Original Products Strategy Parameters (Algo1 based) ---
# Shared
ORIGINAL_HISTORY_MAX_LENGTH = 100
# Resin MM
RESIN_MM_BASE_SPREAD = 4            # Wider base spread
RESIN_MM_SKEW_INTENSITY = 0.5       # Less intense skew
RESIN_MM_MAX_ORDER_SIZE = 5         # Smaller size
# Kelp MA
KELP_MA_WINDOW = 20
KELP_THRESHOLD_PCT = 0.003          # Slightly wider % threshold
KELP_THRESHOLD_ABS = 4              # Slightly wider abs threshold
KELP_ORDER_SIZE = 4                 # Smaller size
# Squid Ink Mean Reversion
SQUID_SHORT_WINDOW = 5
SQUID_LONG_WINDOW = 40
SQUID_VOLATILITY_THRESHOLD = 4      # Higher threshold for significance
SQUID_ORDER_SIZE = 3                # Smaller size

# --- Logging ---
class Logger:
    # (Same Logger class as before)
    def __init__(self) -> None: self.logs = ""; self.verbose = True
    def log(self, message: str) -> None:
        if self.verbose: self.logs += message + "\n"
    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps({"logs": self.logs,}, cls=ProsperityEncoder, indent=2)); self.logs = ""

logger = Logger()

# ------------------------------------------------------------------------------
# Helper Functions (Keep Math, General Helpers)
# ------------------------------------------------------------------------------
# (norm_cdf, black_scholes_call, get_mid_price, get_best_bid_ask, calculate_safe_position_limits - remain the same)
# --- Math Helpers (for Options) ---
def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
def black_scholes_call(S, K, T, sigma, r=0.0):
    if sigma <= 0 or T <= 0: return max(0.0, S - K)
    ratio = max(1e-9, S / K); d1_num = math.log(ratio) + (r + 0.5 * sigma ** 2) * T
    d1_den = sigma * math.sqrt(T);
    if d1_den == 0 : return max(0.0, S-K)
    d1 = d1_num / d1_den; d2 = d1 - sigma * math.sqrt(T)
    try: call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
    except Exception as e: logger.log(f"WARN: BS error: {e}"); return max(0.0, S-K)
    return call_price
# --- General Helpers ---
def get_mid_price(od: OrderDepth) -> Optional[float]:
    bid = max(od.buy_orders.keys()) if od.buy_orders else None; ask = min(od.sell_orders.keys()) if od.sell_orders else None
    if bid and ask: return (bid + ask) / 2.0
    elif bid: return float(bid)
    elif ask: return float(ask)
    else: return None
def get_best_bid_ask(od: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    bid = max(od.buy_orders.keys()) if od.buy_orders else None; ask = min(od.sell_orders.keys()) if od.sell_orders else None
    return bid, ask
def calculate_safe_position_limits(sym: str, pos: int) -> Tuple[int, int]:
    max_p = POSITION_LIMITS.get(sym, 0);
    if max_p == 0: return 0, 0
    buf = int(max_p * POSITION_BUFFER_PCT); eff_max = max_p - buf; eff_min = -max_p + buf
    max_b = eff_max - pos; max_s = pos - eff_min; return max(0, max_b), max(0, max_s)

# ------------------------------------------------------------------------------
# Trader Class
# ------------------------------------------------------------------------------
class Trader:
    def __init__(self):
        logger.log("Initializing Combined Trader (Safer MM Version)")
        # State Histories
        self.volcanic_rock_price_history = collections.deque(maxlen=VOLATILITY_WINDOW + 5)
        self.original_product_history = {
            prod: collections.deque(maxlen=ORIGINAL_HISTORY_MAX_LENGTH + 5)
            for prod in ORIGINAL_SYMBOLS
        }
        # Other State
        self.current_day = -1
        self.current_volatility = None

    def load_state(self, traderData: str):
        if traderData:
            try:
                loaded_data = json.loads(traderData)
                vr_hist = loaded_data.get("vr_history", [])
                self.volcanic_rock_price_history = collections.deque(vr_hist, maxlen=VOLATILITY_WINDOW + 5)
                orig_hist = loaded_data.get("orig_history", {})
                for prod, hist_list in orig_hist.items():
                    if prod in self.original_product_history:
                        self.original_product_history[prod] = collections.deque(hist_list, maxlen=ORIGINAL_HISTORY_MAX_LENGTH + 5)
                logger.log(f"Loaded VR history ({len(self.volcanic_rock_price_history)}), Original Product Histories")
            except Exception as e:
                logger.log(f"Error loading traderData: {e}. Starting fresh histories.")
                self.volcanic_rock_price_history.clear()
                for prod in self.original_product_history: self.original_product_history[prod].clear()
        else:
             logger.log("No traderData found, starting fresh histories.")

    def save_state(self) -> str:
        try:
            state_to_save = {
                "vr_history": list(self.volcanic_rock_price_history),
                "orig_history": {prod: list(hist) for prod, hist in self.original_product_history.items()},
            }
            return json.dumps(state_to_save)
        except Exception as e:
            logger.log(f"Error saving state: {e}")
            return ""

    def calculate_historical_volatility(self) -> Optional[float]:
        # (same as before)
        if len(self.volcanic_rock_price_history) < MIN_VOLATILITY_POINTS: return None
        prices = np.array(list(self.volcanic_rock_price_history), dtype=float);
        if np.any(prices <= 0): return self.current_volatility
        if len(prices) < 2: return None
        log_returns = np.log(prices[1:] / prices[:-1]); volatility = np.std(log_returns)
        if volatility > 0.5: return self.current_volatility
        self.current_volatility = max(volatility, 1e-6); return self.current_volatility

    # --- Strategy Execution Methods ---

    def run_options_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        """ Executes Options Arb + Hesitant MM """
        logger.log("--- Running Options Strategy (Arb + Hesitant MM) ---")
        # Update VR History & Volatility
        vr_order_depth = state.order_depths.get(VOLCANIC_ROCK)
        current_vr_price = None
        if vr_order_depth:
            current_vr_price = get_mid_price(vr_order_depth)
            if current_vr_price is not None: self.volcanic_rock_price_history.append(current_vr_price)
        volatility = self.calculate_historical_volatility()

        # Calculate Time to Expiry
        day_from_timestamp = state.timestamp // 1000000
        if day_from_timestamp != self.current_day: self.current_day = day_from_timestamp; logger.log(f"--- Day {self.current_day + 1} ---")
        days_remaining = max(0.001, TOTAL_EXPIRATION_DAYS - self.current_day)
        time_to_expiration = days_remaining / TOTAL_EXPIRATION_DAYS
        sqrt_TTE = math.sqrt(time_to_expiration) if time_to_expiration > 0 else 0

        if current_vr_price is None or volatility is None or sqrt_TTE == 0:
            logger.log(f"  Insufficient data for Options (VR Price={current_vr_price}, Vol={volatility}, sqrtTTE={sqrt_TTE})")
            return

        logger.log(f"  Options Input: S={current_vr_price:.2f}, sigma={volatility:.4f}, T={time_to_expiration:.3f}")

        for symbol, strike_price in VOUCHER_STRIKES.items():
            arb_order_placed = False # Flag to prevent MM if arb happens
            if symbol not in state.order_depths: continue
            voucher_od = state.order_depths[symbol]
            voucher_bid, voucher_ask = get_best_bid_ask(voucher_od)
            current_pos = current_positions.get(symbol, 0)
            max_buy_vol, max_sell_vol = calculate_safe_position_limits(symbol, current_pos)

            if voucher_bid is None and voucher_ask is None: continue

            theoretical_price = black_scholes_call(current_vr_price, strike_price, time_to_expiration, volatility, RISK_FREE_RATE)

            # Moneyness calculation (m_t) for threshold adjustment
            moneyness = math.log(strike_price / current_vr_price) / sqrt_TTE if current_vr_price > 0 else 0
            # Scale threshold based on how far ITM/OTM the option is
            moneyness_factor = 1.0 + max(0, min(OPTIONS_MONEYNESS_THRESHOLD_FACTOR - 1.0,
                                                (abs(moneyness) / OPTIONS_MONEYNESS_LIMIT) * (OPTIONS_MONEYNESS_THRESHOLD_FACTOR - 1.0)))
            adjusted_threshold = OPTIONS_ARB_BASE_THRESHOLD * moneyness_factor

            logger.log(f"    {symbol} (K={strike_price}): Mkt Bid={voucher_bid}, Ask={voucher_ask}, Theo={theoretical_price:.2f}, Pos={current_pos}")
            logger.log(f"      Moneyness={moneyness:.2f}, AdjThreshold={adjusted_threshold:.2f}")


            # --- Arbitrage Check ---
            if voucher_ask is not None and max_buy_vol > 0:
                if voucher_ask < theoretical_price - adjusted_threshold:
                    vol = min(max_buy_vol, OPTIONS_MAX_ARB_SIZE, abs(voucher_od.sell_orders.get(voucher_ask, 0)))
                    if vol > 0:
                        logger.log(f"      ARBITRAGE BUY {vol} @ {voucher_ask}")
                        result[symbol].append(Order(symbol, voucher_ask, vol)); arb_order_placed = True

            if not arb_order_placed and voucher_bid is not None and max_sell_vol > 0: # Check flag
                if voucher_bid > theoretical_price + adjusted_threshold:
                    vol = min(max_sell_vol, OPTIONS_MAX_ARB_SIZE, abs(voucher_od.buy_orders.get(voucher_bid, 0)))
                    if vol > 0:
                        logger.log(f"      ARBITRAGE SELL {vol} @ {voucher_bid}")
                        result[symbol].append(Order(symbol, voucher_bid, -vol)); arb_order_placed = True


            # --- Hesitant Market Making (If no arb order placed and enabled) ---
            if OPTIONS_MM_ENABLE and not arb_order_placed:
                inventory_skew = -current_pos * OPTIONS_MM_SKEW_FACTOR
                mm_bid_price = math.floor(theoretical_price - OPTIONS_MM_SPREAD / 2 + inventory_skew)
                mm_ask_price = math.ceil(theoretical_price + OPTIONS_MM_SPREAD / 2 + inventory_skew)

                if mm_bid_price >= mm_ask_price: mm_ask_price = mm_bid_price + 1 # Ensure spread

                logger.log(f"      MM Quotes: Bid={mm_bid_price}, Ask={mm_ask_price}")

                if max_buy_vol > 0:
                    buy_vol = min(OPTIONS_MM_SIZE, max_buy_vol)
                    if buy_vol > 0 and (voucher_ask is None or mm_bid_price < voucher_ask):
                        logger.log(f"        Placing MM Buy {buy_vol} @ {mm_bid_price}")
                        result[symbol].append(Order(symbol, mm_bid_price, buy_vol))

                if max_sell_vol > 0:
                    sell_vol = min(OPTIONS_MM_SIZE, max_sell_vol)
                    if sell_vol > 0 and (voucher_bid is None or mm_ask_price > voucher_bid):
                        logger.log(f"        Placing MM Sell {sell_vol} @ {mm_ask_price}")
                        result[symbol].append(Order(symbol, mm_ask_price, -sell_vol))


    def run_basket_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        # (Using the safe arbitrage logic from the previous version - no changes needed here)
        logger.log("--- Running Basket Strategy (Safe Arb) ---")
        # ... (Keep the exact logic from algo_combined_safe_arb.py's run_basket_strategy) ...
        for basket_symbol in BASKET_SYMBOLS:
            basket_od = state.order_depths.get(basket_symbol); components = BASKET_COMPONENTS.get(basket_symbol, {})
            if not basket_od or not components: continue
            basket_pos = current_positions.get(basket_symbol, 0); basket_max_buy, basket_max_sell = calculate_safe_position_limits(basket_symbol, basket_pos)
            basket_best_bid, basket_best_ask = get_best_bid_ask(basket_od)
            # Opp 1: Sell Basket, Buy Comp
            if basket_best_bid is not None and basket_max_sell > 0:
                comp_buy_cost = self.calculate_component_value_best_price(state, basket_symbol, use_asks_for_components=True)
                if comp_buy_cost is not None and basket_best_bid > comp_buy_cost + BASKET_PROFIT_MARGIN:
                    vol_limit_basket_bid = abs(basket_od.buy_orders.get(basket_best_bid, 0)); vol_limit_comp_ask = []; vol_limit_comp_pos = []; possible = True
                    for prod, qty in components.items():
                        comp_od = state.order_depths.get(prod); comp_pos = current_positions.get(prod, 0); comp_max_buy, _ = calculate_safe_position_limits(prod, comp_pos)
                        if not comp_od: possible = False; break
                        _, comp_best_ask = get_best_bid_ask(comp_od);
                        if comp_best_ask is None: possible = False; break
                        vol_at_ask = abs(comp_od.sell_orders.get(comp_best_ask, 0));
                        if qty <= 0: continue
                        vol_limit_comp_ask.append(math.floor(vol_at_ask / qty)); vol_limit_comp_pos.append(math.floor(comp_max_buy / qty))
                    if not possible: continue
                    max_volume = min(basket_max_sell, BASKET_MAX_ARB_TRADE_SIZE, vol_limit_basket_bid, min(vol_limit_comp_ask) if vol_limit_comp_ask else 0, min(vol_limit_comp_pos) if vol_limit_comp_pos else 0)
                    if max_volume > 0:
                        logger.log(f"    BASKET ARB Sell {basket_symbol} {max_volume} @ {basket_best_bid}")
                        result[basket_symbol].append(Order(basket_symbol, basket_best_bid, -max_volume))
                        for prod, qty in components.items():
                             comp_od = state.order_depths.get(prod); _, comp_best_ask = get_best_bid_ask(comp_od)
                             if prod not in result: result[prod] = []
                             if comp_best_ask is not None: result[prod].append(Order(prod, comp_best_ask, max_volume * qty))
            # Opp 2: Buy Basket, Sell Comp
            if basket_best_ask is not None and basket_max_buy > 0:
                 comp_sell_rev = self.calculate_component_value_best_price(state, basket_symbol, use_asks_for_components=False)
                 if comp_sell_rev is not None and comp_sell_rev > basket_best_ask + BASKET_PROFIT_MARGIN:
                    vol_limit_basket_ask = abs(basket_od.sell_orders.get(basket_best_ask, 0)); vol_limit_comp_bid = []; vol_limit_comp_pos = []; possible = True
                    for prod, qty in components.items():
                        comp_od = state.order_depths.get(prod); comp_pos = current_positions.get(prod, 0); _, comp_max_sell = calculate_safe_position_limits(prod, comp_pos)
                        if not comp_od: possible = False; break
                        comp_best_bid, _ = get_best_bid_ask(comp_od);
                        if comp_best_bid is None: possible = False; break
                        vol_at_bid = abs(comp_od.buy_orders.get(comp_best_bid, 0));
                        if qty <= 0: continue
                        vol_limit_comp_bid.append(math.floor(vol_at_bid / qty)); vol_limit_comp_pos.append(math.floor(comp_max_sell / qty))
                    if not possible: continue
                    max_volume = min(basket_max_buy, BASKET_MAX_ARB_TRADE_SIZE, vol_limit_basket_ask, min(vol_limit_comp_bid) if vol_limit_comp_bid else 0, min(vol_limit_comp_pos) if vol_limit_comp_pos else 0)
                    if max_volume > 0:
                        logger.log(f"    BASKET ARB Buy {basket_symbol} {max_volume} @ {basket_best_ask}")
                        result[basket_symbol].append(Order(basket_symbol, basket_best_ask, max_volume))
                        for prod, qty in components.items():
                             comp_od = state.order_depths.get(prod); comp_best_bid, _ = get_best_bid_ask(comp_od)
                             if prod not in result: result[prod] = []
                             if comp_best_bid is not None: result[prod].append(Order(prod, comp_best_bid, -max_volume * qty))

    # Add back calculate_component_value_best_price if it wasn't kept
    def calculate_component_value_best_price(self, state: TradingState, basket_symbol: str, use_asks_for_components: bool) -> Optional[float]:
        components = BASKET_COMPONENTS.get(basket_symbol, {});
        if not components: return None
        total_value = 0
        for product, quantity in components.items():
            comp_od = state.order_depths.get(product);
            if not comp_od: return None
            comp_bid, comp_ask = get_best_bid_ask(comp_od)
            if use_asks_for_components:
                if comp_ask is None: return None; total_value += comp_ask * quantity
            else:
                if comp_bid is None: return None; total_value += comp_bid * quantity
        return total_value


    def run_macaron_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]) -> int:
        """ Executes Macaron CONVERSION ARBITRAGE ONLY (Safer). """
        logger.log("--- Running Macaron Strategy (Safe Conv Arb ONLY) ---")
        # (Keep the exact logic from algo_combined_safe_arb.py's run_macaron_strategy)
        conversions = 0; symbol = MAGNIFICENT_MACARONS; current_pos = current_positions.get(symbol, 0)
        max_buy, max_sell = calculate_safe_position_limits(symbol, current_pos)
        if not state.observations or not state.observations.conversionObservations: return 0
        conv_obs = state.observations.conversionObservations.get(symbol);
        if not conv_obs: return 0
        transport_fees=conv_obs.transportFees; import_tariff=conv_obs.importTariff; export_tariff=conv_obs.exportTariff
        pc_bid=conv_obs.bidPrice; pc_ask=conv_obs.askPrice
        effective_pc_buy_price = pc_ask + transport_fees + import_tariff
        effective_pc_sell_price = pc_bid - transport_fees - export_tariff
        order_depth = state.order_depths.get(symbol);
        if not order_depth or (not order_depth.buy_orders and not order_depth.sell_orders): return 0
        best_bid, best_ask = get_best_bid_ask(order_depth)
        logger.log(f"  MM Market: Bid={best_bid}, Ask={best_ask}"); logger.log(f"  MM PC Prices: Eff Buy={effective_pc_buy_price:.2f}, Eff Sell={effective_pc_sell_price:.2f}")
        # Opp 1: Buy PC -> Sell Exch
        if best_bid is not None:
            profit = best_bid - effective_pc_buy_price; required_profit = MM_ARB_PROFIT_THRESHOLD
            if profit > required_profit:
                vol_avail = abs(order_depth.buy_orders.get(best_bid, 0)); vol = min(MM_CONVERSION_LIMIT, max_buy, vol_avail)
                if vol > 0:
                    logger.log(f"    MACARON ARB Conv BUY: {vol}, Sell Exch @ {best_bid} (Profit={profit:.2f})")
                    conversions = vol;
                    if symbol not in result: result[symbol] = []
                    result[symbol].append(Order(symbol, best_bid, -vol)); return conversions
        # Opp 2: Buy Exch -> Sell PC
        if best_ask is not None and conversions == 0:
            profit = effective_pc_sell_price - best_ask; required_profit = MM_ARB_PROFIT_THRESHOLD
            if profit > required_profit and current_pos > 0:
                vol_avail = abs(order_depth.sell_orders.get(best_ask, 0)); vol = min(MM_CONVERSION_LIMIT, current_pos, max_buy, vol_avail)
                if vol > 0:
                    logger.log(f"    MACARON ARB Conv SELL: {vol}, Buy Exch @ {best_ask} (Profit={profit:.2f})")
                    conversions = -vol;
                    if symbol not in result: result[symbol] = []
                    result[symbol].append(Order(symbol, best_ask, vol)); return conversions
        return conversions

    # --- Original Products Strategies (Adapted from FirstAlgo.py) ---

    def update_original_price_history(self, product: str, mid_price: float | None):
        """Update price history for Resin, Kelp, Squid Ink."""
        if mid_price is not None and product in self.original_product_history:
            history = self.original_product_history[product]
            history.append(mid_price)
            # Deque handles maxlen automatically

    def trade_rainforest_resin(self, order_depth: OrderDepth, current_pos: int, result: Dict[str, List[Order]]):
        """ Market making strategy for Rainforest Resin with inventory skewing (Safer Params). """
        product = RAINFOREST_RESIN; max_buy, max_sell = calculate_safe_position_limits(product, current_pos)
        pos_limit = POSITION_LIMITS[product]
        best_bid, best_ask = get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None or best_bid >= best_ask: return

        skew_factor = current_pos / pos_limit if pos_limit != 0 else 0
        price_adjustment = skew_factor * RESIN_MM_SKEW_INTENSITY # No rounding yet
        spread_multiplier = 1.0 + abs(skew_factor) * 0.5
        effective_spread = max(2, RESIN_MM_BASE_SPREAD * spread_multiplier) # Ensure minimum spread >= 2

        reference_price = (best_bid + best_ask) / 2
        desired_bid = math.floor(reference_price - effective_spread / 2 - price_adjustment)
        desired_ask = math.ceil(reference_price + effective_spread / 2 - price_adjustment)

        final_bid = min(desired_bid, best_bid + 1) # Place inside market if needed
        final_ask = max(desired_ask, best_ask - 1) # Place inside market if needed

        if final_bid >= final_ask: final_ask = final_bid + 1 # Ensure spread > 0

        logger.log(f"  RESIN MM: Pos={current_pos}, SkewF={skew_factor:.2f}, EffSpread={effective_spread:.2f}, PxAdj={price_adjustment:.2f}, FinalBid={final_bid}, FinalAsk={final_ask}")

        if max_buy > 0:
            buy_order_size = min(RESIN_MM_MAX_ORDER_SIZE, max_buy)
            if buy_order_size > 0 and final_bid < best_ask: # Safety check
                result[product].append(Order(product, final_bid, buy_order_size))
                logger.log(f"    Placing RESIN MM Buy {buy_order_size} @ {final_bid}")

        if max_sell > 0:
            sell_order_size = min(RESIN_MM_MAX_ORDER_SIZE, max_sell)
            if sell_order_size > 0 and final_ask > best_bid: # Safety check
                result[product].append(Order(product, final_ask, -sell_order_size))
                logger.log(f"    Placing RESIN MM Sell {sell_order_size} @ {final_ask}")


    def trade_kelp(self, order_depth: OrderDepth, current_pos: int, result: Dict[str, List[Order]]):
        """ Passive Moving Average LIMIT ORDER strategy for Kelp (Safer Params). """
        product = KELP; history = self.original_product_history[product]
        max_buy, max_sell = calculate_safe_position_limits(product, current_pos)
        if len(history) < KELP_MA_WINDOW: return
        best_bid, best_ask = get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None or best_bid >= best_ask: return
        mid_price = (best_bid + best_ask) / 2

        recent_avg = statistics.mean(list(history)[-KELP_MA_WINDOW:])
        threshold = max(recent_avg * KELP_THRESHOLD_PCT, KELP_THRESHOLD_ABS)
        buy_threshold_price = recent_avg - threshold
        sell_threshold_price = recent_avg + threshold
        logger.log(f"  KELP MA: Mid={mid_price:.2f}, MA={recent_avg:.2f}, BuyThr={buy_threshold_price:.2f}, SellThr={sell_threshold_price:.2f}")

        # Passive Limit Order Logic
        if mid_price < buy_threshold_price and max_buy > 0:
            buy_price = best_bid + 1 # Place limit order 1 tick better than best bid
            if buy_price < sell_threshold_price and buy_price < best_ask: # Sanity checks
                order_size = min(KELP_ORDER_SIZE, max_buy)
                result[product].append(Order(product, buy_price, order_size))
                logger.log(f"    Placing KELP Buy Limit {order_size} @ {buy_price}")

        elif mid_price > sell_threshold_price and max_sell > 0:
            sell_price = best_ask - 1 # Place limit order 1 tick better than best ask
            if sell_price > buy_threshold_price and sell_price > best_bid: # Sanity checks
                order_size = min(KELP_ORDER_SIZE, max_sell)
                result[product].append(Order(product, sell_price, -order_size))
                logger.log(f"    Placing KELP Sell Limit {order_size} @ {sell_price}")


    def trade_squid_ink(self, order_depth: OrderDepth, current_pos: int, result: Dict[str, List[Order]]):
        """ Passive Mean Reversion LIMIT ORDER strategy for Squid Ink (Safer Params). """
        product = SQUID_INK; history = self.original_product_history[product]
        max_buy, max_sell = calculate_safe_position_limits(product, current_pos)
        if len(history) < SQUID_LONG_WINDOW: return
        best_bid, best_ask = get_best_bid_ask(order_depth)
        if best_bid is None or best_ask is None or best_bid >= best_ask: return

        short_term_avg = statistics.mean(list(history)[-SQUID_SHORT_WINDOW:])
        long_term_avg = statistics.mean(list(history)[-SQUID_LONG_WINDOW:])
        deviation = short_term_avg - long_term_avg
        logger.log(f"  SQUID MA: Short={short_term_avg:.2f}, Long={long_term_avg:.2f}, Dev={deviation:.2f}, Threshold={SQUID_VOLATILITY_THRESHOLD}")

        # Only trade if deviation is significant (Hint)
        if abs(deviation) > SQUID_VOLATILITY_THRESHOLD:
            # Expect reversion UP -> Place BUY limit
            if deviation < 0 and max_buy > 0:
                buy_price = best_bid + 1
                if buy_price < best_ask: # Safety check
                    order_size = min(SQUID_ORDER_SIZE, max_buy)
                    result[product].append(Order(product, buy_price, order_size))
                    logger.log(f"    Placing SQUID Buy Limit {order_size} @ {buy_price} (Mean Reversion)")

            # Expect reversion DOWN -> Place SELL limit
            elif deviation > 0 and max_sell > 0:
                sell_price = best_ask - 1
                if sell_price > best_bid: # Safety check
                    order_size = min(SQUID_ORDER_SIZE, max_sell)
                    result[product].append(Order(product, sell_price, -order_size))
                    logger.log(f"    Placing SQUID Sell Limit {order_size} @ {sell_price} (Mean Reversion)")

    def run_original_products_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        """ Executes strategies for Resin, Kelp, Squid Ink. """
        logger.log("--- Running Original Products Strategy ---")
        for product in ORIGINAL_SYMBOLS:
            if product not in state.order_depths: continue
            order_depth = state.order_depths[product]
            current_pos = current_positions.get(product, 0)

            # Update history first
            mid_price = get_mid_price(order_depth)
            self.update_original_price_history(product, mid_price)

            # Apply strategy
            if product == RAINFOREST_RESIN:
                self.trade_rainforest_resin(order_depth, current_pos, result)
            elif product == KELP:
                self.trade_kelp(order_depth, current_pos, result)
            elif product == SQUID_INK:
                self.trade_squid_ink(order_depth, current_pos, result)


    # --------------------------------------------------------------------------
    # Main Execution Method
    # --------------------------------------------------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        logger.log(f"\n===== Timestamp: {state.timestamp} (Safer MM Version) =====")
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
        try: self.run_original_products_strategy(state, current_positions, result) # Run Resin, Kelp, Squid strategies
        except Exception as e: logger.log(f"!!! ERROR in Original Products Strategy: {e}"); import traceback; traceback.print_exc()
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