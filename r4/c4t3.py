# Combined Algorithm: algo_combined_safer_mm_v2.py
import json
from datamodel import (
    TradingState, OrderDepth, Order, Trade, Symbol, ProsperityEncoder,
    Listing, ConversionObservation, Observation, Product, Position
)
from typing import List, Dict, Tuple, Optional, Any
import math
import numpy as np
import collections
import statistics

# ------------------------------------------------------------------------------
# Configuration & Constants
# ------------------------------------------------------------------------------

# --- Product Definitions ---
# (Keep all product definitions: VOLCANIC_ROCK, VOUCHERS, MACARONS, ORIGINALS, BASKETS, COMPONENTS)
VOLCANIC_ROCK = "VOLCANIC_ROCK"; VOUCHER_PREFIX = "VOLCANIC_ROCK_VOUCHER_"
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"; RAINFOREST_RESIN = "RAINFOREST_RESIN"
KELP = "KELP"; SQUID_INK = "SQUID_INK"; CROISSANTS = "CROISSANTS"; JAMS = "JAMS"
DJEMBES = "DJEMBES"; PICNIC_BASKET1 = "PICNIC_BASKET1"; PICNIC_BASKET2 = "PICNIC_BASKET2"

# --- All Symbols Traded ---
VOUCHER_STRIKES = { f"{VOUCHER_PREFIX}9500": 9500, f"{VOUCHER_PREFIX}9750": 9750, f"{VOUCHER_PREFIX}10000": 10000, f"{VOUCHER_PREFIX}10250": 10250, f"{VOUCHER_PREFIX}10500": 10500 }
OPTIONS_SYMBOLS = [VOLCANIC_ROCK] + list(VOUCHER_STRIKES.keys())
MACARON_SYMBOLS = [MAGNIFICENT_MACARONS]
ORIGINAL_SYMBOLS = [RAINFOREST_RESIN, KELP, SQUID_INK]
BASKET_COMPONENTS = { PICNIC_BASKET1: {CROISSANTS: 6, JAMS: 3, DJEMBES: 1}, PICNIC_BASKET2: {CROISSANTS: 4, JAMS: 2} }
BASKET_SYMBOLS = list(BASKET_COMPONENTS.keys()); COMPONENT_SYMBOLS = [CROISSANTS, JAMS, DJEMBES]
ALL_SYMBOLS = list(set(OPTIONS_SYMBOLS + MACARON_SYMBOLS + ORIGINAL_SYMBOLS + BASKET_SYMBOLS + COMPONENT_SYMBOLS))

# --- Position Limits ---
POSITION_LIMITS = {
    VOLCANIC_ROCK: 400, f"{VOUCHER_PREFIX}9500": 200, f"{VOUCHER_PREFIX}9750": 200, f"{VOUCHER_PREFIX}10000": 200, f"{VOUCHER_PREFIX}10250": 200, f"{VOUCHER_PREFIX}10500": 200,
    MAGNIFICENT_MACARONS: 75, RAINFOREST_RESIN: 50, KELP: 50, SQUID_INK: 50,
    CROISSANTS: 250, JAMS: 350, DJEMBES: 60, PICNIC_BASKET1: 60, PICNIC_BASKET2: 100,
}
POSITION_BUFFER_PCT = 0.15 # Keep 15% buffer

# --- Options Strategy Parameters ---
VOLATILITY_WINDOW = 100; MIN_VOLATILITY_POINTS = 20; TOTAL_EXPIRATION_DAYS = 7; RISK_FREE_RATE = 0.0
OPTIONS_ARB_BASE_THRESHOLD = 6.0; OPTIONS_MONEYNESS_THRESHOLD_FACTOR = 3.0; OPTIONS_MONEYNESS_LIMIT = 1.5
OPTIONS_MAX_ARB_SIZE = 10; OPTIONS_MM_ENABLE = True; OPTIONS_MM_SPREAD = 8.0
OPTIONS_MM_SIZE = 3; OPTIONS_MM_SKEW_FACTOR = 0.05

# --- Macaron Strategy Parameters ---
MM_CONVERSION_LIMIT = 10
MM_ARB_PROFIT_THRESHOLD = 5.0 # Slightly lower than pure arb, but still high
# Fair Value / MM (Above CSI)
MM_FAIR_VALUE_SMA_WINDOW = 20; MM_FV_WEIGHT_SUGAR = 0.3; MM_FV_WEIGHT_SUNLIGHT = 0.1
MM_BASE_SPREAD = 5.0              # Increased base spread
MM_TRANSPORT_FEE_FACTOR = 0.25    # Increased impact of transport fees on spread
MM_INVENTORY_SKEW_FACTOR = 0.08   # Increased skew intensity
MM_MAX_SPREAD = 8.0               # Allow slightly wider max spread due to fees
MM_ORDER_SIZE = 3                 # Reduced base size
MM_ENABLE = True
# CSI Parameters (TUNABLE)
CRITICAL_SUNLIGHT_INDEX = 750.0   # User suggested CSI
CSI_MM_BUY_AGGRESSION = 0.3       # How much closer to FV (0=center, 1=at fv-half_spread) - Less aggressive than before
CSI_MM_SELL_SPREAD_MULT = 4.0     # Increase multiplier further
CSI_MM_SELL_SIZE_FACTOR = 0.5
CSI_MM_BUY_SIZE_FACTOR = 1.2      # Slightly larger buys below CSI

# --- Basket Strategy Parameters ---
BASKET_PROFIT_MARGIN = 6.0
BASKET_MAX_ARB_TRADE_SIZE = 8

# --- Original Products Strategy Parameters ---
ORIGINAL_HISTORY_MAX_LENGTH = 100; RESIN_MM_BASE_SPREAD = 4; RESIN_MM_SKEW_INTENSITY = 0.5
RESIN_MM_MAX_ORDER_SIZE = 5; KELP_MA_WINDOW = 20; KELP_THRESHOLD_PCT = 0.003
KELP_THRESHOLD_ABS = 4; KELP_ORDER_SIZE = 4; SQUID_SHORT_WINDOW = 5; SQUID_LONG_WINDOW = 40
SQUID_VOLATILITY_THRESHOLD = 4; SQUID_ORDER_SIZE = 3

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
def norm_cdf(x): return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
def black_scholes_call(S, K, T, sigma, r=0.0):
    if sigma <= 0 or T <= 0: return max(0.0, S - K)
    ratio = max(1e-9, S / K); d1_num = math.log(ratio) + (r + 0.5 * sigma ** 2) * T
    d1_den = sigma * math.sqrt(T);
    if d1_den == 0 : return max(0.0, S-K)
    d1 = d1_num / d1_den; d2 = d1 - sigma * math.sqrt(T)
    try: call_price = (S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2))
    except Exception as e: logger.log(f"WARN: BS error: {e}"); return max(0.0, S-K)
    return call_price
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
        logger.log("Initializing Combined Trader (Safer MM v2)")
        self.volcanic_rock_price_history = collections.deque(maxlen=VOLATILITY_WINDOW + 5)
        self.mm_mid_price_history = collections.deque(maxlen=MM_FAIR_VALUE_SMA_WINDOW + 5) # History needed again for MM
        self.original_product_history = { prod: collections.deque(maxlen=ORIGINAL_HISTORY_MAX_LENGTH + 5) for prod in ORIGINAL_SYMBOLS }
        self.current_day = -1; self.current_volatility = None

    def load_state(self, traderData: str):
        # Load VR, MM, and Original Product histories
        if traderData:
            try:
                loaded_data = json.loads(traderData)
                vr_hist = loaded_data.get("vr_history", [])
                mm_hist = loaded_data.get("mm_history", []) # Load MM history
                self.volcanic_rock_price_history = collections.deque(vr_hist, maxlen=VOLATILITY_WINDOW + 5)
                self.mm_mid_price_history = collections.deque(mm_hist, maxlen=MM_FAIR_VALUE_SMA_WINDOW + 5) # Assign MM history
                orig_hist = loaded_data.get("orig_history", {})
                for prod, hist_list in orig_hist.items():
                    if prod in self.original_product_history:
                        self.original_product_history[prod] = collections.deque(hist_list, maxlen=ORIGINAL_HISTORY_MAX_LENGTH + 5)
                logger.log(f"Loaded Histories: VR({len(self.volcanic_rock_price_history)}), MM({len(self.mm_mid_price_history)}), Originals")
            except Exception as e:
                logger.log(f"Error loading traderData: {e}. Starting fresh histories.")
                self.volcanic_rock_price_history.clear(); self.mm_mid_price_history.clear()
                for prod in self.original_product_history: self.original_product_history[prod].clear()
        else: logger.log("No traderData found, starting fresh histories.")

    def save_state(self) -> str:
        # Save VR, MM, and Original Product histories
        try:
            state_to_save = {
                "vr_history": list(self.volcanic_rock_price_history),
                "mm_history": list(self.mm_mid_price_history), # Save MM history
                "orig_history": {prod: list(hist) for prod, hist in self.original_product_history.items()},
            }
            return json.dumps(state_to_save)
        except Exception as e: logger.log(f"Error saving state: {e}"); return ""

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
        # (Using the logic from algo_combined_safer_mm.py - no changes needed here)
        logger.log("--- Running Options Strategy (Arb + Hesitant MM) ---")
        vr_order_depth = state.order_depths.get(VOLCANIC_ROCK); current_vr_price = None
        if vr_order_depth: current_vr_price = get_mid_price(vr_order_depth);
        if current_vr_price is not None: self.volcanic_rock_price_history.append(current_vr_price)
        volatility = self.calculate_historical_volatility()
        day_from_timestamp = state.timestamp // 1000000;
        if day_from_timestamp != self.current_day: self.current_day = day_from_timestamp; logger.log(f"--- Day {self.current_day + 1} ---")
        days_remaining = max(0.001, TOTAL_EXPIRATION_DAYS - self.current_day); time_to_expiration = days_remaining / TOTAL_EXPIRATION_DAYS
        sqrt_TTE = math.sqrt(time_to_expiration) if time_to_expiration > 0 else 0
        if current_vr_price is None or volatility is None or sqrt_TTE == 0: logger.log(f"  Insufficient data for Options"); return
        logger.log(f"  Options Input: S={current_vr_price:.2f}, sigma={volatility:.4f}, T={time_to_expiration:.3f}")
        for symbol, strike_price in VOUCHER_STRIKES.items():
            arb_order_placed = False;
            if symbol not in state.order_depths: continue
            voucher_od = state.order_depths[symbol]; voucher_bid, voucher_ask = get_best_bid_ask(voucher_od)
            current_pos = current_positions.get(symbol, 0); max_buy_vol, max_sell_vol = calculate_safe_position_limits(symbol, current_pos)
            if voucher_bid is None and voucher_ask is None: continue
            theoretical_price = black_scholes_call(current_vr_price, strike_price, time_to_expiration, volatility, RISK_FREE_RATE)
            moneyness = math.log(strike_price / current_vr_price) / sqrt_TTE if current_vr_price > 0 else 0
            moneyness_factor = 1.0 + max(0, min(OPTIONS_MONEYNESS_THRESHOLD_FACTOR - 1.0, (abs(moneyness) / OPTIONS_MONEYNESS_LIMIT) * (OPTIONS_MONEYNESS_THRESHOLD_FACTOR - 1.0)))
            adjusted_threshold = OPTIONS_ARB_BASE_THRESHOLD * moneyness_factor
            logger.log(f"    {symbol} (K={strike_price}): Mkt Bid={voucher_bid}, Ask={voucher_ask}, Theo={theoretical_price:.2f}, Pos={current_pos}, AdjThr={adjusted_threshold:.2f}")
            # Arb Check
            if voucher_ask is not None and max_buy_vol > 0 and voucher_ask < theoretical_price - adjusted_threshold:
                vol = min(max_buy_vol, OPTIONS_MAX_ARB_SIZE, abs(voucher_od.sell_orders.get(voucher_ask, 0)));
                if vol > 0: logger.log(f"      ARBITRAGE BUY {vol} @ {voucher_ask}"); result[symbol].append(Order(symbol, voucher_ask, vol)); arb_order_placed = True
            if not arb_order_placed and voucher_bid is not None and max_sell_vol > 0 and voucher_bid > theoretical_price + adjusted_threshold:
                vol = min(max_sell_vol, OPTIONS_MAX_ARB_SIZE, abs(voucher_od.buy_orders.get(voucher_bid, 0)));
                if vol > 0: logger.log(f"      ARBITRAGE SELL {vol} @ {voucher_bid}"); result[symbol].append(Order(symbol, voucher_bid, -vol)); arb_order_placed = True
            # Hesitant MM
            if OPTIONS_MM_ENABLE and not arb_order_placed:
                inventory_skew = -current_pos * OPTIONS_MM_SKEW_FACTOR; mm_bid_price = math.floor(theoretical_price - OPTIONS_MM_SPREAD / 2 + inventory_skew); mm_ask_price = math.ceil(theoretical_price + OPTIONS_MM_SPREAD / 2 + inventory_skew)
                if mm_bid_price >= mm_ask_price: mm_ask_price = mm_bid_price + 1
                logger.log(f"      MM Quotes: Bid={mm_bid_price}, Ask={mm_ask_price}")
                if max_buy_vol > 0:
                    buy_vol = min(OPTIONS_MM_SIZE, max_buy_vol);
                    if buy_vol > 0 and (voucher_ask is None or mm_bid_price < voucher_ask): logger.log(f"        Placing MM Buy {buy_vol} @ {mm_bid_price}"); result[symbol].append(Order(symbol, mm_bid_price, buy_vol))
                if max_sell_vol > 0:
                    sell_vol = min(OPTIONS_MM_SIZE, max_sell_vol);
                    if sell_vol > 0 and (voucher_bid is None or mm_ask_price > voucher_bid): logger.log(f"        Placing MM Sell {sell_vol} @ {mm_ask_price}"); result[symbol].append(Order(symbol, mm_ask_price, -sell_vol))


    def run_basket_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        # (Using the safe arbitrage logic - no changes needed here)
        logger.log("--- Running Basket Strategy (Safe Arb) ---")
        # ... (Exact logic from previous version) ...
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


    def calculate_component_value_best_price(self, state: TradingState, basket_symbol: str, use_asks_for_components: bool) -> Optional[float]:
        # (same as before)
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


    def calculate_mm_fair_value(self, sma_base: float, sugar_price: float, sunlight_index: float) -> float:
        """ Estimates Macaron fair value based on SMA and factors. """
        fair_value = sma_base + (MM_FV_WEIGHT_SUGAR * sugar_price) - (MM_FV_WEIGHT_SUNLIGHT * sunlight_index)
        return fair_value

    def run_macaron_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]) -> int:
        """ Executes Macaron conversion arbitrage and CSI-aware MM. """
        logger.log("--- Running Macaron Strategy (Conv Arb + CSI MM v2) ---")
        conversions = 0; symbol = MAGNIFICENT_MACARONS; current_pos = current_positions.get(symbol, 0)
        max_buy, max_sell = calculate_safe_position_limits(symbol, current_pos)

        # Observations and PC Prices
        if not state.observations or not state.observations.conversionObservations: return 0
        conv_obs = state.observations.conversionObservations.get(symbol);
        if not conv_obs: return 0
        transport_fees=conv_obs.transportFees; import_tariff=conv_obs.importTariff; export_tariff=conv_obs.exportTariff
        pc_bid=conv_obs.bidPrice; pc_ask=conv_obs.askPrice; sunlight_index=conv_obs.sunlightIndex; sugar_price=conv_obs.sugarPrice
        effective_pc_buy_price = pc_ask + transport_fees + import_tariff
        effective_pc_sell_price = pc_bid - transport_fees - export_tariff

        # Exchange Data and History Update
        order_depth = state.order_depths.get(symbol);
        if not order_depth or (not order_depth.buy_orders and not order_depth.sell_orders): return 0
        best_bid, best_ask = get_best_bid_ask(order_depth); mid_price = get_mid_price(order_depth)
        if mid_price is not None: self.mm_mid_price_history.append(mid_price)
        logger.log(f"  MM Factors: Sunlight={sunlight_index:.2f}, Sugar={sugar_price:.2f}, Transport={transport_fees:.2f}")
        logger.log(f"  MM Market: Bid={best_bid}, Ask={best_ask}, Mid={mid_price}")
        logger.log(f"  MM PC Prices: Eff Buy={effective_pc_buy_price:.2f}, Eff Sell={effective_pc_sell_price:.2f}")

        # --- PRIORITY 1: Conversion Arbitrage ---
        # (Same logic as before, using MM_ARB_PROFIT_THRESHOLD)
        arb_order_placed = False
        if best_bid is not None and best_bid - effective_pc_buy_price > MM_ARB_PROFIT_THRESHOLD:
            vol = min(MM_CONVERSION_LIMIT, max_buy, abs(order_depth.buy_orders.get(best_bid, 0)))
            if vol > 0:
                logger.log(f"    ACTION: MM Arb 1 Executing - Conv BUY: {vol}, Sell Exch @ {best_bid}")
                conversions = vol; arb_order_placed = True
                if symbol not in result: result[symbol] = []
                result[symbol].append(Order(symbol, best_bid, -vol))
        if not arb_order_placed and best_ask is not None and effective_pc_sell_price - best_ask > MM_ARB_PROFIT_THRESHOLD and current_pos > 0:
             vol = min(MM_CONVERSION_LIMIT, current_pos, max_buy, abs(order_depth.sell_orders.get(best_ask, 0)))
             if vol > 0:
                logger.log(f"    ACTION: MM Arb 2 Executing - Conv SELL: {vol}, Buy Exch @ {best_ask}")
                conversions = -vol; arb_order_placed = True
                if symbol not in result: result[symbol] = []
                result[symbol].append(Order(symbol, best_ask, vol))

        # --- PRIORITY 2: Market Making (if no arb and enabled) ---
        if MM_ENABLE and not arb_order_placed:
            logger.log("  Attempting MM...")
            sma_base = np.mean(list(self.mm_mid_price_history)[-MM_FAIR_VALUE_SMA_WINDOW:]) if len(self.mm_mid_price_history) >= MM_FAIR_VALUE_SMA_WINDOW else None

            if sma_base is not None and best_bid is not None and best_ask is not None:
                fair_value = self.calculate_mm_fair_value(sma_base, sugar_price, sunlight_index)
                logger.log(f"  MM Fair Value: SMA={sma_base:.2f} -> FV={fair_value:.2f}")
                below_csi = sunlight_index < CRITICAL_SUNLIGHT_INDEX

                # Define base MM parameters before CSI check
                base_buy_size = base_sell_size = MM_ORDER_SIZE
                inventory_skew = -current_pos * MM_INVENTORY_SKEW_FACTOR # Base skew

                if below_csi: # Below CSI - Aggressive Buy / Passive Sell
                    logger.log(f"  MM Condition: BELOW CSI ({sunlight_index:.2f} < {CRITICAL_SUNLIGHT_INDEX})")
                    # Buy side slightly more aggressive
                    buy_half_spread = (MM_BASE_SPREAD / 2) * 0.8 # Tighter spread for buy
                    buy_target_price = fair_value - buy_half_spread + inventory_skew * 1.5 # Stronger skew impact on buy
                    # Place bid closer to target, but not crossing mid yet easily
                    mm_buy_price = math.floor(buy_target_price * (1-CSI_MM_BUY_AGGRESSION) + (fair_value - buy_half_spread) * CSI_MM_BUY_AGGRESSION)
                    mm_buy_price = min(mm_buy_price, best_ask -1) # Don't cross ask

                    # Sell side very passive
                    sell_half_spread = (MM_BASE_SPREAD / 2 + transport_fees * MM_TRANSPORT_FEE_FACTOR) * CSI_MM_SELL_SPREAD_MULT # Very wide
                    mm_ask_price = math.ceil(fair_value + sell_half_spread + inventory_skew * 0.5) # Less skew impact on sell
                    mm_ask_price = max(mm_ask_price, mm_buy_price + int(MM_BASE_SPREAD * 0.75)) # Ensure wide gap

                    buy_size = math.ceil(base_buy_size * CSI_MM_BUY_SIZE_FACTOR)
                    sell_size = math.ceil(base_sell_size * CSI_MM_SELL_SIZE_FACTOR)

                else: # Above CSI - Normal Cautious MM
                    logger.log(f"  MM Condition: ABOVE or EQUAL CSI - Normal Cautious MM")
                    transport_fee_spread_comp = transport_fees * MM_TRANSPORT_FEE_FACTOR # Explicitly include transport cost effect
                    # Spread slightly widens with inventory deviation
                    inventory_spread_comp = abs(current_pos) * 0.05 # Smaller factor than skew
                    dynamic_spread = min(MM_BASE_SPREAD + transport_fee_spread_comp + inventory_spread_comp, MM_MAX_SPREAD)
                    half_spread = dynamic_spread / 2.0
                    logger.log(f"  MM Spread(Norm): Base={MM_BASE_SPREAD:.1f} Fee={transport_fee_spread_comp:.2f} Inv={inventory_spread_comp:.2f} -> Dyn={dynamic_spread:.2f}")

                    mm_buy_price = math.floor(fair_value - half_spread + inventory_skew)
                    mm_ask_price = math.ceil(fair_value + half_spread + inventory_skew)
                    if mm_buy_price >= mm_ask_price: mm_ask_price = mm_buy_price + 1
                    buy_size = sell_size = base_buy_size

                logger.log(f"  MM Quotes: Bid={mm_buy_price} (Size={buy_size}), Ask={mm_ask_price} (Size={sell_size})")

                # Place orders
                if max_buy > 0:
                    final_buy_vol = min(buy_size, max_buy);
                    if final_buy_vol > 0 and (best_ask is None or mm_buy_price < best_ask):
                        logger.log(f"    Placing MM Buy Order: {final_buy_vol} @ {mm_buy_price}")
                        if symbol not in result: result[symbol] = []
                        result[symbol].append(Order(symbol, mm_buy_price, final_buy_vol))
                if max_sell > 0:
                    final_sell_vol = min(sell_size, max_sell);
                    if final_sell_vol > 0 and (best_bid is None or mm_ask_price > best_bid):
                        logger.log(f"    Placing MM Sell Order: {final_sell_vol} @ {mm_ask_price}")
                        if symbol not in result: result[symbol] = []
                        result[symbol].append(Order(symbol, mm_ask_price, -final_sell_vol))

            else: logger.log("  Skipping MM: Not enough data or prices.")
        elif not MM_ENABLE: logger.log("  Macaron MM is disabled.")

        return conversions # Return conversions count from this strategy


    def run_original_products_strategy(self, state: TradingState, current_positions: Dict[str, int], result: Dict[str, List[Order]]):
        # (Using the safe MM logic from algo_combined_safer_mm.py - no changes needed here)
        logger.log("--- Running Original Products Strategy ---")
        # ... (Keep the exact logic from algo_combined_safer_mm.py's run_original_products_strategy) ...
        for product in ORIGINAL_SYMBOLS:
            if product not in state.order_depths: continue
            order_depth = state.order_depths[product]; current_pos = current_positions.get(product, 0)
            mid_price = get_mid_price(order_depth); self.update_original_price_history(product, mid_price)
            if product == RAINFOREST_RESIN: self.trade_rainforest_resin(order_depth, current_pos, result)
            elif product == KELP: self.trade_kelp(order_depth, current_pos, result)
            elif product == SQUID_INK: self.trade_squid_ink(order_depth, current_pos, result)

    # Add back original product methods (Resin, Kelp, Squid)
    def update_original_price_history(self, product: str, mid_price: float | None):
        if mid_price is not None and product in self.original_product_history:
            self.original_product_history[product].append(mid_price)
    def trade_rainforest_resin(self, od: OrderDepth, pos: int, result: Dict[str, List[Order]]):
        prod = RAINFOREST_RESIN; max_b, max_s = calculate_safe_position_limits(prod, pos); lim = POSITION_LIMITS[prod]
        bid, ask = get_best_bid_ask(od);
        if bid is None or ask is None or bid >= ask: return
        skew_f = pos / lim if lim != 0 else 0; px_adj = skew_f * RESIN_MM_SKEW_INTENSITY
        spread_m = 1.0 + abs(skew_f) * 0.5; eff_spread = max(2, RESIN_MM_BASE_SPREAD * spread_m)
        ref_px = (bid + ask) / 2; des_bid = math.floor(ref_px - eff_spread / 2 - px_adj); des_ask = math.ceil(ref_px + eff_spread / 2 - px_adj)
        fin_bid = min(des_bid, bid + 1); fin_ask = max(des_ask, ask - 1)
        if fin_bid >= fin_ask: fin_ask = fin_bid + 1
        logger.log(f"  RESIN MM: Pos={pos}, SkewF={skew_f:.2f}, EffSpread={eff_spread:.2f}, PxAdj={px_adj:.2f}, Bid={fin_bid}, Ask={fin_ask}")
        if max_b > 0:
            buy_s = min(RESIN_MM_MAX_ORDER_SIZE, max_b);
            if buy_s > 0 and fin_bid < ask: logger.log(f"    Placing RESIN MM Buy {buy_s} @ {fin_bid}"); result[prod].append(Order(prod, fin_bid, buy_s))
        if max_s > 0:
            sell_s = min(RESIN_MM_MAX_ORDER_SIZE, max_s);
            if sell_s > 0 and fin_ask > bid: logger.log(f"    Placing RESIN MM Sell {sell_s} @ {fin_ask}"); result[prod].append(Order(prod, fin_ask, -sell_s))
    def trade_kelp(self, od: OrderDepth, pos: int, result: Dict[str, List[Order]]):
        prod = KELP; hist = self.original_product_history[prod]; max_b, max_s = calculate_safe_position_limits(prod, pos)
        if len(hist) < KELP_MA_WINDOW: return
        bid, ask = get_best_bid_ask(od);
        if bid is None or ask is None or bid >= ask: return
        mid = (bid + ask) / 2; avg = statistics.mean(list(hist)[-KELP_MA_WINDOW:])
        thr = max(avg * KELP_THRESHOLD_PCT, KELP_THRESHOLD_ABS); buy_thr = avg - thr; sell_thr = avg + thr
        logger.log(f"  KELP MA: Mid={mid:.2f}, MA={avg:.2f}, BuyThr={buy_thr:.2f}, SellThr={sell_thr:.2f}")
        if mid < buy_thr and max_b > 0:
            buy_px = bid + 1;
            if buy_px < sell_thr and buy_px < ask: size = min(KELP_ORDER_SIZE, max_b); logger.log(f"    Placing KELP Buy Limit {size} @ {buy_px}"); result[prod].append(Order(prod, buy_px, size))
        elif mid > sell_thr and max_s > 0:
            sell_px = ask - 1;
            if sell_px > buy_thr and sell_px > bid: size = min(KELP_ORDER_SIZE, max_s); logger.log(f"    Placing KELP Sell Limit {size} @ {sell_px}"); result[prod].append(Order(prod, sell_px, -size))
    def trade_squid_ink(self, od: OrderDepth, pos: int, result: Dict[str, List[Order]]):
        prod = SQUID_INK; hist = self.original_product_history[prod]; max_b, max_s = calculate_safe_position_limits(prod, pos)
        if len(hist) < SQUID_LONG_WINDOW: return
        bid, ask = get_best_bid_ask(od);
        if bid is None or ask is None or bid >= ask: return
        short_avg = statistics.mean(list(hist)[-SQUID_SHORT_WINDOW:]); long_avg = statistics.mean(list(hist)[-SQUID_LONG_WINDOW:])
        dev = short_avg - long_avg; logger.log(f"  SQUID MA: Short={short_avg:.2f}, Long={long_avg:.2f}, Dev={dev:.2f}, Threshold={SQUID_VOLATILITY_THRESHOLD}")
        if abs(dev) > SQUID_VOLATILITY_THRESHOLD:
            if dev < 0 and max_b > 0: # Reversion UP -> BUY
                buy_px = bid + 1;
                if buy_px < ask: size = min(SQUID_ORDER_SIZE, max_b); logger.log(f"    Placing SQUID Buy Limit {size} @ {buy_px} (Mean Rev)"); result[prod].append(Order(prod, buy_px, size))
            elif dev > 0 and max_s > 0: # Reversion DOWN -> SELL
                sell_px = ask - 1;
                if sell_px > bid: size = min(SQUID_ORDER_SIZE, max_s); logger.log(f"    Placing SQUID Sell Limit {size} @ {sell_px} (Mean Rev)"); result[prod].append(Order(prod, sell_px, -size))


    # --------------------------------------------------------------------------
    # Main Execution Method
    # --------------------------------------------------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        logger.log(f"\n===== Timestamp: {state.timestamp} (Safer MM v2) =====")
        self.load_state(state.traderData)
        current_positions = state.position if state.position is not None else {}
        logger.log(f"  Current Positions: {current_positions}")
        result: Dict[str, List[Order]] = {symbol: [] for symbol in ALL_SYMBOLS}
        total_conversions = 0

        # Execute Strategies by Group
        try: self.run_options_strategy(state, current_positions, result)
        except Exception as e: logger.log(f"!!! ERROR Options: {e}"); import traceback; traceback.print_exc()
        try: self.run_basket_strategy(state, current_positions, result)
        except Exception as e: logger.log(f"!!! ERROR Basket: {e}"); import traceback; traceback.print_exc()
        try: self.run_original_products_strategy(state, current_positions, result)
        except Exception as e: logger.log(f"!!! ERROR Originals: {e}"); import traceback; traceback.print_exc()
        try:
             macaron_conversions = self.run_macaron_strategy(state, current_positions, result)
             total_conversions += macaron_conversions
        except Exception as e: logger.log(f"!!! ERROR Macaron: {e}"); import traceback; traceback.print_exc()

        # Clean up and Save State
        final_result = {symbol: orders for symbol, orders in result.items() if orders}
        logger.log(f"  Final Orders Sent: {final_result}")
        logger.log(f"  Total Conversions Sent: {total_conversions}")
        traderData = self.save_state()
        logger.log(f"===== End Timestamp: {state.timestamp} =====")
        logger.flush(state, final_result, total_conversions, traderData)
        return final_result, total_conversions, traderData