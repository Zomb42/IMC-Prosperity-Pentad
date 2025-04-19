import json
from datamodel import (
    TradingState, OrderDepth, Order, Trade, Symbol, ProsperityEncoder,
    Listing, ConversionObservation, Observation, Product, Position
)
from typing import List, Dict, Tuple, Optional, Any
import math
import collections

# --- Constants ---
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
POSITION_LIMIT_MM = 75
CONVERSION_LIMIT_MM = 10
STORAGE_COST_PER_UNIT_PER_TICK = 0.1
POSITION_BUFFER_PCT = 0.10
MIN_PROFIT_THRESHOLD = 0.5
STORAGE_COST_BIAS_BUY = 0.2
STORAGE_COST_BIAS_SELL = -0.1

# --- Hesitant MM Parameters (TUNABLE) ---
MM_HESITANT_ENABLE = True       # Set to False to disable this MM block
MM_HESITANT_SPREAD = 6          # Wider fixed spread for hesitancy
MM_HESITANT_SIZE = 2            # Smaller order size for hesitancy
NEUTRAL_SUNLIGHT = 53.0       # Assumed neutral sunlight level (needs tuning based on data!)
SUNLIGHT_SENSITIVITY_FACTOR = 0.01 # How much each unit of sunlight deviation shifts the price center (needs tuning!)
MAX_MARKET_SPREAD_FOR_MM = 8    # Only place MM orders if market bid-ask spread is not wider than this

# (Logger class remains the same)
class Logger:
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

class Trader:
    # __init__, get_mid_price, get_best_bid_ask, calculate_safe_position_limits remain the same
    def __init__(self):
        logger.log("Initializing Trader")
        pass

    def get_mid_price(self, order_depth: OrderDepth) -> Optional[float]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        if best_bid is not None and best_ask is not None: return (best_bid + best_ask) / 2.0
        elif best_bid is not None: return float(best_bid)
        elif best_ask is not None: return float(best_ask)
        else: return None

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def calculate_safe_position_limits(self, symbol: str, current_position: int) -> Tuple[int, int]:
        if symbol == MAGNIFICENT_MACARONS: max_pos = POSITION_LIMIT_MM
        else: return 0, 0
        buffer = int(max_pos * POSITION_BUFFER_PCT)
        effective_max_limit = max_pos - buffer
        effective_min_limit = -max_pos + buffer
        max_buy = effective_max_limit - current_position
        max_sell = current_position - effective_min_limit
        safe_buy_vol = max(0, max_buy)
        safe_sell_vol = max(0, max_sell)
        return safe_buy_vol, safe_sell_vol

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {MAGNIFICENT_MACARONS: []}
        conversions = 0
        traderData = state.traderData

        current_pos_mm = state.position.get(MAGNIFICENT_MACARONS, 0)
        logger.log(f"--- Timestamp: {state.timestamp}, Position MM: {current_pos_mm} ---")

        # --- Access Observations ---
        if not state.observations or not state.observations.conversionObservations:
            logger.log("WARN: No conversion observations found.")
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData
        conv_obs = state.observations.conversionObservations.get(MAGNIFICENT_MACARONS)
        if not conv_obs:
            logger.log(f"WARN: No conversion observation for {MAGNIFICENT_MACARONS}")
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData

        transport_fees = conv_obs.transportFees
        import_tariff = conv_obs.importTariff
        export_tariff = conv_obs.exportTariff
        pc_bid = conv_obs.bidPrice
        pc_ask = conv_obs.askPrice
        sunlight_index = conv_obs.sunlightIndex
        sugar_price = conv_obs.sugarPrice # Still available, just not used in this MM logic
        logger.log(f"  Observations: PC Bid={pc_bid}, PC Ask={pc_ask}, Transport={transport_fees}, ImpT={import_tariff}, ExpT={export_tariff}")
        logger.log(f"  Factors: Sunlight={sunlight_index:.2f}, Sugar={sugar_price:.2f}")

        # --- Calculate Effective PC Prices ---
        effective_pc_buy_price = pc_ask + transport_fees + import_tariff
        effective_pc_sell_price = pc_bid - transport_fees - export_tariff
        logger.log(f"  Effective PC Buy Cost={effective_pc_buy_price:.2f}, Sell Revenue={effective_pc_sell_price:.2f}")

        # --- Access Exchange ---
        mm_order_depth = state.order_depths.get(MAGNIFICENT_MACARONS)
        if not mm_order_depth or (not mm_order_depth.buy_orders and not mm_order_depth.sell_orders):
            logger.log(f"WARN: No order depth or orders found for {MAGNIFICENT_MACARONS}")
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData
        exchange_best_bid, exchange_best_ask = self.get_best_bid_ask(mm_order_depth)
        mid_price = self.get_mid_price(mm_order_depth)
        logger.log(f"  Exchange MM: Best Bid={exchange_best_bid}, Best Ask={exchange_best_ask}, Mid={mid_price}")

        # --- Calculate Safe Limits ---
        max_additional_buy, max_additional_sell = self.calculate_safe_position_limits(MAGNIFICENT_MACARONS, current_pos_mm)
        logger.log(f"  Safe Position Change: Max Buy Vol={max_additional_buy}, Max Sell Vol={max_additional_sell}")

        # --- PRIORITY 1: Conversion Arbitrage ---
        # (Arbitrage logic remains unchanged - it should still take priority)
        # Opp 1: Buy PC -> Sell Exch
        if exchange_best_bid is not None:
            profit_per_unit = exchange_best_bid - effective_pc_buy_price
            required_profit = MIN_PROFIT_THRESHOLD + STORAGE_COST_BIAS_BUY
            if profit_per_unit > required_profit:
                bid_volume_available = mm_order_depth.buy_orders.get(exchange_best_bid, 0)
                trade_volume = min(CONVERSION_LIMIT_MM, max_additional_buy, bid_volume_available)
                if trade_volume > 0:
                    logger.log(f"    ACTION: Arb 1 Executing - Converting BUY from PC: {trade_volume}, Selling on Exch @ {exchange_best_bid}")
                    conversions = trade_volume
                    result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, exchange_best_bid, -trade_volume))
                    logger.flush(state, result, conversions, traderData)
                    return result, conversions, traderData

        # Opp 2: Buy Exch -> Sell PC
        if exchange_best_ask is not None and conversions == 0:
            profit_per_unit = effective_pc_sell_price - exchange_best_ask
            required_profit = MIN_PROFIT_THRESHOLD - STORAGE_COST_BIAS_SELL
            if profit_per_unit > required_profit and current_pos_mm > 0:
                ask_volume_available = abs(mm_order_depth.sell_orders.get(exchange_best_ask, 0))
                trade_volume = min(CONVERSION_LIMIT_MM, current_pos_mm, max_additional_buy, max_additional_sell, ask_volume_available)
                if trade_volume > 0:
                    logger.log(f"    ACTION: Arb 2 Executing - Converting SELL to PC: {trade_volume}, Buying on Exch @ {exchange_best_ask}")
                    conversions = -trade_volume
                    result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, exchange_best_ask, trade_volume))
                    logger.flush(state, result, conversions, traderData)
                    return result, conversions, traderData

        # --- PRIORITY 2: Hesitant Market Making (if no arb and enabled) ---
        if MM_HESITANT_ENABLE and conversions == 0 and not result[MAGNIFICENT_MACARONS]:
            logger.log("  Attempting Hesitant Market Making...")

            market_spread = float('inf')
            if exchange_best_bid is not None and exchange_best_ask is not None:
                 market_spread = exchange_best_ask - exchange_best_bid

            # Only proceed if mid_price exists and market spread isn't excessively wide
            if mid_price is not None and market_spread <= MAX_MARKET_SPREAD_FOR_MM:
                # Calculate sunlight adjustment
                # Positive adjustment means sun is LOW (prices expected high) -> push center UP
                # Negative adjustment means sun is HIGH (prices expected low) -> push center DOWN
                sunlight_deviation = NEUTRAL_SUNLIGHT - sunlight_index
                sunlight_adjustment = sunlight_deviation * SUNLIGHT_SENSITIVITY_FACTOR
                logger.log(f"  MM Sunlight: Index={sunlight_index:.2f}, Neutral={NEUTRAL_SUNLIGHT}, Deviation={sunlight_deviation:.2f}, Adjustment={sunlight_adjustment:.2f}")

                # Adjust the center price based on sunlight
                adjusted_center = mid_price + sunlight_adjustment
                logger.log(f"  MM Center Price: Mid={mid_price:.2f}, Adjusted Center={adjusted_center:.2f}")

                # Calculate wide, fixed spread quotes around the adjusted center
                mm_bid_price = math.floor(adjusted_center - MM_HESITANT_SPREAD / 2)
                mm_ask_price = math.ceil(adjusted_center + MM_HESITANT_SPREAD / 2)

                # Ensure bid < ask
                if mm_bid_price >= mm_ask_price:
                    mm_ask_price = mm_bid_price + 1
                    logger.log(f"  MM Price Correction: Bid >= Ask, adjusting Ask to {mm_ask_price}")

                logger.log(f"  MM Calculated Quotes: Bid={mm_bid_price}, Ask={mm_ask_price} (Spread={MM_HESITANT_SPREAD})")

                # Place small orders if within position limits
                if max_additional_buy > 0:
                    buy_volume = min(MM_HESITANT_SIZE, max_additional_buy)
                    # Optional: Don't place bid if it's >= best market ask
                    if exchange_best_ask is None or mm_bid_price < exchange_best_ask:
                         logger.log(f"    Placing MM Buy Order: {buy_volume} @ {mm_bid_price}")
                         result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, mm_bid_price, buy_volume))
                    else:
                         logger.log(f"    Skipping MM Buy Order: Calculated Bid ({mm_bid_price}) >= Best Ask ({exchange_best_ask})")


                if max_additional_sell > 0:
                    sell_volume = min(MM_HESITANT_SIZE, max_additional_sell)
                    # Optional: Don't place ask if it's <= best market bid
                    if exchange_best_bid is None or mm_ask_price > exchange_best_bid:
                        logger.log(f"    Placing MM Sell Order: {sell_volume} @ {mm_ask_price}")
                        result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, mm_ask_price, -sell_volume))
                    else:
                        logger.log(f"    Skipping MM Sell Order: Calculated Ask ({mm_ask_price}) <= Best Bid ({exchange_best_bid})")

            elif mid_price is None:
                logger.log("  Skipping MM: Mid-price not available.")
            else: # market_spread > MAX_MARKET_SPREAD_FOR_MM
                logger.log(f"  Skipping MM: Market spread ({market_spread}) > Max allowed ({MAX_MARKET_SPREAD_FOR_MM})")

        elif not MM_HESITANT_ENABLE:
             logger.log("  Hesitant Market Making is disabled.")
        elif conversions != 0:
             logger.log("  Skipping MM: Conversion arbitrage executed.")
        elif result[MAGNIFICENT_MACARONS]:
             logger.log("  Skipping MM: Conversion arbitrage orders placed.") # Should not happen if arb logic returns early

        # --- Final Logging and Return ---
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData