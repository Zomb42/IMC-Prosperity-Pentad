import json
from datamodel import (
    TradingState, OrderDepth, Order, Trade, Symbol, ProsperityEncoder,
    Listing, ConversionObservation, Observation, Product, Position
)
from typing import List, Dict, Tuple, Optional, Any
import math
import collections # Using collections for potential future use like deques

# Note: datamodel.py must be in the same directory or accessible in the Python path.

# --- Constants ---
MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"
POSITION_LIMIT_MM = 75
CONVERSION_LIMIT_MM = 10 # Max conversion request per run() call
STORAGE_COST_PER_UNIT_PER_TICK = 0.1 # Info, used for bias calculation
POSITION_BUFFER_PCT = 0.10 # Keep 10% buffer from limits (e.g., trade up to 90% of 75)
MIN_PROFIT_THRESHOLD = 0.5 # Minimum Seashells profit per unit to trigger conversion arbitrage
# --- Bias to account for storage cost (holding MM costs money) ---
# Make buying slightly harder (require more profit)
STORAGE_COST_BIAS_BUY = 0.2
# Make selling slightly easier (less profit required, as it avoids storage cost)
STORAGE_COST_BIAS_SELL = -0.1

# Basic logging setup
class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.verbose = True # Set to False to reduce log output size

    def log(self, message: str) -> None:
        if self.verbose:
            self.logs += message + "\n"

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        # This will print logs to stderr for the visualizer
        print(json.dumps({
            "logs": self.logs,
            # You can uncomment the lines below for more comprehensive debugging,
            # but be mindful of log size limits.
            # "state": state,
            # "orders": orders,
            # "conversions": conversions,
            # "trader_data": trader_data,
        }, cls=ProsperityEncoder, indent=2)) # Using indent=2 for slightly smaller logs
        self.logs = "" # Clear logs for the next iteration

logger = Logger()

class Trader:

    def __init__(self):
        # Example: Store past observations if needed for modeling
        # self.mm_observations_history = collections.deque(maxlen=100)
        logger.log("Initializing Trader")
        # No persistent state needed for this simple strategy yet
        pass

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

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        """Extracts the best bid and ask price from the order depth."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def calculate_safe_position_limits(self, symbol: str, current_position: int) -> Tuple[int, int]:
        """
        Calculate safe position limits with a buffer.
        Returns (max_additional_buy_volume, max_additional_sell_volume).
        These represent the max *change* in position allowed from the current state.
        """
        if symbol == MAGNIFICENT_MACARONS:
            max_pos = POSITION_LIMIT_MM
        else:
            # Define limits for other products if needed
            return 0, 0 # Default to no trading if limits unknown

        buffer = int(max_pos * POSITION_BUFFER_PCT)
        effective_max_limit = max_pos - buffer
        effective_min_limit = -max_pos + buffer

        # How many more units can we buy before hitting the effective max limit?
        max_buy = effective_max_limit - current_position
        # How many units can we sell before hitting the effective min limit?
        # This calculates the allowed *negative* volume (selling action)
        max_sell = current_position - effective_min_limit

        # Ensure non-negative results, as we return volumes
        safe_buy_vol = max(0, max_buy)
        safe_sell_vol = max(0, max_sell) # This is the volume we can sell (e.g., submit order with -qty)

        #logger.log(f"  Pos: {current_position}, Limits: [{effective_min_limit}, {effective_max_limit}], Safe Buy Vol: {safe_buy_vol}, Safe Sell Vol: {safe_sell_vol}")
        return safe_buy_vol, safe_sell_vol

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Takes TradingState and returns orders, conversions, and traderData.
        """
        result: Dict[str, List[Order]] = {MAGNIFICENT_MACARONS: []}
        conversions = 0
        traderData = state.traderData # Use the incoming traderData, ensure correct case

        # --- Log Timestamp and Position ---
        current_pos_mm = state.position.get(MAGNIFICENT_MACARONS, 0)
        logger.log(f"--- Timestamp: {state.timestamp}, Position MM: {current_pos_mm} ---")

        # --- TraderData Handling (Example: Load/Save) ---
        # If you need to store state across timestamps:
        # loaded_data = {}
        # if traderData:
        #     try:
        #         loaded_data = json.loads(traderData)
        #         # Use loaded_data...
        #         # self.mm_observations_history = collections.deque(loaded_data.get("mm_history", []), maxlen=100)
        #         logger.log("Loaded traderData successfully.")
        #     except json.JSONDecodeError:
        #         logger.log("Error decoding traderData JSON.")
        # # ... process ...
        # # Before returning, save state:
        # # save_data = {"mm_history": list(self.mm_observations_history)}
        # # traderData = json.dumps(save_data)


        # --- Access Observations (Crucial Check) ---
        if not state.observations or not state.observations.conversionObservations:
            logger.log("WARN: No conversion observations found in state.")
            # Return early before accessing potentially None objects
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData

        conv_obs = state.observations.conversionObservations.get(MAGNIFICENT_MACARONS)
        if not conv_obs:
            logger.log(f"WARN: No conversion observation found for {MAGNIFICENT_MACARONS}")
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData

        # Extract data using attribute names from datamodel.py
        transport_fees = conv_obs.transportFees
        import_tariff = conv_obs.importTariff
        export_tariff = conv_obs.exportTariff
        pc_bid = conv_obs.bidPrice # Price PC buys MM at (we SELL to PC)
        pc_ask = conv_obs.askPrice # Price PC sells MM at (we BUY from PC)
        sunlight_index = conv_obs.sunlightIndex # Available, not used in logic yet
        sugar_price = conv_obs.sugarPrice       # Available, not used in logic yet

        logger.log(f"  Observations: PC Bid={pc_bid}, PC Ask={pc_ask}, Transport={transport_fees}, ImportT={import_tariff}, ExportT={export_tariff}")
        logger.log(f"  Environment Factors: Sunlight Index={sunlight_index}, Sugar Price={sugar_price}")

        # --- Calculate Effective Conversion Prices ---
        # Cost to BUY 1 unit FROM PC (PC Ask + fees/tariffs)
        effective_pc_buy_price = pc_ask + transport_fees + import_tariff
        # Revenue to SELL 1 unit TO PC (PC Bid - fees/tariffs)
        effective_pc_sell_price = pc_bid - transport_fees - export_tariff

        logger.log(f"  Effective PC Buy Price (Cost): {effective_pc_buy_price:.2f}")
        logger.log(f"  Effective PC Sell Price (Revenue): {effective_pc_sell_price:.2f}")

        # --- Access Exchange Order Book ---
        mm_order_depth = state.order_depths.get(MAGNIFICENT_MACARONS)
        if not mm_order_depth or (not mm_order_depth.buy_orders and not mm_order_depth.sell_orders):
            logger.log(f"WARN: No order depth or orders found for {MAGNIFICENT_MACARONS}")
            logger.flush(state, result, conversions, traderData)
            return result, conversions, traderData

        exchange_best_bid, exchange_best_ask = self.get_best_bid_ask(mm_order_depth)
        logger.log(f"  Exchange MM: Best Bid={exchange_best_bid}, Best Ask={exchange_best_ask}")

        # --- Calculate Safe Position Change Limits ---
        # How much more can we buy / sell from the current position
        max_additional_buy, max_additional_sell = self.calculate_safe_position_limits(MAGNIFICENT_MACARONS, current_pos_mm)
        logger.log(f"  Safe Position Change: Max Buy Vol={max_additional_buy}, Max Sell Vol={max_additional_sell}")


        # --- Strategy Logic: Conversion Arbitrage ---
        # Note: Only one conversion request (positive or negative) can be processed per timestamp.
        # We prioritize the first profitable opportunity found.

        # --- Opportunity 1: Buy from PC (+Conversion), Sell on Exchange (-Order) ---
        # Check if selling on the exchange yields more than buying from PC costs.
        if exchange_best_bid is not None:
            profit_per_unit = exchange_best_bid - effective_pc_buy_price
            # Adjust threshold based on storage cost (buying from PC increases long pos)
            required_profit = MIN_PROFIT_THRESHOLD + STORAGE_COST_BIAS_BUY
            logger.log(f"  Potential Arb (Buy PC -> Sell Exch): Profit={profit_per_unit:.2f}, Required={required_profit:.2f}")

            if profit_per_unit > required_profit:
                 # Volume: limited by conversion limit, position limit (how much more we can buy/increase pos), and exchange bid depth
                bid_volume_available = mm_order_depth.buy_orders.get(exchange_best_bid, 0) # Positive volume

                # Max volume for this combined action:
                # - Cannot convert more than CONVERSION_LIMIT_MM
                # - Conversion increases position, limited by max_additional_buy
                # - Exchange sell order volume must be matched by conversion volume
                # - Exchange sell limited by available bid depth
                trade_volume = min(CONVERSION_LIMIT_MM, max_additional_buy, bid_volume_available)

                if trade_volume > 0:
                    logger.log(f"    ACTION: Converting BUY from PC: {trade_volume} units @ {pc_ask} (Effective Cost: {effective_pc_buy_price:.2f})")
                    logger.log(f"    ACTION: Placing SELL order on Exchange: {trade_volume} units @ {exchange_best_bid}")
                    # Positive conversion: buying from PC
                    conversions = trade_volume
                    # Place sell order on exchange
                    result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, exchange_best_bid, -trade_volume))
                    # Stop checking for other opportunities in this timestamp
                    logger.flush(state, result, conversions, traderData)
                    return result, conversions, traderData # Exit after action

        # --- Opportunity 2: Buy on Exchange (+Order), Sell to PC (-Conversion) ---
        # Check if selling to PC yields more than buying on the exchange costs.
        # Only proceed if Opportunity 1 didn't execute (conversions == 0)
        if exchange_best_ask is not None and conversions == 0:
            profit_per_unit = effective_pc_sell_price - exchange_best_ask
            # Adjust threshold based on storage cost (selling to PC decreases long pos / increases short)
            required_profit = MIN_PROFIT_THRESHOLD - STORAGE_COST_BIAS_SELL # Bias makes selling easier
            logger.log(f"  Potential Arb (Buy Exch -> Sell PC): Profit={profit_per_unit:.2f}, Required={required_profit:.2f}")

            # Condition: MUST have a positive position to sell to PC
            if profit_per_unit > required_profit and current_pos_mm > 0:
                # Volume: limited by conversion limit, CURRENT POSITION, position limits (how much more we can sell/decrease pos), and exchange ask depth
                ask_volume_available = abs(mm_order_depth.sell_orders.get(exchange_best_ask, 0)) # abs() because sell orders have negative qty

                # Max volume for this combined action:
                # - Cannot convert more than CONVERSION_LIMIT_MM
                # - Cannot convert (sell) more than current positive position (current_pos_mm)
                # - Exchange buy increases position, limited by max_additional_buy
                # - Exchange buy limited by available ask depth
                # - Conversion decreases position, limited by max_additional_sell
                trade_volume = min(CONVERSION_LIMIT_MM,
                                   current_pos_mm,       # Cannot sell more than we have
                                   max_additional_buy,   # Limit check for the exchange buy leg
                                   max_additional_sell,  # Limit check for the conversion sell leg (redundant if current_pos_mm checked?) Let's keep for safety.
                                   ask_volume_available)

                if trade_volume > 0:
                    logger.log(f"    ACTION: Converting SELL to PC: {trade_volume} units @ {pc_bid} (Effective Revenue: {effective_pc_sell_price:.2f})")
                    logger.log(f"    ACTION: Placing BUY order on Exchange: {trade_volume} units @ {exchange_best_ask}")
                    # Negative conversion: selling to PC
                    conversions = -trade_volume
                    # Place buy order on exchange
                    result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, exchange_best_ask, trade_volume))
                    # Stop checking for other opportunities in this timestamp
                    logger.flush(state, result, conversions, traderData)
                    return result, conversions, traderData # Exit after action

        # --- No conversion arbitrage opportunity found ---
        logger.log("  No conversion arbitrage opportunity found or executed.")

        # --- (Optional Placeholder: Basic Market Making if no arb) ---
        if conversions == 0 and not result[MAGNIFICENT_MACARONS]:
            # Example: Place orders around mid-price if liquid market
            mid_price = self.get_mid_price(mm_order_depth)
            if mid_price is not None and exchange_best_bid is not None and exchange_best_ask is not None:
                spread = max(2, exchange_best_ask - exchange_best_bid) # Adaptive spread?
                mm_buy_price = math.floor(mid_price - spread / 2)
                mm_sell_price = math.ceil(mid_price + spread / 2)
                mm_order_size = 5 # Small size
        
                # Place buy order if we have room
                if max_additional_buy > 0:
                    vol = min(mm_order_size, max_additional_buy)
                    logger.log(f"    Placing MM Buy Order: {vol} @ {mm_buy_price}")
                    result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, mm_buy_price, vol))
        
                # Place sell order if we have room
                if max_additional_sell > 0:
                    vol = min(mm_order_size, max_additional_sell)
                    logger.log(f"    Placing MM Sell Order: {vol} @ {mm_sell_price}")
                    result[MAGNIFICENT_MACARONS].append(Order(MAGNIFICENT_MACARONS, mm_sell_price, -vol))
         
        #--- End Optional MM ---


        # --- Final Logging and Return ---
        logger.flush(state, result, conversions, traderData)

        # Make sure to return the dictionary of orders, the conversion request int, and the traderData string
        return result, conversions, traderData