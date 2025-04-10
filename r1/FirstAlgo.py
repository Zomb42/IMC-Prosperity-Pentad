from r1.datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import statistics
import jsonpickle # For saving/loading state
import math

# Set USE_JSONPICKLE to False if you encounter issues with it saving/loading state
# Set it to True to enable saving price history between runs
USE_JSONPICKLE = True

class Trader:
    def __init__(self):
        # Initialize price history for each product
        self.product_state: Dict[str, Dict[str, Any]] = {
            "RAINFOREST_RESIN": {"price_history": [], "long_ma": None},
            "KELP": {"price_history": [], "long_ma": None},
            "SQUID_INK": {"price_history": [], "short_ma": None, "long_ma": None}
        }

        # Position tracking (will be updated from state)
        self.positions = {product: 0 for product in self.product_state}

        # Position limits (adjust if needed)
        self.position_limits = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }

        # Parameters for trading strategies
        self.resin_base_spread = 2
        self.resin_skew_intensity = 1
        self.resin_max_order_size = 10

        self.kelp_ma_window = 20
        self.kelp_threshold_pct = 0.002
        self.kelp_threshold_abs = 3 # <- Increased from 2 to 3
        self.kelp_order_size = 8

        self.squid_short_window = 5
        self.squid_long_window = 40
        self.squid_volatility_threshold = 3
        self.squid_order_size = 5

        self.history_max_length = 100

        print("Trader Initialized with parameters:")
        print(f"  RAINFOREST_RESIN: base_spread={self.resin_base_spread}, skew_intensity={self.resin_skew_intensity}, max_order_size={self.resin_max_order_size}")
        print(f"  KELP: ma_window={self.kelp_ma_window}, threshold_pct={self.kelp_threshold_pct}, threshold_abs={self.kelp_threshold_abs}, order_size={self.kelp_order_size}") # Updated print
        print(f"  SQUID_INK: short_window={self.squid_short_window}, long_window={self.squid_long_window}, vol_threshold={self.squid_volatility_threshold}, order_size={self.squid_order_size}")


    def load_state(self, traderData: str):
        """Load price history and potentially other state from traderData."""
        if USE_JSONPICKLE and traderData:
            try:
                loaded_state = jsonpickle.decode(traderData)
                if isinstance(loaded_state, dict):
                     for product, data in loaded_state.items():
                         if product in self.product_state and isinstance(data, dict): # Check data is dict
                             for key, value in data.items():
                                 if key in self.product_state[product]:
                                     # Basic type check for history before assigning
                                     if key == 'price_history' and isinstance(value, list):
                                        self.product_state[product][key] = value
                                     elif key != 'price_history': # Load other keys if needed
                                        self.product_state[product][key] = value
                     print("Successfully loaded state from traderData.")
                else:
                     print("Warning: traderData format unrecognized, using default state.")

            except Exception as e:
                print(f"Error loading traderData: {e}. Using default state.")
        else:
            # This is expected on the first run or if JSONPICKLE is False
            if traderData:
                 print("Warning: traderData found but JSONPICKLE is False or disabled.")
            else:
                 print("No previous state found or JSONPICKLE disabled.")


    def save_state(self) -> str:
        """Save relevant state (like price history) to traderData string."""
        if USE_JSONPICKLE:
            try:
                 # Only save parts that need persisting (e.g., history)
                 state_to_save = {
                     product: {'price_history': data['price_history']}
                     for product, data in self.product_state.items()
                 }
                 return jsonpickle.encode(state_to_save)
            except Exception as e:
                print(f"Error saving state: {e}")
                return ""
        else:
            return ""

    def get_mid_price(self, product: str, order_depth: OrderDepth) -> float | None:
        """Calculate the mid price from order book, returns None if invalid."""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            #print(f"Warning: No buy or sell orders for {product} to calculate mid price.")
            return None

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        if best_bid >= best_ask:
            #print(f"Warning: Best bid ({best_bid}) >= best ask ({best_ask}) for {product}. Spread crossed.")
            return None

        return (best_bid + best_ask) / 2

    def update_price_history(self, product: str, mid_price: float | None):
        """Update price history for a product if mid_price is valid."""
        if mid_price is not None:
            history = self.product_state[product].get("price_history", []) # Safely get history
            history.append(mid_price)
            # Keep history length manageable
            if len(history) > self.history_max_length:
                self.product_state[product]["price_history"] = history[-self.history_max_length:]
            else:
                 self.product_state[product]["price_history"] = history


    def trade_rainforest_resin(self, order_depth: OrderDepth) -> List[Order]:
        """Market making strategy for Rainforest Resin with inventory skewing."""
        orders = []
        product = "RAINFOREST_RESIN"
        current_pos = self.positions[product]
        pos_limit = self.position_limits[product]

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        market_spread = best_ask - best_bid

        if market_spread <= 0:
            return orders

        skew_factor = current_pos / pos_limit if pos_limit != 0 else 0
        price_adjustment = int(round(skew_factor * self.resin_skew_intensity))
        spread_multiplier = 1.0 + abs(skew_factor) * 0.5
        effective_spread = max(1, int(round(self.resin_base_spread * spread_multiplier)))

        reference_price = (best_bid + best_ask) / 2
        desired_bid = int(math.floor(reference_price - effective_spread / 2 - price_adjustment))
        desired_ask = int(math.ceil(reference_price + effective_spread / 2 - price_adjustment))

        final_bid = min(desired_bid, best_bid + 1)
        final_ask = max(desired_ask, best_ask - 1)

        if final_bid >= final_ask:
             # Fallback if quotes cross after skew/spread adjustment
            market_mid = (best_bid + best_ask) / 2
            final_bid = int(math.floor(market_mid - self.resin_base_spread)) # Use base spread for fallback
            final_ask = int(math.ceil(market_mid + self.resin_base_spread))
            if final_bid >= final_ask: # Still crossed? Don't quote.
                print(f"  RESIN Debug: Fallback quotes crossed ({final_bid}>={final_ask}), skipping orders.")
                return orders

        print(f"  RESIN Debug: Pos={current_pos}, Skew={skew_factor:.2f}, EffSpread={effective_spread}, PxAdj={price_adjustment}, DesiredBid={desired_bid}, DesiredAsk={desired_ask}, FinalBid={final_bid}, FinalAsk={final_ask}")

        buy_capacity = pos_limit - current_pos
        sell_capacity = pos_limit + current_pos

        buy_order_size = min(self.resin_max_order_size, buy_capacity)
        sell_order_size = min(self.resin_max_order_size, sell_capacity)

        if buy_order_size > 0:
            orders.append(Order(product, final_bid, buy_order_size))
            #print(f"  RESIN: Placing BUY {buy_order_size} @ {final_bid} (pos={current_pos}, skew={skew_factor:.2f})")

        if sell_order_size > 0:
            orders.append(Order(product, final_ask, -sell_order_size))
            #print(f"  RESIN: Placing SELL {sell_order_size} @ {final_ask} (pos={current_pos}, skew={skew_factor:.2f})")

        return orders


    def trade_kelp(self, order_depth: OrderDepth) -> List[Order]:
        """Passive Moving Average strategy for Kelp."""
        orders = []
        product = "KELP"
        history = self.product_state[product].get("price_history", [])
        pos_limit = self.position_limits[product]
        current_pos = self.positions[product]

        if len(history) < self.kelp_ma_window:
            #print(f"  KELP: Not enough history ({len(history)}/{self.kelp_ma_window})")
            return orders

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        mid_price = (best_bid + best_ask) / 2

        if mid_price is None or best_ask <= best_bid:
            return orders

        # Calculate moving average
        recent_avg = statistics.mean(history[-self.kelp_ma_window:])
        self.product_state[product]["long_ma"] = recent_avg

        # Calculate deviation thresholds
        threshold = max(recent_avg * self.kelp_threshold_pct, self.kelp_threshold_abs)
        buy_threshold_price = recent_avg - threshold
        sell_threshold_price = recent_avg + threshold

        print(f"  KELP Debug: Mid={mid_price:.2f}, MA={recent_avg:.2f}, BuyThr={buy_threshold_price:.2f}, SellThr={sell_threshold_price:.2f}, AbsThreshold={self.kelp_threshold_abs}")

        # Calculate available capacity
        buy_capacity = pos_limit - current_pos
        sell_capacity = pos_limit + current_pos

        # --- Passive Limit Order Logic ---
        if mid_price < buy_threshold_price:
            print(f"  KELP Debug: Buy condition met (mid < buy_thr).")
            if buy_capacity > 0:
                buy_price = best_bid + 1
                # Sanity check: Don't place buy order above the MA sell threshold.
                if buy_price < sell_threshold_price:
                    order_size = min(self.kelp_order_size, buy_capacity)
                    orders.append(Order(product, buy_price, order_size))
                    print(f"  KELP Placing BUY Order: {order_size} @ {buy_price}")
                else:
                    print(f"  KELP Skipping BUY: Order price {buy_price} >= MA Sell Threshold {sell_threshold_price:.2f}")

        elif mid_price > sell_threshold_price:
            print(f"  KELP Debug: Sell condition met (mid > sell_thr).")
            if sell_capacity > 0:
                sell_price = best_ask - 1
                 # Sanity check: Don't place sell order below the MA buy threshold.
                if sell_price > buy_threshold_price:
                    order_size = min(self.kelp_order_size, sell_capacity)
                    orders.append(Order(product, sell_price, -order_size))
                    print(f"  KELP Placing SELL Order: {order_size} @ {sell_price}")
                else:
                     print(f"  KELP Skipping SELL: Order price {sell_price} <= MA Buy Threshold {buy_threshold_price:.2f}")

        return orders


    def trade_squid_ink(self, order_depth: OrderDepth) -> List[Order]:
        """Passive Mean Reversion strategy for Squid Ink using rolling MAs."""
        orders = []
        product = "SQUID_INK"
        history = self.product_state[product].get("price_history", [])
        pos_limit = self.position_limits[product]
        current_pos = self.positions[product]

        if len(history) < self.squid_long_window:
            #print(f"  SQUID: Not enough history ({len(history)}/{self.squid_long_window})")
            return orders

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        if best_ask <= best_bid:
             return orders

        # Calculate short and long term rolling averages
        short_term_avg = statistics.mean(history[-self.squid_short_window:])
        long_term_avg = statistics.mean(history[-self.squid_long_window:])
        self.product_state[product]["short_ma"] = short_term_avg
        self.product_state[product]["long_ma"] = long_term_avg

        deviation = short_term_avg - long_term_avg
        abs_deviation = abs(deviation)

        print(f"  SQUID Debug: ShortMA={short_term_avg:.2f}, LongMA={long_term_avg:.2f}, Deviation={deviation:.2f}, AbsDev={abs_deviation:.2f}, Threshold={self.squid_volatility_threshold}")

        buy_capacity = pos_limit - current_pos
        sell_capacity = pos_limit + current_pos

        # --- Passive Limit Order Logic based on Mean Reversion ---
        if abs_deviation > self.squid_volatility_threshold:
            # Expect reversion UP -> BUY
            if deviation < 0:
                print(f"  SQUID Debug: Buy condition met (dev < -threshold).")
                if buy_capacity > 0:
                    buy_price = best_bid + 1
                    # Optional check: Only buy if the target price is below the long MA? (Could prevent chasing)
                    # if buy_price < long_term_avg:
                    order_size = min(self.squid_order_size, buy_capacity)
                    orders.append(Order(product, buy_price, order_size))
                    print(f"  SQUID Placing BUY Order: {order_size} @ {buy_price}")
                    #else:
                    #    print(f"  SQUID Skipping BUY: Order price {buy_price} >= LongMA {long_term_avg:.2f}")

            # Expect reversion DOWN -> SELL
            elif deviation > 0:
                print(f"  SQUID Debug: Sell condition met (dev > threshold).")
                if sell_capacity > 0:
                    sell_price = best_ask - 1
                    # Optional check: Only sell if the target price is above the long MA? (Could prevent chasing)
                    # if sell_price > long_term_avg:
                    order_size = min(self.squid_order_size, sell_capacity)
                    orders.append(Order(product, sell_price, -order_size))
                    print(f"  SQUID Placing SELL Order: {order_size} @ {sell_price}")
                    # else:
                    #    print(f"  SQUID Skipping SELL: Order price {sell_price} <= LongMA {long_term_avg:.2f}")

        return orders

    def update_positions(self, state: TradingState):
        """Update internal position tracking based on the provided state."""
        print("  Updating Positions:")
        for product in self.product_state:
            self.positions[product] = state.position.get(product, 0)
            print(f"    {product}: {self.positions[product]}")


    def run(self, state: TradingState):
        """Main method required by the competition."""
        print(f"\n----- Timestamp: {state.timestamp} -----")

        # 1. Load previous state (if any)
        self.load_state(state.traderData)

        # 2. Update current positions from the latest state
        self.update_positions(state)

        # 3. Initialize results for this timestamp
        result = {}
        conversions = 0

        # 4. Process each product
        for product, order_depth in state.order_depths.items():
            if product not in self.product_state:
                 print(f"Warning: Received data for unknown product {product}")
                 continue

            orders: list[Order] = []

            # 4a. Calculate and update price history
            mid_price = self.get_mid_price(product, order_depth)
            self.update_price_history(product, mid_price)
            print(f"  Mid Price {product}: {mid_price if mid_price is not None else 'N/A'}")

            # 4b. Apply product-specific trading strategy
            try:
                if product == "RAINFOREST_RESIN":
                    orders.extend(self.trade_rainforest_resin(order_depth))
                elif product == "KELP":
                    orders.extend(self.trade_kelp(order_depth))
                elif product == "SQUID_INK":
                    orders.extend(self.trade_squid_ink(order_depth))

            except Exception as e:
                import traceback
                print(f"!!! Error running strategy for {product}: {e}")
                traceback.print_exc() # Print full traceback for detailed debugging


            # 4c. Store generated orders for the product
            if orders:
                result[product] = orders
                # Print orders for debugging
                for order in orders:
                    print(f"  >>> Placing Order: {order.symbol} {'BUY' if order.quantity > 0 else 'SELL'} {abs(order.quantity)} @ {order.price}")


        # 5. Save state for the next iteration
        traderData = self.save_state()

        print(f"----- End Timestamp: {state.timestamp} -----")

        return result, conversions, traderData

# --- END OF FILE FirstAlgo.py ---