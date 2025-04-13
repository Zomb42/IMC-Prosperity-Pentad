import json
from r2.datamodel import OrderDepth, TradingState, Order, Trade, Symbol, ProsperityEncoder
from typing import List, Dict, Tuple, Optional
import math
import collections

# Define constants for readability and easy tuning
PRODUCTS = {
    "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
    "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2},
    "CROISSANTS": {},
    "JAMS": {},
    "DJEMBES": {}
}

POSITION_LIMITS = {
    "CROISSANTS": 250,
    "JAMS": 350,
    "DJEMBES": 60,
    "PICNIC_BASKET1": 60,
    "PICNIC_BASKET2": 100,
}

# Arbitrage profit margin threshold - more conservative
PROFIT_MARGIN_PER_BASKET = 10  # Increased from 8

# Maximum volume per arbitrage trade to reduce impact
MAX_ARB_TRADE_SIZE = 10  # Reduced from 15

# Position buffer percentage (keep this % away from limits)
POSITION_BUFFER_PCT = 0.15

# Safety factor for VWAP calculations
VWAP_SAFETY_FACTOR = 1.02  # Add 2% to component costs when buying

class Trader:

    def get_best_bid_ask(self, order_depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
        """Extracts the best bid and ask price from the order depth."""
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        return best_bid, best_ask

    def calculate_vwap(self, orders: Dict[int, int], volume_to_trade: int, is_buy_side: bool) -> Optional[float]:
        """
        Calculates the Volume Weighted Average Price for a given volume.
        Returns VWAP or None if the volume cannot be filled.
        orders: buy_orders if selling, sell_orders if buying
        volume_to_trade: positive integer
        is_buy_side: True if we are buying (hitting asks), False if selling (hitting bids)
        """
        if volume_to_trade <= 0:
            return None

        sorted_levels = sorted(orders.keys(), reverse=not is_buy_side) # Asks: low to high, Bids: high to low
        total_cost = 0
        volume_filled = 0

        for price in sorted_levels:
            available_volume = abs(orders[price])
            volume_at_level = min(available_volume, volume_to_trade - volume_filled)

            if volume_at_level > 0:
                total_cost += volume_at_level * price
                volume_filled += volume_at_level

            if volume_filled >= volume_to_trade:
                break

        if volume_filled < volume_to_trade:
            return None # Not enough liquidity

        return total_cost / volume_filled

    def calculate_theoretical_value_vwap(self, state: TradingState, basket_symbol: str, basket_trade_volume: int, use_asks_for_components: bool) -> Tuple[Optional[float], Dict[Symbol, Tuple[float, int]]]:
        """
        Calculates the theoretical value/cost of a basket based on its components using VWAP.
        use_asks_for_components: True if calculating cost to BUY components (hit asks),
                                 False if calculating value from SELLING components (hit bids).
        Returns the theoretical value and a dict of component VWAPs and quantities needed.
        """
        components = PRODUCTS.get(basket_symbol, {})
        if not components or basket_trade_volume <= 0:
            return None, {}

        total_theoretical_value = 0
        component_details = {}
        possible = True

        for product, quantity_per_basket in components.items():
            order_depth = state.order_depths.get(product)
            if not order_depth:
                possible = False
                break # Missing data for a component

            component_volume_needed = basket_trade_volume * quantity_per_basket
            component_vwap = None

            if use_asks_for_components: # Buying components -> Hit asks
                if order_depth.sell_orders:
                    component_vwap = self.calculate_vwap(order_depth.sell_orders, component_volume_needed, is_buy_side=True)
            else: # Selling components -> Hit bids
                if order_depth.buy_orders:
                     component_vwap = self.calculate_vwap(order_depth.buy_orders, component_volume_needed, is_buy_side=False)

            if component_vwap is None:
                possible = False # Cannot fill required component volume
                break

            # Apply safety factor when buying components
            if use_asks_for_components:
                component_vwap *= VWAP_SAFETY_FACTOR

            total_theoretical_value += component_vwap * quantity_per_basket
            component_details[product] = (component_vwap, quantity_per_basket) # Store VWAP and quantity per basket

        if not possible:
            return None, {}

        # Return total value/cost for the specified number of baskets
        return total_theoretical_value, component_details

    def get_max_order_size_at_best_price(self, order_depth: OrderDepth, best_price: Optional[int], is_buy_side: bool) -> int:
        """Gets the volume available at the single best price level."""
        if best_price is None:
            return 0

        orders = order_depth.sell_orders if is_buy_side else order_depth.buy_orders
        return abs(orders.get(best_price, 0))

    def calculate_safe_position_limits(self, symbol: str, current_position: int) -> Tuple[int, int]:
        """
        Calculate safe position limits with a buffer to avoid hitting limits.
        Returns (max_buy, max_sell) volumes that can be executed safely.
        """
        max_position = POSITION_LIMITS.get(symbol, 0)
        buffer = int(max_position * POSITION_BUFFER_PCT)
        
        # Reduce the effective limits by the buffer
        effective_max = max_position - buffer
        effective_min = -max_position + buffer
        
        # How much we can buy/sell from current position
        max_buy = effective_max - current_position
        max_sell = current_position - effective_min
        
        return max(0, max_buy), max(0, max_sell)

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Takes TradingState information and returns orders to be placed.
        """
        result: Dict[str, List[Order]] = {symbol: [] for symbol in PRODUCTS.keys()}
        traderData = "" # Optional state to pass to next round
        current_positions = state.position if state.position is not None else {}
        conversions = 0 # Not used in this strategy

        for basket_symbol in ["PICNIC_BASKET1", "PICNIC_BASKET2"]:
            basket_limit = POSITION_LIMITS[basket_symbol]
            basket_position = current_positions.get(basket_symbol, 0)
            basket_order_depth = state.order_depths.get(basket_symbol)
            components = PRODUCTS.get(basket_symbol, {})

            if not basket_order_depth or not components:
                continue # Skip if no basket data or components defined

            # Calculate safe position limits with buffer
            basket_max_buy, basket_max_sell = self.calculate_safe_position_limits(basket_symbol, basket_position)

            basket_best_bid, basket_best_ask = self.get_best_bid_ask(basket_order_depth)

            # --- Opportunity 1: Sell Basket, Buy Components ---
            if basket_best_bid is not None and basket_max_sell > 0:
                # Calculate potential max volume based on position limits first                
                component_max_vols_pos = []
                for product, quantity in components.items():
                    comp_position = current_positions.get(product, 0)
                    comp_max_buy, _ = self.calculate_safe_position_limits(product, comp_position)
                    
                    if quantity > 0 and comp_max_buy < quantity:
                        # Cannot even buy components for one basket
                        component_max_vols_pos.append(0)
                    elif quantity > 0:
                        # Maximum baskets we can create based on this component
                        component_max_vols_pos.append(math.floor(comp_max_buy / quantity))
                    else:
                        # No restriction from this component
                        component_max_vols_pos.append(float('inf'))

                # Overall max volume limited by positions
                max_vol_pos_limit = min([basket_max_sell] + component_max_vols_pos)
                
                # Limit trade size for market impact reasons
                potential_trade_volume = min(max_vol_pos_limit, MAX_ARB_TRADE_SIZE)

                if potential_trade_volume > 0:
                    # Calculate realistic component cost using VWAP for this potential volume
                    component_buy_cost_total, _ = self.calculate_theoretical_value_vwap(
                        state, basket_symbol, potential_trade_volume, use_asks_for_components=True)

                    if component_buy_cost_total is not None:
                        component_buy_cost_per_basket = component_buy_cost_total / potential_trade_volume
                        # Check profitability using the actual basket bid price
                        if basket_best_bid > component_buy_cost_per_basket + PROFIT_MARGIN_PER_BASKET:
                            # Now check liquidity at BEST prices (simpler check for execution volume)
                            vol_at_basket_bid = self.get_max_order_size_at_best_price(basket_order_depth, basket_best_bid, False)
                            
                            comp_liquidity_limits = []
                            for product, quantity in components.items():
                                comp_order_depth = state.order_depths.get(product)
                                if not comp_order_depth:
                                    comp_liquidity_limits.append(0)
                                    continue
                                    
                                _, comp_best_ask = self.get_best_bid_ask(comp_order_depth)
                                vol_at_comp_ask = self.get_max_order_size_at_best_price(comp_order_depth, comp_best_ask, True)
                                if quantity > 0 and vol_at_comp_ask < quantity:
                                    comp_liquidity_limits.append(0) # Cannot even fill 1 basket's worth
                                elif quantity > 0:
                                    comp_liquidity_limits.append(math.floor(vol_at_comp_ask / quantity))
                                else:
                                     comp_liquidity_limits.append(float('inf'))

                            max_vol_liquidity = min([vol_at_basket_bid] + comp_liquidity_limits)
                            
                            # Final trade volume is minimum of position limit, impact limit, and liquidity limit
                            final_trade_volume = min(potential_trade_volume, max_vol_liquidity)

                            if final_trade_volume > 0:
                                print(f"TRADE OPPORTUNITY (Sell {basket_symbol}): Est Profit/Basket={(basket_best_bid - component_buy_cost_per_basket):.2f}, Vol={final_trade_volume}")
                                # Place orders at best prices
                                result[basket_symbol].append(Order(basket_symbol, basket_best_bid, -final_trade_volume))
                                for product, quantity in components.items():
                                    comp_order_depth = state.order_depths.get(product, None)
                                    if comp_order_depth:
                                        _, comp_best_ask = self.get_best_bid_ask(comp_order_depth)
                                        if comp_best_ask is not None:
                                            # Ensure product exists in result before appending
                                            if product not in result: result[product] = []
                                            result[product].append(Order(product, comp_best_ask, final_trade_volume * quantity))


            # --- Opportunity 2: Buy Basket, Sell Components ---
            if basket_best_ask is not None and basket_max_buy > 0:
                 # Calculate potential max volume based on position limits first
                component_max_vols_pos = []
                for product, quantity in components.items():
                    comp_position = current_positions.get(product, 0)
                    _, comp_max_sell = self.calculate_safe_position_limits(product, comp_position)
                    
                    if quantity > 0 and comp_max_sell < quantity:
                        # Cannot even sell components for one basket
                        component_max_vols_pos.append(0)
                    elif quantity > 0:
                        # Maximum baskets we can process based on this component
                        component_max_vols_pos.append(math.floor(comp_max_sell / quantity))
                    else:
                        # No restriction from this component
                        component_max_vols_pos.append(float('inf'))

                # Overall max volume limited by positions
                max_vol_pos_limit = min([basket_max_buy] + component_max_vols_pos)
                
                # Limit trade size for market impact reasons
                potential_trade_volume = min(max_vol_pos_limit, MAX_ARB_TRADE_SIZE)

                if potential_trade_volume > 0:
                    # Calculate realistic component value using VWAP for this potential volume
                    component_sell_value_total, _ = self.calculate_theoretical_value_vwap(
                        state, basket_symbol, potential_trade_volume, use_asks_for_components=False)

                    if component_sell_value_total is not None:
                        component_sell_value_per_basket = component_sell_value_total / potential_trade_volume
                        # Check profitability using the actual basket ask price
                        if component_sell_value_per_basket > basket_best_ask + PROFIT_MARGIN_PER_BASKET:
                            # Now check liquidity at BEST prices
                            vol_at_basket_ask = self.get_max_order_size_at_best_price(basket_order_depth, basket_best_ask, True)

                            comp_liquidity_limits = []
                            for product, quantity in components.items():
                                comp_order_depth = state.order_depths.get(product)
                                if not comp_order_depth:
                                    comp_liquidity_limits.append(0)
                                    continue
                                    
                                comp_best_bid, _ = self.get_best_bid_ask(comp_order_depth)
                                vol_at_comp_bid = self.get_max_order_size_at_best_price(comp_order_depth, comp_best_bid, False)
                                if quantity > 0 and vol_at_comp_bid < quantity:
                                     comp_liquidity_limits.append(0) # Cannot even fill 1 basket's worth
                                elif quantity > 0:
                                    comp_liquidity_limits.append(math.floor(vol_at_comp_bid / quantity))
                                else:
                                     comp_liquidity_limits.append(float('inf'))

                            max_vol_liquidity = min([vol_at_basket_ask] + comp_liquidity_limits)

                            # Final trade volume
                            final_trade_volume = min(potential_trade_volume, max_vol_liquidity)

                            if final_trade_volume > 0:
                                print(f"TRADE OPPORTUNITY (Buy {basket_symbol}): Est Profit/Basket={(component_sell_value_per_basket - basket_best_ask):.2f}, Vol={final_trade_volume}")
                                # Place orders at best prices
                                result[basket_symbol].append(Order(basket_symbol, basket_best_ask, final_trade_volume))
                                for product, quantity in components.items():
                                    comp_order_depth = state.order_depths.get(product, None)
                                    if comp_order_depth:
                                        comp_best_bid, _ = self.get_best_bid_ask(comp_order_depth)
                                        if comp_best_bid is not None:
                                            # Ensure product exists in result before appending
                                            if product not in result: result[product] = []
                                            result[product].append(Order(product, comp_best_bid, -final_trade_volume * quantity))


        # Clean up result: remove products with no orders
        final_result = {symbol: orders for symbol, orders in result.items() if orders}

        return final_result, conversions, traderData