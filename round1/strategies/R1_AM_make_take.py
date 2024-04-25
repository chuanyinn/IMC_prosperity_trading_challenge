import json, jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
from collections import OrderedDict

TRADER_DATA = {
    'AMETHYSTS': {
        'POSITION_LIMIT': 20,
        'NUM_BID': 0, # amount buy
        'NUM_ASK': 0, # amount sell
        'MID_PRICE': 10000,

        'METHOD': 'static',

        'TAKE_THRESHOLD': [0, 20], # absolute position thresholds, as market taker
        'TAKE_SPREAD_BA': [(2, 2), (2, 0)], # (bid, ask) offset from mid

        'MAKE_THRESHOLD': [0, 15, 20], # absolute position thresholds, as market maker
        'MAKE_SPREAD_BA': [(2, 2), (2, 1), (2, 0)], # (bid, ask) offset from mid
        'MAKE_MIN_RANGE': [1, 1],
    },
    'STARFRUIT': {
        'POSITION_LIMIT': 20,

        'METHOD': 'moving_average',

        'MID': None,
        'BID': None,
        'ASK': None,
    },
}
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()

class Trader:
    def get_spread(self, position: int, stage: tuple[int], spread:tuple[tuple[int, int]]) -> tuple[int, int]:
        for i, pos in enumerate(stage):
            if abs(position) <= pos:
                if position >= 0:
                    return spread[i]
                else:
                    return spread[i][::-1]
                

    def update_trader_data(self, state: TradingState, trader_data: Dict[Symbol, Dict[str, Any]]) -> None:
        for product in trader_data:
            if product == 'AMETHYSTS':

                prod_data = trader_data[product]
                prod_data['NUM_BID'] = prod_data['NUM_ASK'] = 0

                if prod_data['METHOD'] == 'static':
                    continue


    def take_orders(self, product: Symbol, state: TradingState, prod_data: Dict[str, Any]):
        if prod_data['MID_PRICE'] is None:
            return []
        
        orders = []

        # other_bid_orders = state.order_depths[product].buy_orders.items()
        # other_ask_orders = state.order_depths[product].sell_orders.items()

        other_bid_orders = OrderedDict(sorted(
                    state.order_depths[product].buy_orders.items(), reverse=True)).items()
        other_ask_orders = OrderedDict(sorted(
                    state.order_depths[product].sell_orders.items())).items()

        position = state.position.get(product, 0)

        bid_offset, ask_offset = self.get_spread(position + prod_data['NUM_BID'] - prod_data['NUM_ASK'], 
                                                 prod_data['TAKE_THRESHOLD'],
                                                 prod_data['TAKE_SPREAD_BA'])
        
        acc_other_ask = prod_data['MID_PRICE'] - bid_offset # price for us to buy
        acc_other_bid = prod_data['MID_PRICE'] + ask_offset # price for us to sell

        for other_ask, other_ask_qty in other_ask_orders:
            if (other_ask <= acc_other_ask) and (position + prod_data['NUM_BID'] < prod_data['POSITION_LIMIT']):
                bid_quantity = min(-other_ask_qty, prod_data['POSITION_LIMIT'] - position - prod_data['NUM_BID'])
                orders.append(Order(product, other_ask, bid_quantity))
                prod_data['NUM_BID'] += bid_quantity

        for other_bid, other_bid_qty in other_bid_orders:
            if (other_bid >= acc_other_bid) and (position - prod_data['NUM_ASK'] > -prod_data['POSITION_LIMIT']):
                ask_quantity = min(other_bid_qty, prod_data['POSITION_LIMIT'] + position - prod_data['NUM_ASK'])
                orders.append(Order(product, other_bid, -ask_quantity))
                prod_data['NUM_ASK'] += ask_quantity

        return orders
        
    
    def make_orders(self, product: Symbol, state: TradingState, prod_data: Dict[str, Any]):
        if prod_data['MID_PRICE'] is None:
            return []
        
        orders = []

        best_other_bid = list(state.order_depths[product].buy_orders.items())[0][0]
        best_other_ask = list(state.order_depths[product].sell_orders.items())[0][0]

        position = state.position.get(product, 0)

        bid_offset, ask_offset = self.get_spread(position + prod_data['NUM_BID'] - prod_data['NUM_ASK'], 
                                                 prod_data['MAKE_THRESHOLD'],
                                                 prod_data['MAKE_SPREAD_BA'])
        
        bid_price = min(best_other_bid + bid_offset, 
                        prod_data['MID_PRICE'] - prod_data['MAKE_MIN_RANGE'][0])
        ask_price = max(best_other_ask - ask_offset, 
                        prod_data['MID_PRICE'] + prod_data['MAKE_MIN_RANGE'][1])
        
        bid_quantity = prod_data['POSITION_LIMIT'] - position - prod_data['NUM_BID']
        ask_quantity = prod_data['POSITION_LIMIT'] + position - prod_data['NUM_ASK']

        if bid_quantity > 0:
            orders.append(Order(product, bid_price, bid_quantity))
            prod_data['NUM_BID'] += bid_quantity
        if ask_quantity > 0:
            orders.append(Order(product, ask_price, -ask_quantity))
            prod_data['NUM_ASK'] += ask_quantity

        return orders
    

    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
    
        # Initialize traderData in the first iteration
        if state.traderData == "":
            trader_data = TRADER_DATA
        else:
            trader_data = jsonpickle.decode(state.traderData)

        self.update_trader_data(state, trader_data)

        for product in state.listings:
            orders = []

            if product == 'AMETHYSTS':
                orders += self.take_orders(product, state, trader_data[product])
                orders += self.make_orders(product, state, trader_data[product])

            result[product] = orders

        # Format the output
        trader_data = jsonpickle.encode(trader_data)
            
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data