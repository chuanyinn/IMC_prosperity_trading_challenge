import json, jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List

PARAMS = {
    'AMETHYSTS': {
        'POSITION_LIMIT': 20,
        'MID': 10000,
        'TAKER_BID': 10000,
        'TAKER_ASK': 10000,
        'MAKER_BID': 9998,
        'MAKER_ASK': 10002,
    },
    'STARFRUIT': {
        'POSITION_LIMIT': 20,

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
    def take_orders_AMETHYSTS(self, product, state):
        orders = []

        order_depth  = state.order_depths[product]
        position = state.position.get(product, 0)

        for other_ask, volume in order_depth.sell_orders.items():
            logger.print("here", other_ask, volume)
            if ((other_ask < PARAMS[product]['TAKER_BID']) and
                (position <  PARAMS[product]['POSITION_LIMIT'])):
                order_size = min(-volume, PARAMS[product]['POSITION_LIMIT'] - position)
                logger.print(order_size)
                position += order_size
                assert(order_size >= 0)
                orders.append(Order(product, other_ask, order_size))

        position = state.position.get(product, 0)

        for other_bid, volume in order_depth.buy_orders.items():
            logger.print("here2", other_bid, volume)
            if ((other_bid > PARAMS[product]['TAKER_ASK']) and
                (position > -PARAMS[product]['POSITION_LIMIT'])):
                order_size = min(volume, PARAMS[product]['POSITION_LIMIT'] + position)
                logger.print(order_size)
                position -= order_size
                assert(order_size >= 0)
                orders.append(Order(product, other_bid, -order_size))

        return orders

    def make_orders_AMETHYSTS(self, product, state):
        # edit this now
        orders = []

        order_size_bid =  PARAMS[product]['POSITION_LIMIT'] - state.position.get(product, 0)
        order_size_ask = -PARAMS[product]['POSITION_LIMIT'] - state.position.get(product, 0)

        orders.append(Order(product, PARAMS[product]['MAKER_BID'], order_size_bid))
        orders.append(Order(product, PARAMS[product]['MAKER_ASK'], order_size_ask))

        return orders


    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}
        conversions = 0
        trader_data = ""
        orders = []

        for product in state.listings:
            if product == 'AMETHYSTS':
                result[product] = self.take_orders_AMETHYSTS(product, state)
                # result[product] = self.make_orders_AMETHYSTS(product, state)
                # if product not in state.position:
                #     state.position[product] = 0

                # order_size_bid =  PARAMS[product]['POSITION_LIMIT'] - state.position[product]
                # order_size_ask = -PARAMS[product]['POSITION_LIMIT'] - state.position[product]

                # orders.append(Order(product, PARAMS[product]['MAKER_BID'], order_size_bid))
                # orders.append(Order(product, PARAMS[product]['MAKER_ASK'], order_size_ask))

                # result[product] = orders

        # for product, order_depth in state.order_depths.items():
        #     if product == 'AMETHYSTS':
        #         orders = []
                
                # Market-taking
                # for 
                # taker_bid = 9998
                # taker_ask = 10002

                # if product in state.position:
                #     if state.position[product] > 0:
                #         order_size_bid = state.position[product]
                #         orders.append(Order(product, acceptable_bid, order_size_bid))
                #     elif state.position[product] < 0:
                #         acceptable_ask = 10002
                #         order_size_ask = -state.position[product]
                #         orders.append(Order(product, acceptable_ask, order_size_ask))


                # state.position[product]


        # Format the output
        # trader_data = jsonpickle.encode(trader_data_prev)
            
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data