import json, jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import numpy as np

PRODUCTS = [
    "AMETHYSTS",
    "STARFRUIT",
    # "ORCHIDS",
    # "GIFT_BASKET",
]

TRADER_DATA = {    
    'GIFT_BASKET':{        
        'PRODUCT': {'CHOCOLATE': 4, 'STRAWBERRIES': 6, 'ROSES': 1},
        'PROD_POS_LIMIT': {'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60},
        'POS_LIMIT': 60,
        'num_buy': 0,
        'num_sell': 0,
        'sell_limit': 6,
        'buy_limit': 6,

        'price_method': 'average',
        'expected_mid_price': None,
        'mid_price_data': [],
        'price_data_size': 1,

        # 'strategy': ['pair_trading', 'momentum'],
        'strategy': ['pair_trading', 'threshold'],
        'res_offset': 379.5,
        'trade_offset': 40,
        'place_make_order': False,
        'make_price_offset': [10, 10],
    },

    'ORCHIDS': {
        'POS_LIMIT': 100,
        'num_buy': 0,
        'num_sell': 0,

        # 'price_method': 'polyfit_foreign',
        # 'expected_mid_price': None,
        # 'mid_price_data': [],
        # 'foreign_price_data': [],
        # 'humidity_data': [],
        # 'sunlight_data': [],
        # 'price_data_size': 30,
        # 'polyfit_degree': 2,

        'price_method': 'foreign',
        'expected_mid_price': None,
        'mid_price_data': [],
        'foreign_price_data': [],
        'humidity_data': [],
        'sunlight_data': [],
        'price_data_size': 1,

        'strategy': ['cross_market_make'],
        'make_price_offset': [1, -2],
    },

    'AMETHYSTS': {
        'POS_LIMIT': 20,
        'num_buy': 0,
        'num_sell': 0,
        'strategy': ['market_take', 'market_make'],
        'price_method': 'static',
        'expected_mid_price': 10000,
        'take_position_stage': [0, 20],
        'take_price_spread': [(2, 2), (2, 0)],
        'make_position_stage': [0, 15, 20],
        'make_price_spread': [(1, 1), (1, 2), (0, 2)],
        'make_price_offset': [1, 1],
    },
    
    'STARFRUIT': {
        'POS_LIMIT': 20,
        'num_buy': 0,
        'num_sell': 0,
        'strategy': ['market_take', 'market_make'],

        # Data and parameters specific for mid_price calculation: method = "MA_1
        'price_method': 'MA_1',
        'expected_mid_price': None, 
        'MA_coef': -0.7086,
        'mid_price_data': [],
        'price_data_size': 1,

        # Data and parameters specific for mid_price calculation: method = "average"
        # 'price_method': 'average',
        # 'expected_mid_price': None,
        # 'mid_price_data': [],
        # 'price_data_size': 8,

        # Data and parameters specific for mid_price calculation: method = "regression"
        # 'price_method': 'regression',
        # 'expected_mid_price': None,
        # 'coef': [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892],
        # 'intercept': 4.481696494462085,
        # 'mid_price_data': [],
        # 'price_data_size': 4,

        'take_position_stage': [0, 20],
        'take_price_spread': [(1, 1), (1, 0)],
        'make_position_stage': [20],
        'make_price_spread': [(1, 1)],
        'make_price_offset': [1, 1],
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

    def calculateAverage(self, data: list[int], weights: list[float]|None, round_output=True) -> int:
        if weights:
            ans = sum([data[i] * weights[i] for i in range(len(data))])/sum(weights)
        else:
            ans = sum(data) / len(data)
        if round_output:
            return round(ans)
        return ans



    def getPriceSpread(self, position:int, stage:tuple[int], spread:tuple[tuple[int, int]]) -> tuple[int, int]:
        sign = 1 if position >= 0 else -1
        for i, pos in enumerate(stage):
            if abs(position) <= pos:
                return spread[i][::sign]



    def getBestBidAsk(self, state: TradingState, product: Symbol) -> tuple[int, int]:
        best_bid = list(state.order_depths[product].buy_orders.items())[0][0]
        best_ask = list(state.order_depths[product].sell_orders.items())[0][0]
        return best_bid, best_ask



    def updatePriceObervations(self, state: TradingState, product: Symbol, data: Dict[str, Any]) -> None:
        
        # Update mid_price_data
        if 'mid_price_data' in data:
            best_bid, best_ask = self.getBestBidAsk(state, product)
            mid_price = (best_ask + best_bid) / 2

            data['mid_price_data'].append(mid_price)
            if len(data['mid_price_data']) > data['price_data_size']:
                data['mid_price_data'].pop(0)

        # Update conversion observations data
        if product == 'ORCHIDS':
            conversion_observations = state.observations.conversionObservations[product]
            foreign_price = (conversion_observations.bidPrice + conversion_observations.askPrice) / 2

            data['humidity_data'].append(conversion_observations.humidity)
            data['sunlight_data'].append(conversion_observations.sunlight)
            data['foreign_price_data'].append(foreign_price)

            if len(data['humidity_data']) > data['price_data_size']:
                data['humidity_data'].pop(0)
                data['sunlight_data'].pop(0)            
                data['foreign_price_data'].pop(0)



    def updateData(self, state: TradingState, product: Symbol, data: Dict[str, Any]) -> None:
        data['num_buy'] = data['num_sell'] = 0
        
        if data['price_method'] == 'static':
            return

        self.updatePriceObervations(state, product, data)

        # Moving Average 1 model to predict mid_price
        if data['price_method'] == 'MA_1':      
            if data['expected_mid_price'] is None:
                data['expected_mid_price'] = data['mid_price_data'][-1]
            else:
                d = data['MA_coef'] * (data['mid_price_data'][-1] - data['expected_mid_price'])
                data['expected_mid_price'] = data['mid_price_data'][-1] + d


        # Simple Average model to predict mid_price
        elif data['price_method'] == 'average':
            if len(data['mid_price_data']) == data['price_data_size']:
                data['expected_mid_price'] = self.calculateAverage(data['mid_price_data'], None, round_output=False)


        # Regression model to predict mid_price
        elif data['price_method'] == 'regression':
            if len(data['mid_price_data']) == data['price_data_size']:
                data['expected_mid_price'] = sum([data['mid_price_data'][i] * data['coef'][i] for i in range(data['price_data_size'])]) + data['intercept']
        

        # Use foreign market mid price as mid_price
        elif data['price_method'] == 'foreign':
            conversion_observations = state.observations.conversionObservations[product]
            data['expected_mid_price'] = (conversion_observations.askPrice + conversion_observations.bidPrice) / 2


        # Average of foreign market price model to predict mid_price
        elif data['price_method'] == 'average_foreign':
            if len(data['foreign_price_data']) == data['price_data_size']:
                data['expected_mid_price'] = self.calculateAverage(data['foreign_price_data'], None, round_output=False)

        
        # Polynomial fit model to predict mid_price
        elif data['price_method'] == 'polyfit_foreign':
            if len(data['foreign_price_data']) == data['price_data_size']:
                data['expected_mid_price'] = np.polyval(np.polyfit(range(data['price_data_size']), data['foreign_price_data'], data['polyfit_degree']), data['price_data_size'])



    def computeTakeOrders(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:
        if data['expected_mid_price'] is None:
            return []

        orders = []
        buy_orders, sell_orders = state.order_depths[product].buy_orders.items(), state.order_depths[product].sell_orders.items()
        POS_LIMIT, position = data['POS_LIMIT'], state.position.get(product, 0)
        
        buy_offset, sell_offset = self.getPriceSpread(position + data['num_buy'] - data['num_sell'], 
                                                      data['take_position_stage'], 
                                                      data['take_price_spread'])
        
        acc_ask = data['expected_mid_price'] - buy_offset # price to buy, lower the better
        acc_bid = data['expected_mid_price'] + sell_offset # price to sell, higher the better

        for ask, ask_amount in sell_orders:
            if (ask <= acc_ask) and (data['num_buy'] < POS_LIMIT - position):
                buy_amount = min(-ask_amount, POS_LIMIT - position - data['num_buy'])
                orders.append(Order(product, ask, buy_amount))
                data['num_buy'] += buy_amount

        for bid, bid_amount in buy_orders:
            if (bid >= acc_bid) and (data['num_sell'] < POS_LIMIT + position):
                sell_amount = min(bid_amount, POS_LIMIT + position - data['num_sell'])
                orders.append(Order(product, bid, -sell_amount))
                data['num_sell'] += sell_amount

        return orders
    


    def computeMakeOrders(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:
        if data['expected_mid_price'] is None:
            return []
        
        orders = []
        best_bid, best_ask = self.getBestBidAsk(state, product)
        position = state.position.get(product, 0)

        buy_offset, sell_offset = self.getPriceSpread(position + data['num_buy'] - data['num_sell'], 
                                                      data['make_position_stage'], 
                                                      data['make_price_spread'])

        our_bid = min(best_bid + buy_offset, round(data['expected_mid_price'] - data['make_price_offset'][0]))
        our_ask = max(best_ask - sell_offset, round(data['expected_mid_price'] + data['make_price_offset'][1]))
        buy_amount = data['POS_LIMIT'] - position - data['num_buy']
        sell_amount = data['POS_LIMIT'] + position - data['num_sell']
        
        if buy_amount > 0:
            orders.append(Order(product, our_bid, buy_amount))
            data['num_buy'] += buy_amount

        if sell_amount > 0:
            orders.append(Order(product, our_ask, -sell_amount))
            data['num_sell'] += sell_amount
        
        return orders
    


    def computeCrossMarketMake(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:
        if data['expected_mid_price'] is None:
            return [], 0
        
        POS_LIMIT, position = data['POS_LIMIT'], state.position.get(product, 0)
        conversions = max(0, -position)
        # After conversion, position is 0
        
        orders = []
        observations = state.observations.conversionObservations[product]
        ask_price = observations.askPrice + observations.importTariff + observations.transportFees

        bid, bid_amount = list(state.order_depths[product].buy_orders.items())[0]

        sell_price = int(max(round(ask_price + data['make_price_offset'][0]), \
                             round(data['expected_mid_price'] + data['make_price_offset'][1])))
        
        # Market taker
        if bid > ask_price:
            sell_amount = min(POS_LIMIT, bid_amount)
            orders.append(Order(product, bid, -sell_amount))
            data['num_sell'] += sell_amount

        # Market maker
        sell_amount = POS_LIMIT - data['num_sell']
        orders.append(Order(product, sell_price, -sell_amount))

        return orders, conversions



    def computePairTrading(self, orders: Dict[Symbol, List[Order]], state: TradingState, data: Dict[str, Any]) -> None:
        # Get the current position of GIFT_BASKET
        position = state.position.get('GIFT_BASKET', 0)

        # Pair trading with individual products to hedge the GIFT_BASKET
        for p in data['PRODUCT']:
            prod_position, prod_pos_limit = state.position.get(p, 0), data['PROD_POS_LIMIT'][p]
            pos_diff  = prod_position + position * data['PRODUCT'][p]
            
            logger.print(f"Position difference for {p}: {pos_diff}")

            if pos_diff > 0: # We want to sell the individual products
                sell_amount = min(pos_diff, prod_pos_limit + prod_position)
                buy_orders = state.order_depths[p].buy_orders.items()
                for bid, bid_amount in buy_orders:
                    if sell_amount == 0:
                        break
                    sell_vol = min(sell_amount, bid_amount)
                    orders[p].append(Order(p, bid, -sell_vol))
                    sell_amount -= sell_vol

            elif pos_diff < 0: # We want to buy the individual products
                buy_amount = min(-pos_diff, prod_pos_limit - prod_position)
                sell_orders = state.order_depths[p].sell_orders.items()
                for ask, ask_amount in sell_orders:
                    if buy_amount == 0:
                        break
                    buy_vol = min(buy_amount, -ask_amount)
                    orders[p].append(Order(p, ask, buy_vol))
                    buy_amount -= buy_vol



    def computeThresholdOrders(self, orders: dict[Symbol, list[Order]], state: TradingState, data: Dict[str, Any]) -> None:
        POS_LIMIT, position = data['POS_LIMIT'], state.position.get('GIFT_BASKET', 0)
        best_bid, best_ask = self.getBestBidAsk(state, 'GIFT_BASKET')
        mid_price = (best_bid + best_ask) / 2

        res = data['res_offset']
        for p in data['PRODUCT']:
            bid, ask = self.getBestBidAsk(state, p)
            res += data['PRODUCT'][p] * (bid + ask) / 2
        
        if mid_price - res > data['trade_offset']:

            # Good to sell GIFT_BASKET
            buy_orders = state.order_depths['GIFT_BASKET'].buy_orders.items()
            sell_limit = data['sell_limit']

            for bid, bid_amount in buy_orders:
                if data['num_sell'] == min(POS_LIMIT + position, sell_limit):
                    break
                sell_vol = min([bid_amount, POS_LIMIT + position - data['num_sell'], sell_limit - data['num_sell']])
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', bid, -sell_vol))
                data['num_sell'] += sell_vol

            if data['num_sell'] < min(POS_LIMIT + position, sell_limit) and data['place_make_order']:
                sell_vol = min(position + POS_LIMIT, sell_limit) - data['num_sell']
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', round(mid_price + data['make_price_offset'][1]), -sell_vol))

        elif mid_price - res < -data['trade_offset']:

            # Good to buy GIFT_BASKET
            sell_orders = state.order_depths['GIFT_BASKET'].sell_orders.items()
            buy_limit = data['buy_limit']

            for ask, ask_amount in sell_orders:
                if data['num_buy'] == min(POS_LIMIT - position, buy_limit):
                    break
                buy_vol = min([-ask_amount, POS_LIMIT - position - data['num_buy'], buy_limit - data['num_buy']])
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', ask, buy_vol))
                data['num_buy'] += buy_vol

            if data['num_buy'] < min(POS_LIMIT - position, buy_limit) and data['place_make_order']:
                buy_vol = min(POS_LIMIT - position , buy_limit) - data['num_buy']
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', round(mid_price - data['make_price_offset'][0]), buy_vol))



    def computeMomentum(self, orders: dict[Symbol, list[Order]], state: TradingState, data: Dict[str, Any]) -> None:
        pass


    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]: 
        
        # Initialize returned variables
        result = {}
        conversions = 0
        
        # Initialize traderData in the first iteration
        trader_data = TRADER_DATA if state.traderData == "" else jsonpickle.decode(state.traderData)

        for product in PRODUCTS:

            # Update data with new information from market
            self.updateData(state, product, trader_data[product])

            if product != 'GIFT_BASKET':
                orders = []
                for strategy in trader_data[product]["strategy"]:
                    if strategy == "market_take":
                        orders += self.computeTakeOrders(product, state, trader_data[product])                        
                    elif strategy == "market_make":
                        orders += self.computeMakeOrders(product, state, trader_data[product])
                    elif strategy == "cross_market_make":
                        orders, conversions = self.computeCrossMarketMake(product, state, trader_data[product])

                result[product] = orders
            
            else:
                orders = {'GIFT_BASKET': [], 'CHOCOLATE': [], 'STRAWBERRIES': [], 'ROSES': []}
                for strategy in trader_data[product]['strategy']:
                    if strategy == 'threshold':
                        self.computeThresholdOrders(orders, state, trader_data[product])
                    elif strategy == 'momentum':
                        self.computeMomentum(orders, state, trader_data[product])
                    elif strategy == 'pair_trading':
                        self.computePairTrading(orders, state, trader_data[product])

                for p in orders:
                    result[p] = orders[p]

        # Format the output
        trader_data = jsonpickle.encode(trader_data)

        logger.flush(state, result, conversions, "")
        return result, conversions, trader_data