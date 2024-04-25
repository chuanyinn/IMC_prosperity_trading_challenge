import json, jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import numpy as np

PRODUCTS = [
    "AMETHYSTS",
    "STARFRUIT",
    "ORCHIDS",
]

TRADER_DATA = {
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
        'make_price_spread': [(2, 2), (2, 1), (2, 0)],
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

    'ORCHIDS': {

        'POS_LIMIT': 100,
        'num_buy': 0,
        'num_sell': 0,

        # 'price_method': 'multivar_trend',
        # 'expected_mid_price': None,
        # 'expected_trend': None, # Unit is price move per iteration
        # 'expected_trend_confidence': None,
        # 'price_data_size': 8,
        # 'mid_price_data': [],
        # 'foreign_price_data': [], # Mid price data from foreign market
        # 'humidity_data': [],
        # 'sunlight_data': [],

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
        'make_spread': 2,

        # 'strategy': ['cross_market'],
        # 'spread': [5, 2],
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
        

        # Use foreign market ask price as mid_price
        elif data['price_method'] == 'foreign':
            conversion_observations = state.observations.conversionObservations[product]
            data['expected_mid_price'] = conversion_observations.askPrice + conversion_observations.importTariff + conversion_observations.transportFees


        # Average of foreign market price model to predict mid_price
        elif data['price_method'] == 'average_foreign':
            if len(data['foreign_price_data']) == data['price_data_size']:
                data['expected_mid_price'] = self.calculateAverage(data['foreign_price_data'], None, round_output=False)

        
        # Polynomial fit model to predict mid_price
        elif data['price_method'] == 'polyfit_foreign':
            if len(data['foreign_price_data']) == data['price_data_size']:
                data['expected_mid_price'] = np.polyval(np.polyfit(range(data['price_data_size']), data['foreign_price_data'], data['polyfit_degree']), data['price_data_size'])


        # Multivariate trend model to predict mid_price
        elif data['price_method'] == 'multivar_trend':
            if len(data['mid_price_data']) == data['price_data_size']:
                pass # TODO: Implement the model here



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
    


    def computeCrossMarket(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:

        def computeBuySell(expected_mid_price, s, ask, ask_amount, bid, bid_amount, ask_foreign, import_fee, POS_LIMIT, position) -> tuple[int, int, int]:
            c1 = expected_mid_price - (ask_foreign + import_fee) - s[0]
            c2 = expected_mid_price - ask - s[0]
            c3 = bid - expected_mid_price - (s[1] + import_fee)
            
            logger.print(f"Expected mid price: {expected_mid_price}, Ask: ({ask}, {ask_amount}), Bid: ({bid},{bid_amount}), Ask foreign: {ask_foreign}, Import fee: {import_fee}, Position: {position}")
            logger.print(f'c1: {c1}, c2: {c2}, c3: {c3}')

            best = -float('inf')
            for i in range(max(1, 1 - position)):
                for j in range(min(POS_LIMIT - position, -ask_amount) + 1):
                    for k in range(min(POS_LIMIT + position, bid_amount) + 1):
                        if c1*i + c2*j + c3*k - 0.1*max(0, position + i + j - k) > best:
                            best = c1*i + c2*j + c3*k - 0.1*max(0, position + i + j - k)
                            conversions = i
                            buy_amount = j
                            sell_amount = k

            logger.print(f"Conversions: {conversions}, Buy: {buy_amount}, Sell: {sell_amount}, Best: {best}")
            return conversions, buy_amount, sell_amount
        
        if data['expected_mid_price'] is None:
            return [], 0

        orders = []

        observations = state.observations.conversionObservations[product]
        ask_foreign = observations.askPrice
        import_fee = observations.importTariff + observations.transportFees

        bid, bid_amount = list(state.order_depths[product].buy_orders.items())[0]     
        ask, ask_amount = list(state.order_depths[product].sell_orders.items())[0]

        POS_LIMIT, position = data['POS_LIMIT'], state.position.get(product, 0)
        expected_mid_price = data['expected_mid_price']

        conversions, buy_amount, sell_amount = computeBuySell(expected_mid_price, data['spread'], ask, ask_amount, bid, bid_amount, ask_foreign, import_fee, POS_LIMIT, position)
        
        if buy_amount > 0:
            orders.append(Order(product, ask, buy_amount))
            data['num_buy'] += buy_amount
        if sell_amount > 0:
            orders.append(Order(product, bid, -sell_amount))
            data['num_sell'] += sell_amount

        # Market make
        if position + conversions + buy_amount - sell_amount > -POS_LIMIT:
            sell_amount = position + conversions + data['num_buy'] - data['num_sell'] + POS_LIMIT
            orders.append(Order(product, round(expected_mid_price + 2), -sell_amount))

        return orders, conversions
    


    def computeCrossMarketMake(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:
        if data['expected_mid_price'] is None:
            return [], 0

        orders = []

        bid, bid_amount = list(state.order_depths[product].buy_orders.items())[0]
        POS_LIMIT, position = data['POS_LIMIT'], state.position.get(product, 0)

        conversions = max(0, -position)

        if bid > data['expected_mid_price']:
            sell_amount = min(POS_LIMIT + position, bid_amount)
            orders.append(Order(product, bid, -sell_amount))
            data['num_sell'] += sell_amount

        sell_amount = POS_LIMIT + position - data['num_sell']
        orders.append(Order(product, round(data['expected_mid_price'] + data['make_spread']), -sell_amount))

        return orders, conversions



    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]: 
        
        # Initialize returned variables
        result = {}
        conversions = 0
        
        # Initialize traderData in the first iteration
        if state.traderData == "":
            trader_data = TRADER_DATA
        else:
            trader_data = jsonpickle.decode(state.traderData)

        for product in PRODUCTS:

            # Update data with new information from market
            self.updateData(state, product, trader_data[product])

            orders = []
            for strategy in trader_data[product]["strategy"]:

                if strategy == "market_take":
                    orders += self.computeTakeOrders(product, state, trader_data[product])
                    
                elif strategy == "market_make":
                    orders += self.computeMakeOrders(product, state, trader_data[product])
                
                elif strategy == "cross_market":
                    orders, conversions = self.computeCrossMarket(product, state, trader_data[product])

                elif strategy == "cross_market_make":
                    orders, conversions = self.computeCrossMarketMake(product, state, trader_data[product])             

            result[product] = orders

        # Format the output
        trader_data = jsonpickle.encode(trader_data)

        logger.flush(state, result, conversions, "")
        return result, conversions, trader_data