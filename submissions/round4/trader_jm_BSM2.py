import json, jsonpickle
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any, List, Dict
import numpy as np
import pandas as pd

PRODUCTS = [
    "AMETHYSTS",
    "STARFRUIT",
    "ORCHIDS",
    "GIFT_BASKET",
    'COCONUT_COUPON',
]

# Parameters and data that can be initialized at the beginning of each iteration
PARAMS = {
    'COCONUT_COUPON': {
        'POS_LIMIT': 600,
        'sell_limit': 60,
        'buy_limit': 60,
        'price_method': 'Black_Scholes',
        'BASE_PRODUCT': 'COCONUT',
        'price_data_size': 1,
        'strategy': ['pair_trading', 'threshold'],
        'HEDGE_PRODUCT': {'COCONUT': None},
        'PROD_POS_LIMIT': {'COCONUT': 300},
        'trade_offset': 8,
        
        'num_buy': 0,
        'num_sell': 0,
        'expected_mid_price': None,
    },

    'GIFT_BASKET':{        
        'POS_LIMIT': 60,
        'sell_limit': 6,
        'buy_limit': 6,
        'price_method': 'basket',
        'price_data_size': 1,
        'basket_offset': 379.5,
        'strategy': ['pair_trading', 'threshold'],
        'HEDGE_PRODUCT': {'CHOCOLATE': 4, 'STRAWBERRIES': 6, 'ROSES': 1},
        'PROD_POS_LIMIT': {'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60},
        'trade_offset': 40,

        'num_buy': 0,
        'num_sell': 0,
        'expected_mid_price': None,
    },

    'ORCHIDS': {
        'POS_LIMIT': 100,
        'price_method': 'foreign',
        'price_data_size': 1,
        'strategy': ['cross_market_make'],
        'make_price_offset': [1, -2],

        'num_buy': 0,
        'num_sell': 0,
        'expected_mid_price': None,
    },
    
    'AMETHYSTS': {
        'POS_LIMIT': 20,
        'strategy': ['market_take', 'market_make'],
        'price_method': 'static',
        'take_position_stage': [0, 20],
        'take_price_spread': [(2, 2), (2, 0)],
        'make_position_stage': [0, 15, 20],
        'make_price_spread': [(1, 1), (1, 2), (0, 2)],
        'make_price_offset': [1, 1],

        'num_buy': 0,
        'num_sell': 0,
        'expected_mid_price': 10_000,
    },
    
    'STARFRUIT': {
        'POS_LIMIT': 20,
        'strategy': ['market_take', 'market_make'],

        # Data and parameters specific for mid_price calculation: method = "MA_1
        'price_method': 'MA_1',
        'MA_coef': -0.7086,
        'price_data_size': 1,

        # 'price_method': 'average',
        # 'price_data_size': 8,

        # Parameters specific for mid_price calculation: method = "regression"
        # 'price_method': 'regression',
        # 'coef': [-0.01869561,  0.0455032 ,  0.16316049,  0.8090892],
        # 'intercept': 4.481696494462085,
        # 'price_data_size': 4,

        'take_position_stage': [0, 20],
        'take_price_spread': [(1, 1), (1, 0)],
        'make_position_stage': [20],
        'make_price_spread': [(1, 1)],
        'make_price_offset': [1, 1],

        'num_buy': 0,
        'num_sell': 0,
        'expected_mid_price': None,
    },
}

# Persistent data that need to be stored and passed to the next iteration
TRADER_DATA = {
    'COCONUT_COUPON': {
        'mid_price_data': [],
    },

    'GIFT_BASKET':{
        'mid_price_data': [],
    },

    'ORCHIDS': {
        'mid_price_data': [],
        'foreign_price_data': [],
        'humidity_data': [],
        'sunlight_data': [],
    },

    'AMETHYSTS': {
    },
    
    'STARFRUIT': {
        'mid_price_data': [],
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
    
    def getBlackScholes(self, S: float, K: float=10_000, T: float=246, sigma: float=0.01009) -> float:
        d1 = (np.log(S/10_000) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        norm_cdf_d1 = 1/2 + d1/(np.sqrt(2 * np.pi))
        norm_cdf_d2 = 1/2 + d2/(np.sqrt(2 * np.pi))
        return float(S * norm_cdf_d1 - 10_000 * norm_cdf_d2), float(norm_cdf_d1), float(10_000 * norm_cdf_d2)
    


    def calculateAverage(self, data: list[int], weights: list[float]|None, round_output=True) -> int:
        if weights:
            ans = sum([data[i] * weights[i] for i in range(len(data))])/sum(weights)
        else:
            ans = sum(data) / len(data)
        if round_output:
            return round(ans)
        return ans



    def getPriceSpread(self, position: int, stage:tuple[int], spread:tuple[tuple[int, int]]) -> tuple[int, int]:
        sign = 1 if position >= 0 else -1
        for i, pos in enumerate(stage):
            if abs(position) <= pos:
                return spread[i][::sign]



    def getBestBidAsk(self, product: Symbol, state: TradingState) -> tuple[int, int, int]:
        buy_orders, sell_orders = state.order_depths[product].buy_orders.items(), state.order_depths[product].sell_orders.items()
        if len(buy_orders) > 0 and len(sell_orders) > 0:
            best_bid = list(buy_orders)[0][0]
            best_ask = list(sell_orders)[0][0]
            mid_price = (best_ask + best_bid) / 2
        elif len(buy_orders) > 0:
            best_bid = list(buy_orders)[0][0]
            best_ask = None
            mid_price = None
        elif len(sell_orders) > 0:
            best_bid = None
            best_ask = list(sell_orders)[0][0]
            mid_price = None
        return best_bid, best_ask, mid_price



    def updatePriceObervations(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> None:
        params = PARAMS[product]
        
        # Update mid_price_data
        if 'mid_price_data' in data:
            _, _, mid_price = self.getBestBidAsk(product, state)
            if mid_price is not None:
                data['mid_price_data'].append(mid_price)
            if len(data['mid_price_data']) > params['price_data_size']:
                data['mid_price_data'].pop(0)

        # Update conversion observations data
        if product == 'ORCHIDS':
            conversion_observations = state.observations.conversionObservations[product]
            foreign_price = (conversion_observations.bidPrice + conversion_observations.askPrice) / 2

            data['humidity_data'].append(conversion_observations.humidity)
            data['sunlight_data'].append(conversion_observations.sunlight)
            data['foreign_price_data'].append(foreign_price)

            if len(data['humidity_data']) > params['price_data_size']:
                data['humidity_data'].pop(0)
                data['sunlight_data'].pop(0)            
                data['foreign_price_data'].pop(0)



    def updateData(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> None:
        params = PARAMS[product]
        params['num_buy'] = params['num_sell'] = 0
        
        self.updatePriceObervations(product, state, data)

        # Moving Average 1 model to predict mid_price
        if params['price_method'] == 'MA_1':      
            if params['expected_mid_price'] is None:
                params['expected_mid_price'] = data['mid_price_data'][-1]
            else:
                d = params['MA_coef'] * (data['mid_price_data'][-1] - params['expected_mid_price'])
                params['expected_mid_price'] = data['mid_price_data'][-1] + d


        # Simple Average model to predict mid_price
        elif params['price_method'] == 'average':
            if len(data['mid_price_data']) == params['price_data_size']:
                params['expected_mid_price'] = self.calculateAverage(data['mid_price_data'], None, round_output=False)


        # Regression model to predict mid_price
        elif params['price_method'] == 'regression':
            if len(data['mid_price_data']) == params['price_data_size']:
                params['expected_mid_price'] = sum([data['mid_price_data'][i] * 
                                                  params['coef'][i] 
                                                  for i in range(params['price_data_size'])]) + params['intercept']
        

        # Use foreign market mid price as mid_price
        elif params['price_method'] == 'foreign':
            conversion_observations = state.observations.conversionObservations[product]
            params['expected_mid_price'] = (conversion_observations.askPrice + conversion_observations.bidPrice) / 2


        # Use basket components to calculate mid_price
        elif params['price_method'] == 'basket':
            price = params['basket_offset']
            for p in params['HEDGE_PRODUCT']:
                _, _, p_mid_price = self.getBestBidAsk(p, state)
                if p_mid_price is None: return
                price += params['HEDGE_PRODUCT'][p] * p_mid_price
            params['expected_mid_price'] = price

        
        # Use Black Scholes model to calculate mid_price and hedge factor
        elif params['price_method'] == 'Black_Scholes':
            BASE_PRODUCT = params['BASE_PRODUCT']
            base_price = self.getBestBidAsk(params['BASE_PRODUCT'], state)[2]
            params['expected_mid_price'], params['HEDGE_PRODUCT'][BASE_PRODUCT], _ = self.getBlackScholes(base_price)



    def computeTakeOrders(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:
        params = PARAMS[product]
        if params['expected_mid_price'] is None:
            return []

        orders = []
        POS_LIMIT, position = params['POS_LIMIT'], state.position.get(product, 0)
        
        buy_offset, sell_offset = self.getPriceSpread(position, 
                                                      params['take_position_stage'], 
                                                      params['take_price_spread'])
        
        acc_ask = params['expected_mid_price'] - buy_offset # price to buy, lower the better
        acc_bid = params['expected_mid_price'] + sell_offset # price to sell, higher the better

        buy_orders, sell_orders = state.order_depths[product].buy_orders.items(), state.order_depths[product].sell_orders.items()
        for ask, ask_amount in sell_orders:
            if (ask <= acc_ask) and (params['num_buy'] < POS_LIMIT - position):
                buy_amount = min(-ask_amount, POS_LIMIT - position - params['num_buy'])
                orders.append(Order(product, ask, buy_amount))
                params['num_buy'] += buy_amount

        for bid, bid_amount in buy_orders:
            if (bid >= acc_bid) and (params['num_sell'] < POS_LIMIT + position):
                sell_amount = min(bid_amount, POS_LIMIT + position - params['num_sell'])
                orders.append(Order(product, bid, -sell_amount))
                params['num_sell'] += sell_amount

        return orders
    


    def computeMakeOrders(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> list[Order]:
        params = PARAMS[product]
        if params['expected_mid_price'] is None:
            return []
        
        orders = []
        best_bid, best_ask, _ = self.getBestBidAsk(product, state)
        POS_LIMIT, position = params['POS_LIMIT'], state.position.get(product, 0)

        buy_offset, sell_offset = self.getPriceSpread(position + params['num_buy'] - params['num_sell'], 
                                                      params['make_position_stage'], 
                                                      params['make_price_spread'])
        buy_offset2, sell_offset2 = params['make_price_offset']

        our_bid = min(best_bid + buy_offset, round(params['expected_mid_price'] - buy_offset2))
        our_ask = max(best_ask - sell_offset, round(params['expected_mid_price'] + sell_offset2))
        buy_amount = POS_LIMIT - position - params['num_buy']
        sell_amount = POS_LIMIT + position - params['num_sell']
        
        if buy_amount > 0:
            orders.append(Order(product, our_bid, buy_amount))
            params['num_buy'] += buy_amount

        if sell_amount > 0:
            orders.append(Order(product, our_ask, -sell_amount))
            params['num_sell'] += sell_amount
        
        return orders
    


    def computeCrossMarketMake(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> tuple[list[Order], int]:
        params = PARAMS[product]
        if params['expected_mid_price'] is None:
            return [], 0
        
        POS_LIMIT, position = params['POS_LIMIT'], state.position.get(product, 0)
        conversions = max(0, -position)
        # After conversion, position is 0
        
        orders = []
        observations = state.observations.conversionObservations[product]
        ask_price = observations.askPrice + observations.importTariff + observations.transportFees
        ask_offset, ask_offset2 = params['make_price_offset']

        bid, bid_amount = list(state.order_depths[product].buy_orders.items())[0]
        sell_price = round(max(ask_price + ask_offset, params['expected_mid_price'] + ask_offset2))
        
        # Market taker
        if bid > ask_price:
            sell_amount = min(POS_LIMIT, bid_amount)
            orders.append(Order(product, bid, -sell_amount))
            params['num_sell'] += sell_amount

        # Market maker
        sell_amount = POS_LIMIT - params['num_sell']
        orders.append(Order(product, sell_price, -sell_amount))

        return orders, conversions



    def computePairTrading(self, product: Symbol, state: TradingState) -> Dict[Symbol, List[Order]]:
        params = PARAMS[product]
        position = state.position.get(product, 0)

        # Pair trading with individual products to hedge the position
        orders = {p: [] for p in params['HEDGE_PRODUCT']}
        for p in params['HEDGE_PRODUCT']:
            prod_position, prod_pos_limit = state.position.get(p, 0), params['PROD_POS_LIMIT'][p]
            pos_diff  = round(prod_position + position * params['HEDGE_PRODUCT'][p])
            
            if pos_diff > 0:
                sell_amount = min(pos_diff, prod_pos_limit + prod_position)
                buy_orders = state.order_depths[p].buy_orders.items()
                for bid, bid_amount in buy_orders:
                    if sell_amount == 0:
                        break
                    sell_vol = min(sell_amount, bid_amount)
                    orders[p].append(Order(p, bid, -sell_vol))
                    sell_amount -= sell_vol

            elif pos_diff < 0:
                buy_amount = min(-pos_diff, prod_pos_limit - prod_position)
                sell_orders = state.order_depths[p].sell_orders.items()
                for ask, ask_amount in sell_orders:
                    if buy_amount == 0:
                        break
                    buy_vol = min(buy_amount, -ask_amount)
                    orders[p].append(Order(p, ask, buy_vol))
                    buy_amount -= buy_vol
                    
        return orders



    def computeThresholdOrders(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> List[Order]:
        params = PARAMS[product]
        orders = []

        POS_LIMIT, position = params['POS_LIMIT'], state.position.get(product, 0)
        buy_limit, sell_limit = params['buy_limit'], params['sell_limit']
        _, _, mid_price = self.getBestBidAsk(product, state)
        basket_price = params['expected_mid_price']

        if mid_price is None: return orders
        
        if (mid_price - basket_price > params['trade_offset']):
            buy_orders = state.order_depths[product].buy_orders.items()
            for bid, bid_amount in buy_orders:
                if params['num_sell'] == min(POS_LIMIT + position, sell_limit):
                    break
                sell_vol = min([bid_amount, POS_LIMIT + position - params['num_sell'], sell_limit - params['num_sell']])
                orders.append(Order(product, bid, -sell_vol))
                params['num_sell'] += sell_vol

        elif (mid_price - basket_price < -params['trade_offset']):
            sell_orders = state.order_depths[product].sell_orders.items()          
            for ask, ask_amount in sell_orders:
                if params['num_buy'] == min(POS_LIMIT - position, buy_limit):
                    break
                buy_vol = min([-ask_amount, POS_LIMIT - position - params['num_buy'], buy_limit - params['num_buy']])
                orders.append(Order(product, ask, buy_vol))
                params['num_buy'] += buy_vol
        return orders



    def computeMomentum(self, product: Symbol, state: TradingState, data: Dict[str, Any]) -> None:
        pass



    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]: 
        
        # Initialize returned variables
        result = {p: [] for p in PRODUCTS}
        conversions = 0
        
        # Initialize traderData in the first iteration
        trader_data = TRADER_DATA if state.traderData == "" else jsonpickle.decode(state.traderData)

        for product in PRODUCTS:
            if product not in state.listings: continue

            # Update data with new information from market
            self.updateData(product, state, trader_data[product])

            logger.print(f"Product: {product}, trader_data: {trader_data[product]}")

            for strategy in PARAMS[product]["strategy"]:
                if strategy == "market_take":
                    result[product] = self.computeTakeOrders(product, state, trader_data[product])                        
                elif strategy == "market_make":
                    result[product] += self.computeMakeOrders(product, state, trader_data[product])
                elif strategy == "cross_market_make":
                    result[product], conversions = self.computeCrossMarketMake(product, state, trader_data[product])
                elif strategy == 'threshold':
                    result[product] = self.computeThresholdOrders(product, state, trader_data[product])
                elif strategy == 'momentum':
                    result[product] = self.computeMomentum(product, state, trader_data[product])
                elif strategy == 'pair_trading':
                    orders = self.computePairTrading(product, state)
                    for p in orders:
                        result[p] = orders[p]

        # Format the output
        trader_data = jsonpickle.encode(trader_data)

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
    