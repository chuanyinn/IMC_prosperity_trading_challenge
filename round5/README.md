## Algorithm challenge

The final round of the challenge is already here! And surprise, no new products are introduced for a change. Dull? Probably not, as you do get another treat. The island exchange now discloses to you who the counterparty is you have traded against. This means that the `counter_party` property of the `OwnTrade` object is now populated. Perhaps interesting to see if you can leverage this information to make your algorithm even more profitable?

```python
class OwnTrade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, counter_party: UserId = None) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.counter_party = counter_party
```

## Manual challenge

You’ve been invited to trade on the exchange of the north archipelago for one day only. An exclusive event and perfect opportunity to make some big final profits before the champion is crowned. The penguins have granted you access to their trusted news source: Iceberg. You’ll find all the information you need right there. Be aware that trading these foreign goods comes at a price. The more you trade in one good, the more expensive it will get. This is the final stretch. Make it count!