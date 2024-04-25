## Algorithm challenge

Our inhabitants are crazy about coconuts. So crazy even, that they invented a new tradable good, `COCONUT_COUPON`. The coupons will give you the right to buy `COCONUT` at a certain price by the end of the round and can be traded as a separate good. Of course you will have to pay a premium for these coupons, but if you play your coconuts right, SeaShells spoils will be waiting for you on the horizon.

Coconut Coupons give the right to buy Coconut at a certain price at some time in future. Certain price = 10000 and some time in future = 250 trading days, each round is 1 trading day.

Position limits for the newly introduced products:

- `COCONUT`: 300
- `COCONUT_COUPON`: 600

## Manual challenge

The goldfish are back with more `SCUBA_GEAR`. Each of the goldfish will have new reserve prices, but they still follow the same distribution as in Round 1.

Your trade options are similar to before. You’ll have two chances to offer a good price. Each one of the goldfish will accept the lowest bid that is over their reserve price. But this time, for your second bid, they also take into account the average of the second bids by other traders in the archipelago. They’ll trade with you when your offer is above the average of all second bids. But if you end up under the average, the probability of a deal decreases rapidly.

To simulate this probability, the PNL obtained from trading with a fish for which your second bid is under the average of all second bids will be scaled by a factor *p*:

$$p = (1000 – \text{average bid}) / (1000 – \text{your bid})$$

What can you learn from the available data and how will you anticipate on this new dynamic? Place your feet firmly in the sand and brace yourself, because this could get messy. 