## Algorithm challenge

Cupid has landed in our archipelago and infected almost every inhabitant. Next to the products from the previous two rounds, the `GIFT_BASKET` is now available as a tradable good. This lovely basket contains three things: 

1. Four `CHOCOLATE` bars
2. Six `STRAWBERRIES`
3. A single `ROSES`

All four of the above products can now also be traded on the island exchange. Are the gift baskets a bit expensive for your taste? Then perhaps see if you can get the contents directly.

Position limits for the newly introduced products:

- `CHOCOLATE`: 250
- `STRAWBERRIES`: 350
- `ROSES`: 60
- `GIFT_BASKET`: 60

## Manual challenge

A mysterious treasure map and accompanying note has everyone acting like heroic adventurers. You get to go on a maximum of three expeditions to search for treasure. Your first expedition is free, but the second and third one will come at a cost. Keep in mind that you are not the only one searching and you’ll have to split the spoils with all the others that search in the same spot. Plan your expeditions carefully and you might return with the biggest loot of all. 

Here's a breakdown of how your profit from an expedition will be computed:
Every spot has its **treasure multiplier** (up to 100) and the number of **hunters** (up to 8). The spot's total treasure is the product of the **base treasure** (7500, same for all spots) and the spot's specific treasure multiplier. However, the resulting amount is then divided by the sum of the hunters and the percentage of all the expeditions (from other players) that took place there. For example, if a field has 5 hunters, and 10% of all the expeditions (from all the other players) are also going there, the prize you get from that field will be divided by 15. After the division, **expedition costs** apply (if there are any), and profit is what remains.

Second and third expeditions are optional: you are not required to do all 3. Fee for embarking upon a second expedition is 25 000, and for third it's 75 000. Order of submitted expeditions does not matter for grading.