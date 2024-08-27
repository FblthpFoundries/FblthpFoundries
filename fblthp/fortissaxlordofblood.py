import random
from typing import List
import numpy as np
from collections import defaultdict

def seed_card():

    def pick_even_cid(enable_colorless: bool = True) -> List[str]:
        color_props = [0.08, 0.74, 0.11, 0.06, 0.005, 0.005]
        color_options = ["White", "Blue", "Black", "Red", "Green"]
        possible = [0, 1, 2, 3, 4, 5]

        # If colorless is not enabled, remove the option for 0 colors
        if not enable_colorless:
            possible = possible[1:]
            color_props = color_props[1:]
            
        # Step 1: Determine the number of colors to pick
        num_colors = random.choices(possible, weights=color_props, k=1)[0]

        # Step 2: Randomly pick the required number of new colors
        c_list = random.sample(color_options, min(num_colors, len(color_options)))

        # Step 3: Return the updated color list
        return c_list
    
    
    mana_symbols = {"White": "{W}", "Blue": "{U}", "Black": "{B}", "Red": "{R}", "Green": "{G}"}
    
    types = {
        "Artifact": 3185,
        "Enchantment": 3303,
        "Land": 1040,
        "Creature": 15555,
        "Instant": 3375,
        "Sorcery": 3154,
        "Planeswalker": 292,
    }
    options = list(types.keys())
    weights = list(types.values())

    # Select one type based on the weights
    selected_type = random.choices(options, weights=weights, k=1)[0]
    colors = []

    if selected_type == "Artifact":
        if random.random() < 0.15:
            selected_type += " Equipment"
        elif random.random() < 0.1:
            selected_type += " Vehicle"
        if random.random() < 0.7:
            colors = []
        else:
            colors = pick_even_cid(enable_colorless=False)
    
    elif selected_type == "Land":
        colors = pick_even_cid()
    elif selected_type == "Enchantment":
        if random.random() < 0.2:
            selected_type += " Aura"
        colors = pick_even_cid()
                
    else:
        colors = pick_even_cid()

    
    # Adjusted distribution of converted mana cost
    cmc_weights = [0.20, 0.267, 0.217, 0.15, 0.10, 0.067]
    cmc_values = [1, 2, 3, 4, 5, random.randint(6, 10)]  # 6+ treated as a random value from 6 to 10

    generic_mana = random.choices(cmc_values, weights=cmc_weights, k=1)[0]
    mana_cost = ""
    
    
    # Weighted choice for the number of mana symbols
    symbol_weights = [0.89, 0.1, 0.01]  # 75% chance for 1 symbol, 20% for 2, 5% for 3
    symbol_choices = [1, 2, 3]
    total_syms = 0
    for color in colors:
        num_symbols = random.choices(symbol_choices, weights=symbol_weights, k=1)[0]
        total_syms += num_symbols
        mana_cost += mana_symbols[color] * num_symbols
    gm = max(generic_mana - total_syms, 0)
    if gm > 0:
        mana_cost = f"{gm}" + mana_cost
    return selected_type, colors, mana_cost

def test_seed_card(runs=1000):
    from collections import defaultdict
    
    type_counter = defaultdict(int)
    color_counter = defaultdict(int)
    total_mana_cost = 0
    num_cards_with_colors = 0

    for _ in range(runs):
        card_type, colors, mana_cost = seed_card()
        
        # Count card types
        type_counter[card_type] += 1
        
        # Count colors
        for color in colors:
            color_counter[color] += 1
        
        # Accumulate total mana cost
        mana_cost_values = []
        
        # Extract and sum all mana symbols (both generic and colored)
        for part in mana_cost.split("{"):
            if part:
                value = part.split("}")[0]
                if value.isdigit():  # This handles generic mana like {2}, {3}, etc.
                    mana_cost_values.append(int(value))
                else:  # This handles colored mana like {W}, {U}, etc.
                    mana_cost_values.append(1)
        
        total_mana_cost += sum(mana_cost_values)

        # Count how many cards had colors
        if colors:
            num_cards_with_colors += 1

    # Calculate averages
    avg_mana_cost = total_mana_cost / runs
    avg_colors_per_card = num_cards_with_colors / runs

    # Print the results
    print("Card Type Averages:")
    for card_type, count in type_counter.items():
        print(f"{card_type}: {count / runs * 100:.2f}%")
    
    print("\nColor Distribution Averages:")
    for color, count in color_counter.items():
        print(f"{color}: {count / runs * 100:.2f}%")
    
    print(f"\nAverage Mana Cost: {avg_mana_cost:.2f}")
    print(f"Average Number of Cards with Colors: {avg_colors_per_card:.2f}")


# Example usage
if __name__ == "__main__":
    test_seed_card(runs=60000)
