import random

def get_color_labels(prediction, skin_tone):
    color_labels = {
        'Fair': [
            {'name': 'Navy Blue', 'hex_code': '#000080'},
            {'name': 'Beige', 'hex_code': '#F5F5DC'},
            {'name': 'Saddle Brown', 'hex_code': '#8B4513'},
            {'name': 'Olive Green', 'hex_code': '#808000'},
            {'name': 'Sky Blue', 'hex_code': '#87CEEB'},
            {'name': 'Gray', 'hex_code': '#808080'},
            {'name': 'Light Pink', 'hex_code': '#FFB6C1'},
            {'name': 'Mint Green', 'hex_code': '#98FF98'},
            {'name': 'Lavender', 'hex_code': '#E6E6FA'},
            {'name': 'Baby Blue', 'hex_code': '#89CFF0'},
            {'name': 'Emerald Green', 'hex_code': '#50C878'},
            {'name': 'Turquoise', 'hex_code': '#40E0D0'},
            {'name': 'Cobalt Blue', 'hex_code': '#0047AB'},
            {'name': 'Camel', 'hex_code': '#C19A6B'},
            {'name': 'Ivory', 'hex_code': '#FFFFF0'},
        ],
        'Medium': [
            {'name': 'Coral', 'hex_code': '#FF7F50'},
            {'name': 'Black', 'hex_code': '#000000'},
            {'name': 'Copper', 'hex_code': '#B87333'},
            {'name': 'Teal', 'hex_code': '#008080'},
            {'name': 'Camel', 'hex_code': '#C19A6B'},
            {'name': 'Chocolate Brown', 'hex_code': '#D2691E'},
            {'name': 'Deep Purple', 'hex_code': '#673AB7'},
            {'name': 'Jade Green', 'hex_code': '#00A36C'},
            {'name': 'Peach', 'hex_code': '#FFDAB9'},
            {'name': 'Soft Mint', 'hex_code': '#98FF98'},
            {'name': 'Dusty Rose', 'hex_code': '#DCAE96'},
            {'name': 'Off-White', 'hex_code': '#F8F8FF'},
            {'name': 'Bronze', 'hex_code': '#CD7F32'},
            {'name': 'Olive Green', 'hex_code': '#808000'},
            {'name': 'Burgundy', 'hex_code': '#800020'},
        ],
        'Dark': [
            {'name': 'Electric Blue', 'hex_code': '#7DF9FF'},
            {'name': 'Fuchsia', 'hex_code': '#FF00FF'},
            {'name': 'Bright Yellow', 'hex_code': '#FFD700'},
            {'name': 'Vibrant Orange', 'hex_code': '#FF6700'},
            {'name': 'Deep Emerald Green', 'hex_code': '#006400'},
            {'name': 'Royal Purple', 'hex_code': '#7851A9'},
            {'name': 'Deep Purple', 'hex_code': '#673AB7'},
            {'name': 'Rich Burgundy', 'hex_code': '#800020'},
            {'name': 'Deep Sapphire Blue', 'hex_code': '#082567'},
            {'name': 'Burnt Orange', 'hex_code': '#CC5500'},
            {'name': 'Pink', 'hex_code': '#FFC0CB'},
            {'name': 'Olive Green', 'hex_code': '#808000'},
            {'name': 'Mahogany', 'hex_code': '#C04000'},
            {'name': 'Gold', 'hex_code': '#FFD700'},
            {'name': 'Bronze', 'hex_code': '#CD7F32'},
            {'name': 'Copper', 'hex_code': '#B87333'},
        ]
    }

    if skin_tone in color_labels:
        return random.sample(color_labels[skin_tone], 3)  # Randomly pick 3 colors
    else:
        return [{'name': 'Unknown', 'hex_code': '#FFFFFF'}]