import json
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import re
import cairosvg
import io

class MagicCardViewer:
    def __init__(self, root, folder_path, mana_symbol_path):
        self.root = root
        self.root.title("Magic Card Viewer")
        self.root.geometry("600x800")
        self.root.resizable(False, False)

        self.folder_path = os.path.abspath(folder_path)
        self.mana_symbol_path = os.path.abspath(mana_symbol_path)
        self.json_files = [f for f in os.listdir(self.folder_path) if f.endswith('.json') or f.endswith('.txt')]
        self.current_index = 0

        self.mana_symbols = {}  # Cache for mana symbol images
        self.card_images = {}  # Cache for card images

        if not self.json_files:
            messagebox.showerror("Error", "No text files found in the specified folder.")
            root.destroy()
            return

        # Create card frame with a border
        self.card_frame = tk.Frame(root, bd=2, relief="solid")
        self.card_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create a Canvas for the gradient background
        self.gradient_canvas = tk.Canvas(self.card_frame, highlightthickness=0)
        self.gradient_canvas.pack(fill=tk.BOTH, expand=True)

        # Create label for card name
        self.name_label = tk.Label(self.gradient_canvas, text="", font=("Helvetica", 16, "bold"), bg="white")
        self.name_label.place(x=10, y=20, anchor="nw")  # Adjusted y position to avoid overlap

        # Create frame for mana cost, aligned to the top right
        self.mana_cost_frame = tk.Frame(self.gradient_canvas, bg="white")
        self.mana_cost_frame.place(x=570, y=20, anchor="ne")  # Adjusted x position

        # Add a frame to hold the image
        self.image_frame = tk.Frame(self.gradient_canvas, bg="white")
        self.image_frame.pack(pady=(65, 20))  # Adjusted padding to avoid overlap

        # Create a label for the image
        self.image_label = tk.Label(self.image_frame, bg="white")
        self.image_label.pack()

        # Create combined label for card type and rarity
        self.type_rarity_label = tk.Label(self.gradient_canvas, text="", font=("Helvetica", 12), anchor="w", bg="white")
        self.type_rarity_label.pack(pady=5)

        self.oracle_text_box = tk.Text(self.gradient_canvas, wrap=tk.WORD, height=7, font=("Helvetica", 10), bg="white", borderwidth=0, relief="flat")
        self.oracle_text_box.config(state=tk.DISABLED)
        self.oracle_text_box.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.flavor_text_box = tk.Text(self.gradient_canvas, wrap=tk.WORD, height=3, font=("Helvetica", 10, "italic"), bg="white", borderwidth=0, relief="flat")
        self.flavor_text_box.config(state=tk.DISABLED)
        self.flavor_text_box.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # Single label for Power/Toughness or Loyalty, right-aligned
        self.pt_loyalty_label = tk.Label(self.gradient_canvas, text="", font=("Helvetica", 12), anchor="e", bg="white")
        self.pt_loyalty_label.pack(pady=5, anchor="e", padx=10)

        # New bar for theme, GPT model, and image generator, anchored to the bottom
        self.info_bar = tk.Label(self.gradient_canvas, text="", font=("Helvetica", 8), anchor="center", bg="lightgray")
        self.info_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Add buttons directly below the card frame
        self.prev_button = tk.Button(root, text="Previous", command=self.prev_card, font=("Helvetica", 12))
        self.prev_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.next_button = tk.Button(root, text="Next", command=self.next_card, font=("Helvetica", 12))
        self.next_button.pack(side=tk.RIGHT, padx=20, pady=10)

        # Load the first card
        self.load_card()

    def load_card(self):
        try:
            if 0 <= self.current_index < len(self.json_files):
                card_file = os.path.join(self.folder_path, self.json_files[self.current_index])
                with open(card_file, 'r') as file:
                    card_data = json.load(file)

                # Set card name
                self.name_label.config(text=card_data['name'])

                # Load and display mana cost, hide frame if no cost
                if card_data['mana_cost']:
                    self.mana_cost_frame.place(x=570, y=20, anchor="ne")
                    self.display_mana_cost(card_data['mana_cost'])
                else:
                    self.mana_cost_frame.place_forget()  # Hide if no mana cost

                # Load and display card image
                self.load_card_image(card_file)

                # Combine the type and rarity
                type_and_rarity = f"{card_data['type_line']} | {card_data['rarity']}"
                self.type_rarity_label.config(text=type_and_rarity)

                # Display the oracle text with mana symbols
                self.display_oracle_text(card_data['oracle_text'])

                # Display the flavor text with respect to newlines
                if 'flavor_text' in card_data:
                    flavor_text = card_data['flavor_text'].replace("\\n", "\n")
                    self.flavor_text_box.config(state=tk.NORMAL)
                    self.flavor_text_box.delete(1.0, tk.END)
                    self.flavor_text_box.insert(tk.END, flavor_text)
                    self.flavor_text_box.config(state=tk.DISABLED)
                else:
                    self.flavor_text_box.pack_forget()  # Hide the text box

                # Display Power/Toughness or Loyalty
                if card_data.get('power') and card_data.get('toughness'):
                    self.pt_loyalty_label.config(text=f"{card_data['power']}/{card_data['toughness']}")
                    self.pt_loyalty_label.pack(anchor="e", pady=5)
                elif card_data.get('loyalty'):
                    self.pt_loyalty_label.config(text=f"{card_data['loyalty']}")
                    self.pt_loyalty_label.pack(anchor="e", pady=5)
                else:
                    self.pt_loyalty_label.pack_forget()  # Hide the label if no data

                # Display theme, GPT model, and image generator in the info bar
                info_text = f"Theme: {card_data['theme']} | GPT Model: {card_data['gpt-model']} | Image Generator: {card_data['image_generator']}"
                self.info_bar.config(text=info_text)

                # Create gradient based on mana cost
                self.create_gradient_background(card_data['mana_cost'], card_data['type_line'])

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load card: {e}")

    def load_card_image(self, card_file):
        # Generate the corresponding image file path
        image_file = os.path.splitext(card_file)[0] + ".png"
        if os.path.exists(image_file):
            if image_file not in self.card_images:
                image = Image.open(image_file)
                image = image.resize((515, 325), Image.Resampling.LANCZOS)  # Resize to fit the card layout
                self.card_images[image_file] = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.card_images[image_file])
            self.image_label.image = self.card_images[image_file]  # Keep a reference to prevent garbage collection
            self.image_label.pack()  # Ensure the image is displayed
        else:
            self.image_label.pack_forget()  # Hide the label if no image is available

    def display_mana_cost(self, mana_cost):
        # Clear previous mana symbols
        for widget in self.mana_cost_frame.winfo_children():
            widget.destroy()

        # Regex to match mana symbols inside curly braces
        symbol_regex = re.compile(r'{(.*?)}')
        symbols = symbol_regex.findall(mana_cost)

        for symbol in symbols:
            if symbol == "T":
                symbol = "tap"
            cache_key = (symbol, 15)  # Include size in the cache key
            if cache_key not in self.mana_symbols:
                normalized_symbol = self.normalize_symbol(symbol)
                svg_path = os.path.join(self.mana_symbol_path, f"{normalized_symbol}.svg")
                if os.path.exists(svg_path):
                    image = self.convert_svg_to_png(svg_path, symbol, size=15)  # Larger size for mana cost
                    self.mana_symbols[cache_key] = ImageTk.PhotoImage(image)
            if cache_key in self.mana_symbols:
                mana_icon = tk.Label(self.mana_cost_frame, image=self.mana_symbols[cache_key], bg="white")
                mana_icon.pack(side=tk.LEFT, padx=2)


    def normalize_symbol(self, symbol):
        """Normalize the mana symbol for file naming conventions."""
        # Lowercase and replace any special characters
        normalized_symbol = symbol.lower()
        normalized_symbol = re.sub(r'\W+', '', normalized_symbol)  # Remove non-alphanumeric characters
        return normalized_symbol

    def convert_svg_to_png(self, svg_path, symbol, size):
        # Load SVG and replace black pixels with the symbol color
        color_map = {
            "W": "#FFFFF0",  # White
            "U": "#ADD8E6",  # Blue
            "B": "#A9A9A9",  # Black (Dark Gray)
            "R": "#FF6347",  # Red
            "G": "#90EE90",  # Green
            # Add more mappings as needed
        }
        color = color_map.get(symbol, "white")  # Default to white if symbol not found

        # Read the SVG content and replace black with the desired color
        with open(svg_path, 'r') as file:
            svg_content = file.read()

        # Replace only black (#000000 or rgb(0,0,0)) with the desired color
        svg_content = svg_content.replace('#000000', color).replace('rgb(0,0,0)', color)

        # Convert the modified SVG to PNG
        png_data = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'))

        # Adjust the size based on the provided size parameter
        image = Image.open(io.BytesIO(png_data))
        image = image.resize((size, size), Image.Resampling.LANCZOS)

        return image


    def display_oracle_text(self, oracle_text):
        self.oracle_text_box.config(state=tk.NORMAL)
        self.oracle_text_box.delete(1.0, tk.END)

        # Remove any stray \n in the oracle text
        oracle_text = oracle_text.replace('\\n', '\n')

        # Split oracle text and identify mana symbols
        parts = re.split(r'({.*?})', oracle_text)

        for part in parts:
            if re.match(r'{.*?}', part):
                symbol = re.sub(r'[{}]', '', part)
                if symbol == "T":
                    symbol = "tap"
                cache_key = (symbol, 11)  # Include size in the cache key
                if cache_key not in self.mana_symbols:
                    normalized_symbol = self.normalize_symbol(symbol)
                    svg_path = os.path.join(self.mana_symbol_path, f"{normalized_symbol}.svg")
                    if os.path.exists(svg_path):
                        image = self.convert_svg_to_png(svg_path, symbol, size=11)  # Smaller size for Oracle text
                        self.mana_symbols[cache_key] = ImageTk.PhotoImage(image)
                if cache_key in self.mana_symbols:
                    self.oracle_text_box.image_create(tk.END, image=self.mana_symbols[cache_key])
                    self.oracle_text_box.insert(tk.END, " ")  # Insert a space after each symbol for padding
            else:
                self.oracle_text_box.insert(tk.END, part)

        self.oracle_text_box.config(state=tk.DISABLED)


    def create_gradient_background(self, mana_cost, type_line):
        self.gradient_canvas.delete("all")
        colors = []
        if "Land" in type_line:
            colors.append("#D2B48C")  # Lighter, earthy brown color for lands
        else:
            if "W" in mana_cost:
                colors.append("lightyellow")
            if "U" in mana_cost:
                colors.append("lightblue")
            if "B" in mana_cost:
                colors.append("darkgray")
            if "R" in mana_cost:
                colors.append("lightcoral")
            if "G" in mana_cost:
                colors.append("lightgreen")
            if len(colors) == 0:  # If no colors match, use neutral gray for colorless cards
                colors.append("lightgray")

        if len(colors) == 1:
            self.gradient_canvas.config(bg=colors[0])
        else:
            self.draw_gradient(colors)

    def draw_gradient(self, colors):
        width = self.gradient_canvas.winfo_width()
        height = self.gradient_canvas.winfo_height()

        if width == 1 or height == 1:  # Prevent errors if the canvas isn't ready yet
            self.gradient_canvas.after(10, lambda: self.draw_gradient(colors))
            return

        num_steps = len(colors) - 1
        step_size = width // num_steps

        for i in range(num_steps):
            color1 = self.gradient_canvas.winfo_rgb(colors[i])
            color2 = self.gradient_canvas.winfo_rgb(colors[i + 1])

            r1, g1, b1 = color1[0] // 256, color1[1] // 256, color1[2] // 256
            r2, g2, b2 = color2[0] // 256, color2[1] // 256, color2[2] // 256

            for j in range(step_size):
                r = int(r1 + (r2 - r1) * j / step_size)
                g = int(g1 + (g2 - g1) * j / step_size)
                b = int(b1 + (b2 - b1) * j / step_size)
                color = f'#{r:02x}{g:02x}{b:02x}'
                x1 = i * step_size + j
                self.gradient_canvas.create_line(x1, 0, x1, height, fill=color)

    def next_card(self):
        if self.current_index < len(self.json_files) - 1:
            self.current_index += 1
            self.load_card()
        else:
            messagebox.showinfo("End", "No more cards to display.")

    def prev_card(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_card()
        else:
            messagebox.showinfo("Start", "You are already at the first card.")

# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    folder_path = "./art/out/"  # Replace with the actual path to your folder with .txt files
    mana_symbol_path = "./mana/svg/"  # Path to the folder containing the SVG mana symbols
    app = MagicCardViewer(root, folder_path, mana_symbol_path)
    root.mainloop()
