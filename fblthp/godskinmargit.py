import json
import os
import tkinter as tk
from tkinter import messagebox

class MagicCardViewer:
    def __init__(self, root, folder_path):
        self.root = root
        self.root.title("Magic Card Viewer")
        self.root.geometry("600x800")
        self.root.resizable(False, False)

        self.folder_path = folder_path
        self.json_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        self.current_index = 0

        if not self.json_files:
            messagebox.showerror("Error", "No text files found in the specified folder.")
            root.destroy()
            return

        # Create card frame with a border
        self.card_frame = tk.Frame(root, bg="white", bd=2, relief="solid")
        self.card_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Create combined label for card name and mana cost
        self.name_mana_label = tk.Label(self.card_frame, text="", font=("Helvetica", 16, "bold"), wraplength=480, anchor="w", bg="white")
        self.name_mana_label.pack(pady=10)

        # Create combined label for card type and rarity
        self.type_rarity_label = tk.Label(self.card_frame, text="", font=("Helvetica", 12), wraplength=480, anchor="w", bg="white")
        self.type_rarity_label.pack(pady=5)

        self.theme_label = tk.Label(self.card_frame, text="", font=("Helvetica", 12, "italic"), wraplength=480, bg="white")
        self.theme_label.pack(pady=5)

        self.oracle_text_label = tk.Label(self.card_frame, text="Oracle Text:", font=("Helvetica", 12, "bold"), bg="white")
        self.oracle_text_label.pack(pady=5)

        self.oracle_text_box = tk.Text(self.card_frame, wrap=tk.WORD, height=10, font=("Helvetica", 10), bg="white", borderwidth=0, relief="flat")
        self.oracle_text_box.config(state=tk.DISABLED)
        self.oracle_text_box.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.flavor_text_label = tk.Label(self.card_frame, text="Flavor Text:", font=("Helvetica", 12, "italic"), bg="white")
        self.flavor_text_label.pack(pady=5)

        self.flavor_text_box = tk.Text(self.card_frame, wrap=tk.WORD, height=5, font=("Helvetica", 10, "italic"), bg="white", borderwidth=0, relief="flat")
        self.flavor_text_box.config(state=tk.DISABLED)
        self.flavor_text_box.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.pt_label = tk.Label(self.card_frame, text="", font=("Helvetica", 12), wraplength=480, bg="white")
        self.pt_label.pack(pady=5)

        self.loyalty_label = tk.Label(self.card_frame, text="", font=("Helvetica", 12), wraplength=480, bg="white")
        self.loyalty_label.pack(pady=5)

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

                # Combine the name and mana cost
                name_and_mana = f"{card_data['name']}  {card_data['mana_cost']}"
                self.name_mana_label.config(text=name_and_mana)

                # Combine the type and rarity
                type_and_rarity = f"{card_data['type_line']} | {card_data['rarity']}"
                self.type_rarity_label.config(text=type_and_rarity)

                self.theme_label.config(text=f"Theme: {card_data['theme']}")

                # Display the oracle text with respect to newlines
                oracle_text = card_data['oracle_text'].replace("\\n", "\n")
                self.oracle_text_box.config(state=tk.NORMAL)
                self.oracle_text_box.delete(1.0, tk.END)
                self.oracle_text_box.insert(tk.END, oracle_text)
                self.oracle_text_box.config(state=tk.DISABLED)

                # Display the flavor text with respect to newlines
                if 'flavor_text' in card_data:
                    flavor_text = card_data['flavor_text'].replace("\\n", "\n")
                    self.flavor_text_label.config(text="Flavor Text:")
                    self.flavor_text_box.config(state=tk.NORMAL)
                    self.flavor_text_box.delete(1.0, tk.END)
                    self.flavor_text_box.insert(tk.END, flavor_text)
                    self.flavor_text_box.config(state=tk.DISABLED)
                else:
                    self.flavor_text_label.config(text="")
                    self.flavor_text_box.config(state=tk.NORMAL)
                    self.flavor_text_box.delete(1.0, tk.END)
                    self.flavor_text_box.config(state=tk.DISABLED)

                if card_data.get('power') and card_data.get('toughness'):
                    self.pt_label.config(text=f"Power/Toughness: {card_data['power']}/{card_data['toughness']}")
                else:
                    self.pt_label.config(text="")

                if card_data.get('loyalty'):
                    self.loyalty_label.config(text=f"Loyalty: {card_data['loyalty']}")
                else:
                    self.loyalty_label.config(text="")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load card: {e}")

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
    app = MagicCardViewer(root, folder_path)
    root.mainloop()
