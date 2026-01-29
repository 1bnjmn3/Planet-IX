import os

# --- CONFIGURATION ---
# Points to: /Users/[YourName]/Desktop/VS_Planet IX/Benjamins code
SOURCE_DIR = os.path.expanduser("~/Desktop/VS_Planet IX/Benjamins code")
OUTPUT_FILE = os.path.join(SOURCE_DIR, "combined_files.txt")
EXTENSION = ".txt"

def fuse_files():
    # Check if directory exists first
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Could not find folder at: {SOURCE_DIR}")
        return

    # Get list of .txt files
    files_found = [f for f in os.listdir(SOURCE_DIR) if f.endswith(EXTENSION)]
    
    # Filter out the output file so it doesn't try to read itself
    files_found = [f for f in files_found if f != os.path.basename(OUTPUT_FILE)]
    
    # Sort files alphabetically so they are in order
    files_found.sort()

    print(f"Found {len(files_found)} files in '{SOURCE_DIR}'. Merging...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for filename in files_found:
            filepath = os.path.join(SOURCE_DIR, filename)
            
            # --- HEADER CREATION ---
            # Make the header stand out nicely
            header = f"\n{'='*40}\nFILE: {filename}\n{'='*40}\n"
            outfile.write(header)
            
            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as infile:
                    content = infile.read()
                    outfile.write(content)
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")
                outfile.write(f"[ERROR READING FILE: {e}]")

            # Add two newlines between files for spacing
            outfile.write("\n\n")

    print(f"Success! Combined file saved to:\n{OUTPUT_FILE}")

if __name__ == "__main__":
    fuse_files()