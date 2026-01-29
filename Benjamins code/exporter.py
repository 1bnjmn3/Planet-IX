import os

# --- Configuration ---
folder_path = './Benjamins code'
output_file = 'full_codebase_numbered.txt'
allowed_extensions = {'.py', '.dat', '.csv'}
ignored_files = {'full_codebase_numbered.txt', 'build_context_numbered.py', '.DS_Store'}

def build_context():
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(folder_path):
            # Skip hidden folders
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_ext = os.path.splitext(file)[1]
                
                if file_ext in allowed_extensions and file not in ignored_files:
                    file_path = os.path.join(root, file)
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            lines = infile.readlines()
                            
                            # 1. Header identifying the file
                            outfile.write(f"\n{'='*20}\n")
                            outfile.write(f"FILE PATH: {file_path}\n")
                            outfile.write(f"{'='*20}\n")
                            
                            # 2. Markdown Code Block Start
                            # We detect extension to set language hint (e.g., python vs csv)
                            lang_hint = 'python' if file_ext == '.py' else ''
                            outfile.write(f"```{lang_hint}\n")
                            
                            # 3. Content with Line Numbers
                            for i, line in enumerate(lines):
                                # Format:  1 | import os
                                outfile.write(f"{i+1:4d} | {line}")
                            
                            # Ensure newline if file doesn't end with one
                            if lines and not lines[-1].endswith('\n'):
                                outfile.write("\n")
                                
                            # 4. Markdown Code Block End
                            outfile.write("```\n\n")
                            
                        print(f"Processed: {file_path}")
                        
                    except Exception as e:
                        print(f"Skipped {file_path}: {e}")

if __name__ == "__main__":
    build_context()
    print(f"\nSuccess! Output saved to: {output_file}")