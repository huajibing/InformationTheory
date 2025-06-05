import re
import os

def convert_formulas_in_file(filepath):
    """
    Converts LaTeX math formula delimiters in a single Markdown file.
    1. Inline $formula$ to $$formula$$
    2. Single-line $$ formula $$ to multi-line.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content

        # 1. Convert inline formulas: $formula$ to $$formula$$
        # Pattern: (?<!\$)\$([^\$]+?)\$(?!\$)
        # Replace: $$\1$$
        # This regex ensures that it doesn't match $$...$$ (double dollar signs)
        # and correctly captures the content within single dollar signs.
        content = re.sub(r'(?<!\$)\$([^\$]+?)\$(?!\$)', r'$$\1$$', content)

        # 2. Convert single-line block formulas to multi-line:
        # Pattern: ^\s*\$\$([^\S\r\n]*)(.+?[^\S\r\n]*)\$\$\s*$
        # Replace: $$\n\2\n$$
        # This regex targets $$...$$ that are entirely on one line,
        # possibly with leading/trailing whitespace on that line.
        # It captures the actual formula content in group 2, stripping any
        # surrounding whitespace within the $$...$$ before placing it on its own line.

        # We need a custom replacement function for this to handle stripping whitespace
        # from the captured group and ensuring correct newline placement.
        def replace_single_line_block(match):
            # group(2) is the formula content itself, potentially with leading/trailing spaces.
            formula_content = match.group(2).strip()
            return f"$$\n{formula_content}\n$$"

        content = re.sub(r'^\s*\$\$([^\S\r\n]*)(.+?[^\S\r\n]*)\$\$\s*$', replace_single_line_block, content, flags=re.MULTILINE)

        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Processed and updated: {filepath}")
        else:
            print(f"No changes needed for: {filepath}")

    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")

def main():
    markdown_files = [f for f in os.listdir('.') if f.endswith('.md')]
    if not markdown_files:
        print("No Markdown files found in the current directory.")
        return

    for md_file in markdown_files:
        convert_formulas_in_file(md_file)

    print("\nConversion process finished.")

if __name__ == "__main__":
    main()
