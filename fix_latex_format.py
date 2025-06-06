import os
import re

def fix_latex_in_file(filepath):
    """
    Reads a file, replaces occurrences of `$$...$$` with $$...$$,
    and writes the modified content back to the file if changes were made.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return

    # Pattern to find `$$...$$`
    # Using a negative lookbehind and lookahead to avoid matching inside ``` blocks
    # This is a simplified approach and might not cover all edge cases with nested blocks.
    pattern = r'(?<!```\n)`\$\$(.*?)\$\$`(?!`\n```)'
    # Replacement pattern
    replacement = r'$$\1$$'

    new_content, num_replacements = re.subn(pattern, replacement, content, flags=re.DOTALL)

    if num_replacements > 0:
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Updated: {filepath}")
        except Exception as e:
            print(f"Error writing file {filepath}: {e}")
    else:
        print(f"No changes needed: {filepath}")

def main():
    """
    Walks through the current directory and its subdirectories,
    and calls fix_latex_in_file() for each Markdown file.
    """
    for root, _, files in os.walk('.'):
        for filename in files:
            if filename.endswith('.md'):
                filepath = os.path.join(root, filename)
                fix_latex_in_file(filepath)

if __name__ == "__main__":
    main()
