import os
import re

def natural_sort_key(s):
    """
    Key for natural sorting. E.g., "Lecture_2.md" before "Lecture_10.md".
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

def update_summary_file():
    """
    Adds entries for all existing Lecture_X.md files into SUMMARY.md.
    """
    summary_filepath = "SUMMARY.md"

    try:
        # 1. List all files and filter for Lecture_*.md
        all_files = os.listdir('.')
        lecture_files = [f for f in all_files if re.match(r'^Lecture_([0-9]+)\.md$', f)]

        # 2. Sort these files naturally
        lecture_files.sort(key=natural_sort_key)

        if not lecture_files:
            print("No Lecture_X.md files found to add to SUMMARY.md.")
            return

        # 3. Read the current content of SUMMARY.md
        if os.path.exists(summary_filepath):
            with open(summary_filepath, 'r', encoding='utf-8') as f:
                summary_content = f.read()
        else:
            # This case should ideally not happen based on previous steps,
            # but as a fallback, initialize with default content.
            print(f"{summary_filepath} not found. Creating a new one.")
            summary_content = "# Summary\n\n* [Introduction](introduction.md)\n"

        # Ensure there's a newline at the end if not already present,
        # before adding new lines.
        if not summary_content.endswith('\n'):
            summary_content += '\n'

        # 4. Append formatted links for each lecture file
        new_entries = []
        for lecture_file in lecture_files:
            # Extract lecture number for the title
            match = re.match(r'^Lecture_([0-9]+)\.md$', lecture_file)
            if match:
                lecture_number = match.group(1)
                # Format: * [Lecture X](Lecture_X.md)
                new_entries.append(f"* [Lecture {lecture_number}]({lecture_file})")

        summary_content += "\n".join(new_entries)
        summary_content += "\n" # Add a final newline

        # 5. Write the updated content back to SUMMARY.md
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        print(f"Successfully updated {summary_filepath} with {len(lecture_files)} lecture entries.")

    except Exception as e:
        print(f"Error updating {summary_filepath}: {e}")

if __name__ == "__main__":
    update_summary_file()
