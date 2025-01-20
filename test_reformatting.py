import os
import sys
import re
import argparse
from main import reformat_markdown_with_claude

def test_reformatting(markdown_file_path, max_words=None):
    # Check if the file exists
    if not os.path.exists(markdown_file_path):
        print(f"Markdown file '{markdown_file_path}' does not exist.")
        sys.exit(1)
    
    # Read the markdown content from the file
    with open(markdown_file_path, 'r', encoding='utf-8') as f:
        original_md = f.read()
    
    # Truncate the markdown content to the specified number of words
    if max_words is not None:
        # Split the content into words and truncate
        words = re.findall(r'\S+', original_md)
        truncated_words = words[:max_words]
        original_md = ' '.join(truncated_words)

    # Call the reformatting function
    reformatted_md = reformat_markdown_with_claude(original_md)
    
    # Print the results
    print('Original Markdown:')
    print(original_md)
    print('\nReformatted Markdown:')
    print(reformatted_md)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test reformatting of markdown output')
    parser.add_argument('--markdown_file', required=True, help='Path to the markdown file to reformat')
    parser.add_argument('--max_words', type=int, default=None, help='Maximum number of words to include from the markdown file')
    args = parser.parse_args()
    
    test_reformatting(args.markdown_file, args.max_words)
