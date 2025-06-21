import markdown_it
from bs4 import BeautifulSoup

class MarkdownProcessor:
    """
    Utility class for cleaning and converting markdown to plain text.
    """
    def __init__(self):            
        self.md_parser = markdown_it.MarkdownIt()
        self.BeautifulSoup = BeautifulSoup


    def to_plain_text(self, markdown_string: str) -> str:
        """
        Converts a markdown string to clean plain text by first converting to HTML
        and then stripping HTML tags. Handles nested/conflicting markdown syntax.
        """
        # Step 1: Clean and convert markdown to HTML using markdown-it-py
        # markdown-it-py follows CommonMark spec, robust for "ugly" markdown [12, 13]
        html = self.md_parser.render(markdown_string)

        # Step 2: Strip HTML tags to get plain text using BeautifulSoup [15]
        soup = self.BeautifulSoup(html, "html.parser")

        # Remove script and style elements to avoid extracting unwanted text
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Get text, handling potential code blocks or other special elements
        text = soup.get_text()

        # Further cleanup: remove extra whitespace and newlines
        text = ' '.join(text.split())
        return text