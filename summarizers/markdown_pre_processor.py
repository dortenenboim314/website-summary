import re
import markdown2
import html
from bs4 import BeautifulSoup

class MarkdownPreprocessor:

    @staticmethod
    def markdown_to_clean_text(markdown: str) -> str:
        """Main method: convert markdown to clean plain text."""
        html_text = markdown2.markdown(markdown)
        plain_text = MarkdownPreprocessor.strip_html_tags(html_text)
        plain_text = MarkdownPreprocessor.normalize_whitespace(plain_text)
        plain_text = MarkdownPreprocessor.decode_html_entities(plain_text)
        return plain_text

    @staticmethod
    def strip_html_tags(html_text: str) -> str:
        """Remove non-paragraph HTML elements like headers, links, images."""
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove headers, links, images
        for tag in soup(["a", "img", "h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.decompose()

        return soup.get_text(separator="\n")

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse multiple spaces/newlines into clean paragraphs."""
        text = re.sub(r'\s+', ' ', text)             # collapse all whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)     # normalize paragraph spacing
        return text.strip()

    @staticmethod
    def decode_html_entities(text: str) -> str:
        """Convert HTML entities to unicode (e.g., &amp; → &)"""
        return html.unescape(text)
