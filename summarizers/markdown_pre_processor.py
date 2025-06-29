import re
import markdown2
import html
from bs4 import BeautifulSoup

class MarkdownPreprocessor:
    @staticmethod
    def markdown_to_clean_text(markdown: str) -> str:
        """Main method: convert markdown to clean plain text."""
        try:
            html_text = markdown2.markdown(markdown)
            plain_text = MarkdownPreprocessor.strip_html_tags(html_text)
            plain_text = MarkdownPreprocessor.normalize_whitespace(plain_text)
            plain_text = MarkdownPreprocessor.decode_html_entities(plain_text)
            return plain_text
        except Exception as e:
            print(f"Error processing Markdown: {e}")
            return ""

    @staticmethod
    def strip_html_tags(html_text: str) -> str:
        """Remove or transform non-paragraph HTML elements."""
        soup = BeautifulSoup(html_text, "html.parser")

        # Remove unwanted tags
        for tag in soup(["a", "img", "script", "style", "code", "pre"]):
            tag.decompose()

        # Preserve headers as plain text
        for tag in soup(["h1", "h2", "h3", "h4", "h5", "h6"]):
            tag.insert(0, soup.new_string(f"### {tag.name.upper()} "))
            tag.unwrap()

        # Transform lists to plain text
        for ul in soup.find_all("ul"):
            for li in ul.find_all("li"):
                li.insert(0, soup.new_string("- "))
                li.append(soup.new_string("\n"))
        for ol in soup.find_all("ol"):
            for i, li in enumerate(ol.find_all("li"), 1):
                li.insert(0, soup.new_string(f"{i}. "))
                li.append(soup.new_string("\n"))

        # Handle tables
        for table in soup.find_all("table"):
            table.string = " [Table content] "

        return soup.get_text(separator="\n")

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Collapse excessive whitespace while preserving structure."""
        text = re.sub(r'[ \t]+', ' ', text)  # Collapse spaces/tabs
        text = re.sub(r'\n\s*\n+', '\n\n', text)  # Normalize paragraph spacing
        text = re.sub(r'\n\s*([^\n])', r'\n\1', text)  # Preserve single newlines
        return text.strip()

    @staticmethod
    def decode_html_entities(text: str) -> str:
        """Convert HTML entities to unicode."""
        return html.unescape(text)