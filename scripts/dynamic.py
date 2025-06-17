import time
import random
import json
import os
from typing import List, Dict
import logging
from tavily import TavilyClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DynamicEvaluationDatasetCreator:
    def __init__(self, output_folder: str = "evaluation_dataset"):
        self.output_folder = output_folder
        self.client = TavilyClient()
        self.datasets_by_type = {}
        self.successful_extractions = 0
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_search_configurations(self) -> Dict[str, Dict]:
        """Define search terms and domains for each content type and language"""
        
        return {
            "news": {
                "english": {
                    "domains": ["bbc.com", "reuters.com", "cnn.com", "apnews.com", "theguardian.com"],
                    "search_terms": [
                        "artificial intelligence breakthrough",
                        "climate change research",
                        "space exploration discovery",
                        "medical research breakthrough",
                        "technology innovation",
                        "cybersecurity threats",
                        "renewable energy development",
                        "scientific discovery"
                    ]
                },
                "spanish": {
                    "domains": ["elpais.com", "eluniversal.com.mx", "clarin.com", "elmundo.es"],
                    "search_terms": [
                        "inteligencia artificial",
                        "cambio climático",
                        "exploración espacial",
                        "investigación médica",
                        "innovación tecnológica",
                        "ciberseguridad",
                        "energía renovable",
                        "descubrimiento científico"
                    ]
                },
                "french": {
                    "domains": ["lemonde.fr", "liberation.fr", "lefigaro.fr", "francetvinfo.fr"],
                    "search_terms": [
                        "intelligence artificielle",
                        "changement climatique",
                        "exploration spatiale",
                        "recherche médicale",
                        "innovation technologique",
                        "cybersécurité",
                        "énergie renouvelable",
                        "découverte scientifique"
                    ]
                },
                "chinese": {
                    "domains": ["sina.com.cn", "163.com", "sohu.com", "xinhuanet.com"],
                    "search_terms": [
                        "人工智能",
                        "气候变化",
                        "太空探索",
                        "医学研究",
                        "技术创新",
                        "网络安全",
                        "可再生能源",
                        "科学发现"
                    ]
                },
                "german": {
                    "domains": ["spiegel.de", "zeit.de", "faz.net", "sueddeutsche.de"],
                    "search_terms": [
                        "künstliche intelligenz",
                        "klimawandel",
                        "weltraumforschung",
                        "medizinische forschung",
                        "technische innovation",
                        "cybersicherheit",
                        "erneuerbare energie",
                        "wissenschaftliche entdeckung"
                    ]
                }
            },
            
            "github": {
                "english": {  # GitHub is primarily English
                    "domains": ["github.com"],
                    "search_terms": [
                        "machine learning framework README",
                        "web development framework README",
                        "data science library README",
                        "cloud computing tools README",
                        "cybersecurity tools README",
                        "mobile app development README",
                        "blockchain project README",
                        "AI model implementation README",
                        "database management README",
                        "DevOps automation README"
                    ]
                }
            },
            
            "reddit": {
                "english": {
                    "domains": ["reddit.com"],
                    "search_terms": [
                        "site:reddit.com/r/MachineLearning breakthrough",
                        "site:reddit.com/r/programming best practices",
                        "site:reddit.com/r/science latest research",
                        "site:reddit.com/r/technology innovation",
                        "site:reddit.com/r/datascience career advice",
                        "site:reddit.com/r/webdev frameworks comparison",
                        "site:reddit.com/r/cybersecurity threats analysis",
                        "site:reddit.com/r/artificial intelligence discussion"
                    ]
                },
                "spanish": {
                    "domains": ["reddit.com"],
                    "search_terms": [
                        "site:reddit.com/r/es tecnología",
                        "site:reddit.com/r/spain programación",
                        "site:reddit.com/r/mexico tecnologia",
                        "site:reddit.com/r/argentina tecnologia"
                    ]
                },
                "french": {
                    "domains": ["reddit.com"],
                    "search_terms": [
                        "site:reddit.com/r/france technologie",
                        "site:reddit.com/r/Quebec technologie",
                        "site:reddit.com/r/francophonie programmation"
                    ]
                }
            },
            
            "blogs": {  # Adding tech blogs as bonus content type
                "english": {
                    "domains": ["medium.com", "dev.to", "hashnode.com", "substack.com"],
                    "search_terms": [
                        "artificial intelligence tutorial",
                        "machine learning implementation",
                        "web development guide",
                        "data science analysis",
                        "software engineering practices",
                        "cloud computing architecture",
                        "cybersecurity analysis",
                        "technology trends"
                    ]
                },
                "spanish": {
                    "domains": ["medium.com"],
                    "search_terms": [
                        "inteligencia artificial tutorial",
                        "desarrollo web guía",
                        "ciencia de datos",
                        "programación práticas"
                    ]
                }
            }
        }
    
    def search_and_extract_content(self, content_type: str, language: str, max_items_per_type: int = 5) -> List[Dict]:
        """Search for content and extract using Tavily"""
        
        search_configs = self.get_search_configurations()
        
        if content_type not in search_configs:
            logger.warning(f"No search configuration for content type: {content_type}")
            return []
        
        if language not in search_configs[content_type]:
            logger.warning(f"No search configuration for {content_type} in {language}")
            return []
        
        config = search_configs[content_type][language]
        domains = config["domains"]
        search_terms = config["search_terms"]
        
        extracted_items = []
        seen_urls = set()
        
        # Shuffle search terms for variety
        shuffled_terms = search_terms.copy()
        random.shuffle(shuffled_terms)
        
        for search_term in shuffled_terms:
            if len(extracted_items) >= max_items_per_type:
                break
                
            try:
                logger.info(f"Searching {content_type} ({language}): '{search_term}' in {domains}")
                
                # Search with Tavily
                response = self.client.search(
                    query=search_term,
                    include_raw_content=True,
                    include_domains=domains,
                    max_results=3,  # Get a few results per search
                    search_depth="basic"
                )
                
                for result in response.get('results', []):
                    if len(extracted_items) >= max_items_per_type:
                        break
                    
                    url = result.get("url", "")
                    raw_content = result.get("raw_content", "")
                    title = result.get("title", "")
                    
                    # Skip if no content or already seen
                    if raw_content is None or url in seen_urls:
                        continue
                    
                    # Apply content length filters based on type
                    min_length = self._get_min_length_for_type(content_type)
                    if len(raw_content.strip()) < min_length:
                        logger.info(f"Content too short for {url} ({len(raw_content)} chars), skipping")
                        continue
                    
                    # Create data entry
                    data_entry = {
                        "url": url,
                        "title": title,
                        "type": content_type,
                        "language": language,
                        "search_term": search_term,
                        "domain": self._extract_domain(url),
                        "markdown": raw_content.strip(),
                        "extracted_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "content_length": len(raw_content.strip()),
                        "word_count": len(raw_content.split()),
                        "estimated_read_time": max(1, len(raw_content.split()) // 200)  # ~200 words per minute
                    }
                    
                    extracted_items.append(data_entry)
                    seen_urls.add(url)
                    
                    logger.info(f"Extracted {content_type} ({language}): {title[:50]}... ({len(raw_content)} chars)")
                
                # Small delay between searches
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error searching for '{search_term}': {str(e)}")
                continue
        
        logger.info(f"Found {len(extracted_items)} {content_type} items in {language}")
        return extracted_items
    
    def _get_min_length_for_type(self, content_type: str) -> int:
        """Get minimum content length based on content type"""
        min_lengths = {
            "news": 400,      # News articles should be substantial
            "github": 300,    # READMEs should have good content
            "reddit": 150,    # Reddit posts can be shorter
            "blogs": 350      # Blog posts should be substantial
        }
        return min_lengths.get(content_type, 200)
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def create_evaluation_dataset(self, target_count: int = 50):
        """Main method to create the dynamic evaluation dataset"""
        logger.info(f"Creating dynamic evaluation dataset with {target_count} items...")
        
        # Define target distribution
        type_language_targets = {
            ("news", "english"): 8,
            ("news", "spanish"): 4,
            ("news", "french"): 3,
            ("news", "chinese"): 2,
            ("news", "german"): 2,
            ("github", "english"): 12,  # GitHub is primarily English
            ("reddit", "english"): 8,
            ("reddit", "spanish"): 2,
            ("reddit", "french"): 2,
            ("blogs", "english"): 5,
            ("blogs", "spanish"): 2
        }
        
        # Ensure targets sum to target_count
        total_planned = sum(type_language_targets.values())
        if total_planned != target_count:
            # Adjust English news to match target
            diff = target_count - total_planned
            type_language_targets[("news", "english")] += diff
        
        logger.info("TARGET DISTRIBUTION:")
        for (content_type, language), target in type_language_targets.items():
            logger.info(f"  {content_type.capitalize()} ({language}): {target} items")
        
        # Extract content for each type-language combination
        all_extracted_items = []
        
        for (content_type, language), target_count_for_combo in type_language_targets.items():
            logger.info(f"\n--- Extracting {content_type} content in {language} (target: {target_count_for_combo}) ---")
            
            items = self.search_and_extract_content(
                content_type=content_type,
                language=language,
                max_items_per_type=target_count_for_combo * 2  # Get extra as backup
            )
            
            # Take only the target number (best quality items first)
            selected_items = items[:target_count_for_combo]
            all_extracted_items.extend(selected_items)
            
            # Add to type-specific dataset
            if content_type not in self.datasets_by_type:
                self.datasets_by_type[content_type] = []
            self.datasets_by_type[content_type].extend(selected_items)
            
            logger.info(f"Added {len(selected_items)} {content_type} items in {language}")
            
            # Delay between different type-language combinations
            time.sleep(random.uniform(2, 4))
        
        self.successful_extractions = len(all_extracted_items)
        
        # Save datasets
        self._save_evaluation_dataset(all_extracted_items)
        self._create_evaluation_readme()
        
        # Print summary
        self._print_summary(type_language_targets)
    
    def _save_evaluation_dataset(self, all_items: List[Dict]):
        """Save the evaluation dataset in multiple formats"""
        
        # Create unified evaluation dataset
        evaluation_data = {
            "metadata": {
                "dataset_type": "dynamic_evaluation_mixed_content",
                "total_items": len(all_items),
                "extraction_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "extraction_method": "tavily_dynamic_search",
                "content_types": {},
                "languages": {},
                "domains": {},
                "avg_content_length": 0,
                "avg_word_count": 0,
                "avg_read_time": 0
            },
            "items": all_items
        }
        
        # Calculate statistics
        if all_items:
            total_length = sum(item['content_length'] for item in all_items)
            total_words = sum(item['word_count'] for item in all_items)
            total_read_time = sum(item['estimated_read_time'] for item in all_items)
            
            evaluation_data["metadata"]["avg_content_length"] = total_length // len(all_items)
            evaluation_data["metadata"]["avg_word_count"] = total_words // len(all_items)
            evaluation_data["metadata"]["avg_read_time"] = total_read_time // len(all_items)
            
            # Count by type, language, and domain
            for item in all_items:
                content_type = item['type']
                language = item['language']
                domain = item['domain']
                
                evaluation_data["metadata"]["content_types"][content_type] = evaluation_data["metadata"]["content_types"].get(content_type, 0) + 1
                evaluation_data["metadata"]["languages"][language] = evaluation_data["metadata"]["languages"].get(language, 0) + 1
                evaluation_data["metadata"]["domains"][domain] = evaluation_data["metadata"]["domains"].get(domain, 0) + 1
        
        # Save unified dataset
        unified_path = os.path.join(self.output_folder, "evaluation_dataset.json")
        with open(unified_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        # Save type-specific files
        for content_type, items in self.datasets_by_type.items():
            if items:
                type_data = {
                    "metadata": {
                        "content_type": content_type,
                        "item_count": len(items),
                        "extraction_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                        "languages": list(set(item['language'] for item in items)),
                        "domains": list(set(item['domain'] for item in items))
                    },
                    "items": items
                }
                
                type_path = os.path.join(self.output_folder, f"evaluation_{content_type}.json")
                with open(type_path, 'w', encoding='utf-8') as f:
                    json.dump(type_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved evaluation datasets to {self.output_folder}/")
    
    def _create_evaluation_readme(self):
        """Create README for evaluation dataset"""
        readme_content = f"""# Dynamic Evaluation Dataset for Markdown Summarization

This evaluation dataset contains fresh, real-world content dynamically sourced using Tavily search across multiple languages and content types.

## Dataset Creation Method

Instead of hardcoded URLs, this dataset uses **dynamic search**:
1. Predefined search terms for each content type and language
2. Targeted domain searches (e.g., BBC for English news, Sina for Chinese news)
3. Real-time content extraction with quality filtering
4. Fresh content that changes with each generation

## Content Distribution

- **News Articles**: From major news sites in multiple languages
- **GitHub READMEs**: Documentation from popular open-source projects  
- **Reddit Discussions**: High-quality tech discussions and Q&A
- **Blog Posts**: Technical articles from Medium, Dev.to, etc.

## Languages Covered

- **English**: Primary language, most content types
- **Spanish**: News, Reddit, blogs
- **French**: News, Reddit discussions  
- **Chinese**: News articles from major Chinese sites
- **German**: News articles from German publications

## Files Structure

```
{self.output_folder}/
├── evaluation_dataset.json      # Complete unified dataset
├── evaluation_news.json         # News articles only
├── evaluation_github.json       # GitHub READMEs only  
├── evaluation_reddit.json       # Reddit discussions only
├── evaluation_blogs.json        # Blog posts only
└── README.md                    # This file
```

## Usage Examples

### Load Complete Dataset
```python
import json

with open('{self.output_folder}/evaluation_dataset.json', 'r', encoding='utf-8') as f:
    eval_data = json.load(f)

print(f"Total items: {{eval_data['metadata']['total_items']}}")
print(f"Content types: {{eval_data['metadata']['content_types']}}")
print(f"Languages: {{eval_data['metadata']['languages']}}")
print(f"Domains: {{eval_data['metadata']['domains']}}")

for item in eval_data['items']:
    print(f"{{item['type']}} ({{item['language']}}): {{item['title'][:50]}}...")
    print(f"Domain: {{item['domain']}}")
    print(f"Search term: {{item['search_term']}}")
    print(f"Length: {{item['content_length']}} chars")
    print("-" * 50)
```

### Filter by Content Type
```python
# Get only news articles
news_items = [item for item in eval_data['items'] if item['type'] == 'news']

# Get only English content
english_items = [item for item in eval_data['items'] if item['language'] == 'english']

# Get specific combinations
spanish_news = [item for item in eval_data['items'] 
                if item['type'] == 'news' and item['language'] == 'spanish']
```

### Analyze Content Quality
```python
# Sort by content length
sorted_items = sorted(eval_data['items'], key=lambda x: x['content_length'], reverse=True)

# Group by domain
from collections import defaultdict
by_domain = defaultdict(list)
for item in eval_data['items']:
    by_domain[item['domain']].append(item)

print("Content by domain:")
for domain, items in by_domain.items():
    avg_length = sum(item['content_length'] for item in items) // len(items)
    print(f"  {{domain}}: {{len(items)}} items, avg {{avg_length}} chars")
```

## Quality Metrics

- **Minimum lengths**: News (400+ chars), GitHub (300+ chars), Reddit (150+ chars), Blogs (350+ chars)
- **Fresh content**: Extracted using current search results
- **Domain variety**: Multiple authoritative sources per language
- **Search diversity**: Different search terms to avoid content clustering

## Evaluation Strategy

This dataset tests summarization models on:

1. **Domain Transfer**: Wikipedia → Real-world content
2. **Language Diversity**: Cross-lingual summarization capability  
3. **Content Variety**: Different markdown structures and writing styles
4. **Freshness**: Current content vs. static training data
5. **Source Diversity**: Multiple domains and publication styles

Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Total items: {self.successful_extractions}
"""
        
        readme_path = os.path.join(self.output_folder, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)
    
    def _print_summary(self, type_language_targets):
        """Print comprehensive summary"""
        logger.info("="*60)
        logger.info("DYNAMIC EVALUATION DATASET CREATION COMPLETED!")
        logger.info("="*60)
        logger.info(f"Successfully extracted: {self.successful_extractions} items")
        logger.info(f"Dataset folder: {self.output_folder}/")
        logger.info("")
        logger.info("FINAL DISTRIBUTION:")
        
        # Count actual distribution
        actual_distribution = {}
        all_items = []
        for items in self.datasets_by_type.values():
            all_items.extend(items)
        
        for item in all_items:
            key = (item['type'], item['language'])
            actual_distribution[key] = actual_distribution.get(key, 0) + 1
        
        for (content_type, language), target in type_language_targets.items():
            actual = actual_distribution.get((content_type, language), 0)
            logger.info(f"  {content_type.capitalize()} ({language}): {actual}/{target} items")
        
        logger.info("")
        logger.info("CONTENT TYPE TOTALS:")
        type_totals = {}
        for (content_type, language), count in actual_distribution.items():
            type_totals[content_type] = type_totals.get(content_type, 0) + count
        
        for content_type, total in type_totals.items():
            percentage = (total / self.successful_extractions) * 100 if self.successful_extractions > 0 else 0
            logger.info(f"  {content_type.capitalize()}: {total} items ({percentage:.1f}%)")
        
        logger.info("")
        logger.info("LANGUAGE TOTALS:")
        lang_totals = {}
        for (content_type, language), count in actual_distribution.items():
            lang_totals[language] = lang_totals.get(language, 0) + count
        
        for language, total in lang_totals.items():
            percentage = (total / self.successful_extractions) * 100 if self.successful_extractions > 0 else 0
            logger.info(f"  {language.capitalize()}: {total} items ({percentage:.1f}%)")
        
        logger.info("")
        logger.info("FILES CREATED:")
        logger.info(f"  📊 evaluation_dataset.json - Complete unified dataset")
        for content_type in self.datasets_by_type.keys():
            logger.info(f"  📄 evaluation_{content_type}.json - {content_type.capitalize()} content only")
        logger.info(f"  📖 README.md - Usage instructions and methodology")

def main():
    """Main function to run the dynamic evaluation dataset creator"""
    creator = DynamicEvaluationDatasetCreator()
    creator.create_evaluation_dataset(200)  # Target 200 evaluation items

if __name__ == "__main__":
    main()