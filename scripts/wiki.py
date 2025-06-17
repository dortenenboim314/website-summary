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

class WikipediaMarkdownDatasetCreator:
    def __init__(self, output_folder: str = "wikipedia_dataset"):
        self.output_folder = output_folder
        self.client = TavilyClient()
        self.datasets_by_language = {}
        self.successful_extractions = 0
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
    def get_wikipedia_articles(self) -> List[Dict[str, str]]:
        """Return a comprehensive list of Wikipedia articles across multiple languages"""
        
        # High-quality Wikipedia topics across various domains
        topics = {
            # Technology & Science
            "tech_science": [
                "Artificial_intelligence", "Machine_learning", "Deep_learning", "Neural_network",
                "Computer_vision", "Natural_language_processing", "Robotics", "Quantum_computing",
                "Blockchain", "Cryptocurrency", "Internet_of_things", "5G", "Cloud_computing",
                "Cybersecurity", "Data_science", "Big_data", "Virtual_reality", "Augmented_reality",
                "Nanotechnology", "Biotechnology", "Genetic_engineering", "CRISPR", "Bioinformatics"
            ],
            
            # Physics & Chemistry
            "physics_chemistry": [
                "Quantum_mechanics", "Relativity", "Thermodynamics", "Electromagnetism", 
                "Particle_physics", "Astrophysics", "Black_hole", "Dark_matter", "Dark_energy",
                "Organic_chemistry", "Inorganic_chemistry", "Physical_chemistry", "Biochemistry",
                "Materials_science", "Crystallography", "Spectroscopy", "Photonics"
            ],
            
            # Biology & Medicine
            "biology_medicine": [
                "Evolution", "Genetics", "DNA", "RNA", "Protein", "Cell_biology", "Molecular_biology",
                "Neuroscience", "Brain", "Nervous_system", "Immunology", "Microbiology", "Virology",
                "Cancer", "Alzheimer's_disease", "COVID-19", "Vaccine", "Antibiotic", "Medicine",
                "Pharmacology", "Anatomy", "Physiology", "Pathology", "Epidemiology"
            ],
            
            # Mathematics
            "mathematics": [
                "Calculus", "Linear_algebra", "Statistics", "Probability", "Differential_equations",
                "Number_theory", "Graph_theory", "Topology", "Geometry", "Algebra", "Logic",
                "Mathematical_analysis", "Discrete_mathematics", "Combinatorics"
            ],
            
            # Energy & Environment
            "energy_environment": [
                "Climate_change", "Global_warming", "Renewable_energy", "Solar_power", "Wind_power",
                "Nuclear_power", "Fossil_fuel", "Carbon_cycle", "Greenhouse_effect", "Sustainability",
                "Ecology", "Biodiversity", "Conservation_biology", "Environmental_science",
                "Pollution", "Deforestation", "Ocean_acidification"
            ],
            
            # Space & Astronomy
            "space_astronomy": [
                "Solar_system", "Milky_Way", "Galaxy", "Universe", "Big_Bang", "Exoplanet",
                "Mars", "Moon", "Sun", "Star", "Supernova", "Neutron_star", "Space_exploration",
                "International_Space_Station", "Hubble_Space_Telescope", "James_Webb_Space_Telescope"
            ],
            
            # Social Sciences
            "social_sciences": [
                "Psychology", "Sociology", "Anthropology", "Economics", "Political_science",
                "Philosophy", "Ethics", "Logic", "Epistemology", "Metaphysics", "Linguistics",
                "Cognitive_science", "Behavioral_economics", "Game_theory"
            ],
            
            # History & Culture
            "history_culture": [
                "World_War_II", "Renaissance", "Industrial_Revolution", "Ancient_Egypt",
                "Roman_Empire", "Byzantine_Empire", "Ottoman_Empire", "History_of_China",
                "History_of_India", "Cold_War", "French_Revolution", "American_Revolution",
                "Civilization", "Culture", "Religion", "Art", "Literature", "Music"
            ]
        }
        
        # Language configurations
        languages = {
            "english": {"code": "en", "weight": 0.4},  # 40%
            "spanish": {"code": "es", "weight": 0.2},   # 20%
            "french": {"code": "fr", "weight": 0.15},   # 15%
            "german": {"code": "de", "weight": 0.1},    # 10%
            "chinese": {"code": "zh", "weight": 0.1},   # 10%
            "arabic": {"code": "ar", "weight": 0.05}    # 5%
        }
        
        # Language-specific topic translations
        topic_translations = {
            "spanish": {
                "Artificial_intelligence": "Inteligencia_artificial",
                "Machine_learning": "Aprendizaje_automático",
                "Climate_change": "Cambio_climático",
                "Quantum_mechanics": "Mecánica_cuántica",
                "Evolution": "Evolución_biológica",
                "Psychology": "Psicología",
                "Mathematics": "Matemáticas",
                "Physics": "Física",
                "Chemistry": "Química",
                "Biology": "Biología",
                "History": "Historia",
                "Philosophy": "Filosofía",
                "Economics": "Economía",
                "Medicine": "Medicina",
                "Computer_science": "Ciencias_de_la_computación"
            },
            "french": {
                "Artificial_intelligence": "Intelligence_artificielle",
                "Machine_learning": "Apprentissage_automatique",
                "Climate_change": "Réchauffement_climatique",
                "Quantum_mechanics": "Mécanique_quantique",
                "Evolution": "Évolution_(biologie)",
                "Psychology": "Psychologie",
                "Mathematics": "Mathématiques",
                "Physics": "Physique",
                "Chemistry": "Chimie",
                "Biology": "Biologie",
                "History": "Histoire",
                "Philosophy": "Philosophie",
                "Economics": "Économie",
                "Medicine": "Médecine"
            },
            "german": {
                "Artificial_intelligence": "Künstliche_Intelligenz",
                "Machine_learning": "Maschinelles_Lernen",
                "Climate_change": "Klimawandel",
                "Quantum_mechanics": "Quantenmechanik",
                "Evolution": "Evolution",
                "Psychology": "Psychologie",
                "Mathematics": "Mathematik",
                "Physics": "Physik",
                "Chemistry": "Chemie",
                "Biology": "Biologie",
                "History": "Geschichte",
                "Philosophy": "Philosophie",
                "Economics": "Wirtschaftswissenschaft",
                "Medicine": "Medizin"
            },
            "chinese": {
                "Artificial_intelligence": "人工智能",
                "Machine_learning": "机器学习",
                "Climate_change": "气候变化",
                "Quantum_mechanics": "量子力学",
                "Evolution": "演化",
                "Psychology": "心理学",
                "Mathematics": "数学",
                "Physics": "物理学",
                "Chemistry": "化学",
                "Biology": "生物学",
                "History": "历史",
                "Philosophy": "哲学",
                "Economics": "经济学",
                "Medicine": "医学"
            },
            "arabic": {
                "Artificial_intelligence": "ذكاء_اصطناعي",
                "Machine_learning": "تعلم_الآلة",
                "Climate_change": "تغير_المناخ",
                "Quantum_mechanics": "ميكانيكا_الكم",
                "Evolution": "تطور",
                "Psychology": "علم_النفس",
                "Mathematics": "رياضيات",
                "Physics": "فيزياء",
                "Chemistry": "كيمياء",
                "Biology": "أحياء",
                "History": "تاريخ",
                "Philosophy": "فلسفة",
                "Economics": "اقتصاد",
                "Medicine": "طب"
            }
        }
        
        urls = []
        
        # Create comprehensive list of all topics
        all_topics = []
        for category_topics in topics.values():
            all_topics.extend(category_topics)
        
        # Generate URLs for each language according to weights
        for language, config in languages.items():
            lang_code = config["code"]
            
            # Determine how many articles for this language
            # We'll generate extra and then sample from them
            topics_for_lang = all_topics.copy()
            
            # Add translations if available
            if language in topic_translations:
                translated_topics = list(topic_translations[language].values())
                topics_for_lang.extend(translated_topics)
            
            # Remove duplicates and shuffle
            topics_for_lang = list(set(topics_for_lang))
            random.shuffle(topics_for_lang)
            
            # Generate URLs for this language
            for topic in topics_for_lang:
                url = f"https://{lang_code}.wikipedia.org/wiki/{topic}"
                
                # Determine category (simplified categorization)
                category = "general"
                for cat_name, cat_topics in topics.items():
                    if any(base_topic in topic for base_topic in cat_topics):
                        category = cat_name
                        break
                
                urls.append({
                    "url": url,
                    "category": category,
                    "language": language,
                    "topic": topic
                })
        
        # Shuffle to mix languages and topics
        random.shuffle(urls)
        
        logger.info(f"Generated {len(urls)} Wikipedia URLs across {len(languages)} languages")
        
        return urls
    
    def extract_batch(self, url_batch: List[Dict[str, str]], batch_size: int = 5) -> List[Dict]:
        """Extract content from a batch of URLs using Tavily"""
        try:
            # Prepare URLs for Tavily
            urls = [item['url'] for item in url_batch]
            
            logger.info(f"Extracting batch of {len(urls)} Wikipedia articles...")
            
            # Call Tavily extract API
            response = self.client.extract(urls=urls, format='markdown')
            
            # Process results
            results = []
            for i, result in enumerate(response.get('results', [])):
                url = result.get('url')
                raw_content = result.get('raw_content', '')
                
                # Find the corresponding metadata
                metadata = next((item for item in url_batch if item['url'] == url), {})
                
                # Skip if content is too short (Wikipedia articles should be substantial)
                if len(raw_content.strip()) < 500:  # Higher threshold for Wikipedia
                    logger.warning(f"Content too short for {url} ({len(raw_content)} chars), skipping")
                    continue
                
                # Create data entry
                data_entry = {
                    "url": url,
                    "category": metadata.get('category', 'unknown'),
                    "language": metadata.get('language', 'unknown'),
                    "topic": metadata.get('topic', 'unknown'),
                    "markdown": raw_content.strip(),
                    "extracted_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "content_length": len(raw_content.strip()),
                    "word_count": len(raw_content.split())
                }
                
                results.append(data_entry)
                logger.info(f"Successfully extracted {metadata.get('topic', 'unknown')} in {metadata.get('language', 'unknown')} ({len(raw_content)} chars)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error extracting batch: {str(e)}")
            return []
    
    def create_dataset(self, target_count: int = 200, batch_size: int = 5):
        """Main method to create the Wikipedia markdown dataset"""
        logger.info(f"Creating Wikipedia dataset with {target_count} articles using Tavily...")
        
        # Language distribution weights
        language_weights = {
            "english": 0.4,   # 40%
            "spanish": 0.2,   # 20%
            "french": 0.15,   # 15%
            "german": 0.1,    # 10%
            "chinese": 0.1,   # 10%
            "arabic": 0.05    # 5%
        }
        
        # Calculate target counts per language
        language_targets = {}
        for lang, weight in language_weights.items():
            language_targets[lang] = int(target_count * weight)
        
        # Adjust for rounding (ensure we hit exactly target_count)
        total_allocated = sum(language_targets.values())
        if total_allocated < target_count:
            # Add remaining to English
            language_targets["english"] += (target_count - total_allocated)
        
        logger.info("TARGET DISTRIBUTION:")
        for lang, target in language_targets.items():
            percentage = (target / target_count) * 100
            logger.info(f"  {lang.capitalize()}: {target} articles ({percentage:.1f}%)")
        
        # Get Wikipedia URLs organized by language
        all_urls_by_language = self._get_urls_by_language()
        
        # Create processing queue respecting language targets
        processing_queue = []
        language_counts = {lang: 0 for lang in language_weights.keys()}
        
        # Build balanced queue
        max_attempts_per_lang = {}
        for lang in language_weights.keys():
            available_urls = len(all_urls_by_language.get(lang, []))
            target = language_targets[lang]
            # Get 2x target as backup URLs for each language
            max_attempts_per_lang[lang] = min(available_urls, target * 2)
            logger.info(f"  {lang.capitalize()}: {available_urls} URLs available, targeting {target}, will attempt up to {max_attempts_per_lang[lang]}")
        
        # Interleave URLs from different languages to create balanced batches
        lang_iterators = {}
        for lang, urls in all_urls_by_language.items():
            random.shuffle(urls)  # Shuffle within each language
            lang_iterators[lang] = iter(urls[:max_attempts_per_lang[lang]])
        
        # Build processing queue by cycling through languages
        while len(processing_queue) < sum(max_attempts_per_lang.values()):
            batch_added = False
            for lang in language_weights.keys():
                if lang in lang_iterators:
                    try:
                        url_item = next(lang_iterators[lang])
                        processing_queue.append(url_item)
                        batch_added = True
                    except StopIteration:
                        # This language is exhausted
                        del lang_iterators[lang]
            
            if not batch_added:
                break  # All languages exhausted
        
        logger.info(f"Created processing queue with {len(processing_queue)} URLs")
        
        # Process in batches with language targets
        processed_urls = 0
        
        while self.successful_extractions < target_count and processed_urls < len(processing_queue):
            # Get next batch
            batch_end = min(processed_urls + batch_size, len(processing_queue))
            current_batch = processing_queue[processed_urls:batch_end]
            
            if not current_batch:
                break
            
            # Extract batch
            batch_results = self.extract_batch(current_batch, batch_size)
            
            # Add successful extractions to dataset, respecting language targets
            for result in batch_results:
                language = result['language']
                current_count = language_counts.get(language, 0)
                target_for_lang = language_targets.get(language, 0)
                
                # Check if we can still add articles for this language
                if current_count < target_for_lang and self.successful_extractions < target_count:
                    # Initialize language dataset if not exists
                    if language not in self.datasets_by_language:
                        self.datasets_by_language[language] = []
                    
                    # Add to language-specific dataset
                    self.datasets_by_language[language].append(result)
                    language_counts[language] += 1
                    self.successful_extractions += 1
                    logger.info(f"Added to {language} dataset: {result['topic']} - #{self.successful_extractions} ({language_counts[language]}/{target_for_lang} for {language})")
                else:
                    logger.info(f"Skipping {result['topic']} ({language}) - target reached ({current_count}/{target_for_lang})")
            
            processed_urls = batch_end
            
            # Check if all language targets are met
            all_targets_met = all(language_counts[lang] >= language_targets[lang] for lang in language_targets.keys())
            if all_targets_met:
                logger.info("All language targets reached!")
                break
            
            # Add delay between batches to respect rate limits
            if self.successful_extractions < target_count and processed_urls < len(processing_queue):
                delay = random.uniform(1, 3)
                progress_info = ", ".join([f"{lang}: {language_counts[lang]}/{language_targets[lang]}" for lang in language_targets.keys()])
                logger.info(f"Progress ({progress_info}) - Waiting {delay:.1f}s before next batch...")
                time.sleep(delay)
        
        # Create language-specific files and master index
        master_index = {
            "metadata": {
                "dataset_type": "wikipedia_markdown_split",
                "total_extracted": self.successful_extractions,
                "target_count": target_count,
                "extraction_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                "languages": {},
                "categories_global": {},
                "files": {},
                "avg_content_length": 0,
                "avg_word_count": 0,
                "total_content_length": 0,
                "total_word_count": 0
            }
        }
        
        all_articles = []
        for lang_articles in self.datasets_by_language.values():
            all_articles.extend(lang_articles)
        
        # Calculate global statistics
        if all_articles:
            total_length = sum(item['content_length'] for item in all_articles)
            total_words = sum(item['word_count'] for item in all_articles)
            
            master_index["metadata"]["avg_content_length"] = total_length // len(all_articles)
            master_index["metadata"]["avg_word_count"] = total_words // len(all_articles)
            master_index["metadata"]["total_content_length"] = total_length
            master_index["metadata"]["total_word_count"] = total_words
            
            # Count global categories
            for entry in all_articles:
                category = entry['category']
                master_index["metadata"]["categories_global"][category] = master_index["metadata"]["categories_global"].get(category, 0) + 1
        
        # Create individual language files
        for language, articles in self.datasets_by_language.items():
            if not articles:
                continue
                
            # Calculate language-specific statistics
            lang_total_length = sum(item['content_length'] for item in articles)
            lang_total_words = sum(item['word_count'] for item in articles)
            
            # Create language dataset
            language_data = {
                "metadata": {
                    "language": language,
                    "article_count": len(articles),
                    "extraction_date": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "avg_content_length": lang_total_length // len(articles),
                    "avg_word_count": lang_total_words // len(articles),
                    "total_content_length": lang_total_length,
                    "total_word_count": lang_total_words,
                    "categories": {},
                    "topics": []
                },
                "articles": articles
            }
            
            # Count categories and topics for this language
            for article in articles:
                category = article['category']
                topic = article['topic']
                language_data["metadata"]["categories"][category] = language_data["metadata"]["categories"].get(category, 0) + 1
                if topic not in language_data["metadata"]["topics"]:
                    language_data["metadata"]["topics"].append(topic)
            
            # Save language file
            filename = f"wikipedia_{language}.json"
            filepath = os.path.join(self.output_folder, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(language_data, f, indent=2, ensure_ascii=False)
            
            # Add to master index
            master_index["metadata"]["languages"][language] = {
                "article_count": len(articles),
                "filename": filename,
                "avg_content_length": lang_total_length // len(articles),
                "avg_word_count": lang_total_words // len(articles),
                "categories": language_data["metadata"]["categories"]
            }
            master_index["metadata"]["files"][filename] = {
                "language": language,
                "article_count": len(articles),
                "file_size_estimate": f"{lang_total_length // 1024}KB"
            }
            
            logger.info(f"Created {filename} with {len(articles)} articles ({lang_total_length//1024}KB)")
        
        # Save master index
        master_index_path = os.path.join(self.output_folder, "index.json")
        with open(master_index_path, 'w', encoding='utf-8') as f:
            json.dump(master_index, f, indent=2, ensure_ascii=False)
        
        # Create README file
        self._create_readme()
        
        # Print comprehensive summary
        logger.info("="*60)
        logger.info("WIKIPEDIA DATASET CREATION COMPLETED!")
        logger.info("="*60)
        logger.info(f"Successfully extracted: {self.successful_extractions} Wikipedia articles")
        logger.info(f"Dataset folder: {self.output_folder}/")
        logger.info(f"Total content: {master_index['metadata']['total_content_length']:,} characters")
        logger.info(f"Total words: {master_index['metadata']['total_word_count']:,} words")
        logger.info("")
        logger.info("FILES CREATED:")
        logger.info(f"  📋 index.json - Master index and statistics")
        logger.info(f"  📖 README.md - Usage instructions")
        for filename, info in master_index["metadata"]["files"].items():
            logger.info(f"  🌍 {filename} - {info['article_count']} articles ({info['file_size_estimate']})")
        logger.info("")
        logger.info("LANGUAGE DISTRIBUTION:")
        for lang, info in sorted(master_index['metadata']['languages'].items(), key=lambda x: x[1]['article_count'], reverse=True):
            percentage = (info['article_count'] / self.successful_extractions) * 100
            logger.info(f"  {lang.capitalize()}: {info['article_count']} articles ({percentage:.1f}%)")
        logger.info("")
    def _get_urls_by_language(self) -> Dict[str, List[Dict[str, str]]]:
        """Get Wikipedia URLs organized by language"""
        
        # Get all URLs from the original method
        all_urls = self.get_wikipedia_articles()
        
        # Organize by language
        urls_by_language = {}
        for url_item in all_urls:
            language = url_item['language']
            if language not in urls_by_language:
                urls_by_language[language] = []
            urls_by_language[language].append(url_item)
        
        # Log availability per language
        logger.info("URL AVAILABILITY BY LANGUAGE:")
        for lang, urls in urls_by_language.items():
            logger.info(f"  {lang.capitalize()}: {len(urls)} URLs available")
        
        return urls_by_language
    
    def _create_readme(self):
        """Create a README file explaining the dataset structure"""
        readme_content = f"""# Wikipedia Markdown Dataset

This dataset contains Wikipedia articles extracted in markdown format across multiple languages.

## Dataset Structure

```
{self.output_folder}/
├── index.json          # Master index with statistics and file information
├── README.md          # This file
├── wikipedia_english.json    # English articles
├── wikipedia_spanish.json    # Spanish articles  
├── wikipedia_french.json     # French articles
├── wikipedia_german.json     # German articles
├── wikipedia_chinese.json    # Chinese articles
└── wikipedia_arabic.json     # Arabic articles
```

## Usage

### Loading the Master Index
```python
import json

# Load master index to see overview
with open('{self.output_folder}/index.json', 'r', encoding='utf-8') as f:
    index = json.load(f)
    
print(f"Total articles: {{index['metadata']['total_extracted']}}")
print(f"Languages: {{list(index['metadata']['languages'].keys())}}")
```

### Loading Specific Language
```python
# Load English articles only
with open('{self.output_folder}/wikipedia_english.json', 'r', encoding='utf-8') as f:
    english_data = json.load(f)
    
articles = english_data['articles']
for article in articles[:3]:  # First 3 articles
    print(f"Title: {{article['topic']}}")
    print(f"Category: {{article['category']}}")
    print(f"Content length: {{article['content_length']}} chars")
    print(f"Content preview: {{article['markdown'][:200]}}...")
    print("-" * 50)
```

### Loading All Articles
```python
import os
import json

def load_all_articles(dataset_folder):
    all_articles = []
    
    # Get list of language files from index
    with open(os.path.join(dataset_folder, 'index.json'), 'r', encoding='utf-8') as f:
        index = json.load(f)
    
    # Load each language file
    for filename in index['metadata']['files'].keys():
        if filename.startswith('wikipedia_') and filename.endswith('.json'):
            filepath = os.path.join(dataset_folder, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                lang_data = json.load(f)
                all_articles.extend(lang_data['articles'])
    
    return all_articles

# Usage
articles = load_all_articles('{self.output_folder}')
print(f"Loaded {{len(articles)}} total articles")
```

## File Formats

### index.json
Contains metadata about the entire dataset:
- Total statistics (article count, content length, word count)
- Language distribution
- Category distribution  
- File information and sizes

### wikipedia_[language].json
Each language file contains:
- `metadata`: Language-specific statistics and information
- `articles`: Array of article objects

### Article Object Structure
```json
{{
    "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "category": "tech_science", 
    "language": "english",
    "topic": "Artificial_intelligence",
    "markdown": "# Artificial Intelligence\\n\\nArtificial intelligence...",
    "extracted_at": "2024-01-15 10:30:45",
    "content_length": 15420,
    "word_count": 2180
}}
```

## Dataset Statistics

- **Extraction Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Articles**: {self.successful_extractions}
- **Languages**: English, Spanish, French, German, Chinese, Arabic
- **Categories**: Technology & Science, Physics & Chemistry, Biology & Medicine, Mathematics, Energy & Environment, Space & Astronomy, Social Sciences, History & Culture

## Use Cases

- Training language models on multilingual content
- Cross-language information retrieval research
- Content summarization and analysis
- Educational content processing
- Multilingual NLP benchmarking

## Notes

- All content is extracted from Wikipedia and subject to Wikipedia's licensing
- Content quality varies by language and topic availability
- Some articles may be shorter in certain languages
- Markdown formatting preserves Wikipedia's structure including headers, lists, and links
"""
        
        readme_path = os.path.join(self.output_folder, "README.md")
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

def main():
    """Main function to run the Wikipedia dataset creator"""
    creator = WikipediaMarkdownDatasetCreator()
    creator.create_dataset(300)  # Target 300 Wikipedia articles

if __name__ == "__main__":
    main()