import time
import random
import csv
import os
from typing import List, Dict, Tuple
import logging
from tavily import TavilyClient
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedDatasetCreator:
    def __init__(self, output_file: str = "unified_dataset.csv"):
        self.output_file = output_file
        self.client = TavilyClient()
        self.collected_items = []
        self.seen_urls = set()
        
    def get_target_distribution(self, total_items: int = 500) -> Dict[Tuple[str, str], int]:
        """Define target distribution by (source, language) - Include Wikipedia + Dynamic content"""
        
        # Language weights (40% English, 60% others uniform)
        english_ratio = 0.4
        other_languages = ['spanish', 'french', 'german', 'chinese', 'arabic']
        other_ratio_each = (1 - english_ratio) / len(other_languages)  # 12% each
        
        distribution = {}
        
        # Split between Wikipedia (30%) and dynamic content (70%)
        wikipedia_items = int(total_items * 0.3)
        dynamic_items = total_items - wikipedia_items
        
        # Wikipedia distribution
        english_wikipedia = int(wikipedia_items * english_ratio)
        other_wikipedia = wikipedia_items - english_wikipedia
        
        distribution[('wikipedia', 'english')] = english_wikipedia
        
        # Distribute other languages for Wikipedia
        other_wiki_each = other_wikipedia // len(other_languages)
        remainder = other_wikipedia % len(other_languages)
        
        for i, lang in enumerate(other_languages):
            count = other_wiki_each + (1 if i < remainder else 0)
            distribution[('wikipedia', lang)] = count
        
        # Dynamic content distribution  
        english_dynamic = int(dynamic_items * english_ratio)
        other_dynamic = dynamic_items - english_dynamic
        
        # English dynamic sources
        distribution[('news', 'english')] = int(english_dynamic * 0.35)  # 35% of English dynamic
        distribution[('github', 'english')] = int(english_dynamic * 0.25)  # 25% of English dynamic
        distribution[('reddit', 'english')] = int(english_dynamic * 0.2)   # 20% of English dynamic
        distribution[('blog', 'english')] = english_dynamic - distribution[('news', 'english')] - distribution[('github', 'english')] - distribution[('reddit', 'english')]
        
        # Other languages dynamic (mainly news, some reddit/blogs)
        other_each_dynamic = other_dynamic // len(other_languages)
        for lang in other_languages:
            if lang in ['spanish', 'french']:  # Spanish and French get some variety
                distribution[('news', lang)] = int(other_each_dynamic * 0.6)
                distribution[('reddit', lang)] = int(other_each_dynamic * 0.2)
                distribution[('blog', lang)] = other_each_dynamic - distribution[('news', lang)] - distribution[('reddit', lang)]
            else:  # German, Chinese, Arabic mainly news
                distribution[('news', lang)] = other_each_dynamic
        
        return distribution
    
    def get_wikipedia_urls(self, language: str, count: int) -> List[Dict[str, str]]:
        """Generate Wikipedia URLs for a specific language"""
        
        # High-quality topics
        base_topics = [
            "Artificial_intelligence", "Machine_learning", "Climate_change", "Quantum_mechanics",
            "Evolution", "Psychology", "Mathematics", "Physics", "Chemistry", "Biology",
            "Renewable_energy", "Space_exploration", "Neuroscience", "Robotics", "Blockchain",
            "Cybersecurity", "Nanotechnology", "Biotechnology", "Computer_vision", "Economics",
            "Philosophy", "History", "Medicine", "Genetics", "Astrophysics", "Ecology"
        ]
        
        # Language-specific translations
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
                "Biology": "Biología"
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
                "Biology": "Biologie"
            },
            "german": {
                "Artificial_intelligence": "Künstliche_Intelligenz",
                "Machine_learning": "Maschinelles_Lernen",
                "Climate_change": "Klimawandel",
                "Quantum_mechanics": "Quantenmechanik",
                "Psychology": "Psychologie",
                "Mathematics": "Mathematik",
                "Physics": "Physik", 
                "Chemistry": "Chemie",
                "Biology": "Biologie"
            },
            "chinese": {
                "Artificial_intelligence": "人工智能",
                "Machine_learning": "机器学习",
                "Climate_change": "气候变化",
                "Quantum_mechanics": "量子力学",
                "Psychology": "心理学",
                "Mathematics": "数学",
                "Physics": "物理学",
                "Chemistry": "化学",
                "Biology": "生物学"
            },
            "arabic": {
                "Artificial_intelligence": "ذكاء_اصطناعي",
                "Machine_learning": "تعلم_الآلة",
                "Climate_change": "تغير_المناخ",
                "Quantum_mechanics": "ميكانيكا_الكم",
                "Psychology": "علم_النفس",
                "Mathematics": "رياضيات",
                "Physics": "فيزياء",
                "Chemistry": "كيمياء",
                "Biology": "أحياء"
            }
        }
        
        # Get language code
        lang_codes = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 
            'german': 'de', 'chinese': 'zh', 'arabic': 'ar'
        }
        lang_code = lang_codes.get(language, 'en')
        
        # Prepare topics
        if language == 'english':
            topics = base_topics
        else:
            topics = list(topic_translations.get(language, {}).values())
            if len(topics) < count:
                topics.extend(base_topics[:count-len(topics)])  # Fill with English topics
        
        # Shuffle and take needed count
        random.shuffle(topics)
        selected_topics = topics[:count * 2]  # Get extra as backup
        
        urls = []
        for topic in selected_topics:
            url = f"https://{lang_code}.wikipedia.org/wiki/{topic}"
            urls.append({
                'url': url,
                'source': 'wikipedia',
                'language': language,
                'topic': topic
            })
        
        return urls
    
    def extract_wikipedia_content(self, language: str, target_count: int) -> int:
        """Extract Wikipedia content for a specific language"""
        
        logger.info(f"Extracting {target_count} Wikipedia items in {language}")
        
        # Get Wikipedia URLs
        wikipedia_urls = self.get_wikipedia_urls(language, target_count)
        
        extracted_count = 0
        
        for url_info in wikipedia_urls:
            if extracted_count >= target_count:
                break
                
            try:
                url = url_info['url']
                topic = url_info['topic']
                
                if url in self.seen_urls:
                    continue
                
                logger.info(f"Fetching Wikipedia ({language}): {topic}")
                
                # Use Tavily to fetch Wikipedia content
                response = self.client.search(
                    query=f"site:wikipedia.org {topic}",
                    include_raw_content=True,
                    max_results=1,
                    search_depth="basic"
                )
                
                if not response.get('results'):
                    logger.warning(f"No content found for {topic}")
                    continue
                
                result = response['results'][0]
                text_content = result.get("raw_content", "")
                title = result.get("title", topic)
                actual_url = result.get("url", url)
                
                if not text_content:
                    logger.warning(f"Empty content for {topic}")
                    continue
                
                # Apply minimum length filter (100 chars for Wikipedia)
                if len(text_content.strip()) < 100:
                    logger.warning(f"Content too short for {topic}: {len(text_content)} chars")
                    continue
                
                domain = self._extract_domain(actual_url)
                
                item = {
                    'url': actual_url,
                    'text': text_content.strip(),
                    'language': language,
                    'domain': domain,
                    'source': 'wikipedia',
                    'length': len(text_content.strip()),
                    'word_count': len(text_content.strip().split())
                }
                
                self.collected_items.append(item)
                self.seen_urls.add(actual_url)
                extracted_count += 1
                
                logger.info(f"✓ Wikipedia {language}: {title[:40]}... ({len(text_content)} chars)")
                
                # Small delay between fetches
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error fetching Wikipedia '{topic}': {str(e)}")
                continue
        
        logger.info(f"Wikipedia {language}: extracted {extracted_count}/{target_count} items")
        return extracted_count
    
    def get_dynamic_search_configs(self) -> Dict[str, Dict]:
        """Get search configurations for dynamic content - Expanded topics"""
        
        return {
            "news": {
                "english": {
                    "domains": ["bbc.com", "reuters.com", "cnn.com", "apnews.com", "theguardian.com", "npr.org"],
                    "search_terms": [
                        # Technology & AI
                        "artificial intelligence breakthrough",
                        "machine learning advancement",
                        "quantum computing progress",
                        "cybersecurity threats 2024",
                        "blockchain technology news",
                        "robotics innovation",
                        "virtual reality development",
                        "5G technology deployment",
                        
                        # Science & Health
                        "climate change research",
                        "medical research breakthrough",
                        "vaccine development news",
                        "space exploration discovery",
                        "renewable energy development",
                        "nuclear fusion progress",
                        "gene therapy advancement",
                        "alzheimer research",
                        
                        # Business & Economy
                        "cryptocurrency market trends",
                        "electric vehicle industry",
                        "sustainable finance",
                        "startup funding news",
                        "supply chain disruption",
                        "inflation economic impact",
                        "remote work trends",
                        "digital transformation",
                        
                        # Social & Politics
                        "social media regulation",
                        "data privacy laws",
                        "election technology",
                        "urban planning innovation",
                        "education technology",
                        "mental health awareness",
                        "environmental policy",
                        "immigration reform"
                    ]
                },
                "spanish": {
                    "domains": ["elpais.com", "eluniversal.com.mx", "clarin.com", "elmundo.es"],
                    "search_terms": [
                        "inteligencia artificial",
                        "cambio climático",
                        "investigación médica", 
                        "innovación tecnológica",
                        "energía renovable",
                        "criptomonedas",
                        "exploración espacial",
                        "vehículos eléctricos",
                        "educación digital",
                        "ciberseguridad"
                    ]
                },
                "french": {
                    "domains": ["lemonde.fr", "liberation.fr", "lefigaro.fr", "francetvinfo.fr"],
                    "search_terms": [
                        "intelligence artificielle",
                        "changement climatique",
                        "recherche médicale",
                        "innovation technologique",
                        "énergie renouvelable",
                        "cryptomonnaies",
                        "exploration spatiale",
                        "véhicules électriques",
                        "transformation numérique",
                        "cybersécurité"
                    ]
                },
                "german": {
                    "domains": ["spiegel.de", "zeit.de", "faz.net", "sueddeutsche.de"],
                    "search_terms": [
                        "künstliche intelligenz",
                        "klimawandel",
                        "medizinische forschung",
                        "technische innovation",
                        "erneuerbare energie",
                        "kryptowährungen",
                        "weltraumforschung",
                        "elektrofahrzeuge",
                        "digitale transformation",
                        "cybersicherheit"
                    ]
                },
                "chinese": {
                    "domains": ["sina.com.cn", "163.com", "xinhuanet.com", "sohu.com"],
                    "search_terms": [
                        "人工智能",
                        "气候变化",
                        "医学研究",
                        "技术创新",
                        "可再生能源",
                        "加密货币",
                        "太空探索",
                        "电动汽车",
                        "数字化转型",
                        "网络安全"
                    ]
                },
                "arabic": {
                    "domains": ["aljazeera.com", "alarabiya.net", "bbc.com"],
                    "search_terms": [
                        "ذكاء اصطناعي",
                        "تغير المناخ",
                        "البحث الطبي",
                        "الابتكار التكنولوجي",
                        "الطاقة المتجددة",
                        "العملات المشفرة"
                    ]
                }
            },
            
            "github": {
                "english": {
                    "domains": ["github.com"],
                    "search_terms": [
                        # AI & ML frameworks
                        "machine learning framework README",
                        "deep learning library README",
                        "computer vision project README",
                        "natural language processing README",
                        "reinforcement learning README",
                        
                        # Web development
                        "web development framework README",
                        "frontend library README",
                        "backend framework README",
                        "full stack application README",
                        "API development README",
                        
                        # Data & Analytics
                        "data science library README",
                        "data visualization tool README",
                        "database management README",
                        "big data processing README",
                        "analytics platform README",
                        
                        # DevOps & Tools
                        "DevOps automation README",
                        "containerization tool README",
                        "CI/CD pipeline README",
                        "monitoring solution README",
                        "deployment tool README",
                        
                        # Security & Blockchain
                        "cybersecurity tool README",
                        "blockchain project README",
                        "cryptocurrency wallet README",
                        "security scanner README",
                        
                        # Mobile & Gaming
                        "mobile app development README",
                        "game development engine README",
                        "cross platform framework README"
                    ]
                }
            },
            
            "reddit": {
                "english": {
                    "domains": ["reddit.com"],
                    "search_terms": [
                        # Tech discussions
                        "site:reddit.com/r/MachineLearning breakthrough",
                        "site:reddit.com/r/programming best practices",
                        "site:reddit.com/r/webdev frameworks comparison",
                        "site:reddit.com/r/datascience career advice",
                        "site:reddit.com/r/cybersecurity threats analysis",
                        "site:reddit.com/r/artificial intelligence discussion",
                        "site:reddit.com/r/technology innovation",
                        "site:reddit.com/r/gamedev indie development",
                        
                        # Science discussions
                        "site:reddit.com/r/science latest research",
                        "site:reddit.com/r/askscience explanation",
                        "site:reddit.com/r/space exploration news",
                        "site:reddit.com/r/medicine breakthrough",
                        "site:reddit.com/r/climate change discussion",
                        
                        # Business & Finance
                        "site:reddit.com/r/entrepreneur startup advice",
                        "site:reddit.com/r/investing market analysis",
                        "site:reddit.com/r/cryptocurrency trends",
                        "site:reddit.com/r/personalfinance tips"
                    ]
                },
                "spanish": {
                    "domains": ["reddit.com"],
                    "search_terms": [
                        "site:reddit.com/r/es tecnología",
                        "site:reddit.com/r/spain programación",
                        "site:reddit.com/r/mexico tecnologia",
                        "site:reddit.com/r/argentina tecnologia",
                        "site:reddit.com/r/chile programacion"
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
            
            "blog": {
                "english": {
                    "domains": ["medium.com", "dev.to", "hashnode.com", "substack.com", "towardsdatascience.com"],
                    "search_terms": [
                        # AI & Data Science
                        "artificial intelligence tutorial",
                        "machine learning implementation",
                        "data science analysis",
                        "python data analysis",
                        "neural network guide",
                        "computer vision tutorial",
                        
                        # Web Development
                        "web development guide",
                        "react tutorial",
                        "javascript best practices",
                        "full stack development",
                        "API design guide",
                        "frontend optimization",
                        
                        # Software Engineering
                        "software engineering practices",
                        "clean code principles",
                        "system design tutorial",
                        "microservices architecture",
                        "database design guide",
                        
                        # Technology Trends
                        "cloud computing architecture",
                        "DevOps best practices",
                        "cybersecurity analysis",
                        "blockchain tutorial",
                        "mobile development guide",
                        
                        # Business & Productivity
                        "tech startup advice",
                        "remote work productivity",
                        "project management tips",
                        "technology trends 2024"
                    ]
                },
                "spanish": {
                    "domains": ["medium.com"],
                    "search_terms": [
                        "inteligencia artificial tutorial",
                        "desarrollo web guía",
                        "ciencia de datos",
                        "programación práticas",
                        "tecnología blockchain",
                        "desarrollo móvil"
                    ]
                }
            }
        }
    

    
    def extract_dynamic_content(self, source: str, language: str, target_count: int) -> int:
        """Extract dynamic content (news, github, reddit, blog) for a specific language"""
        
        logger.info(f"Extracting {target_count} {source} items in {language}")
        
        search_configs = self.get_dynamic_search_configs()
        
        if source not in search_configs or language not in search_configs[source]:
            logger.warning(f"No search config for {source} in {language}")
            return 0
        
        config = search_configs[source][language]
        domains = config["domains"]
        search_terms = config["search_terms"]
        
        extracted_count = 0
        
        # Shuffle search terms for variety
        shuffled_terms = search_terms.copy()
        random.shuffle(shuffled_terms)
        
        for search_term in shuffled_terms:
            if extracted_count >= target_count:
                break
                
            try:
                logger.info(f"Searching {source} ({language}): '{search_term}'")
                
                response = self.client.search(
                    query=search_term,
                    include_raw_content=True,  # True for markdown, text for plain text
                    include_domains=domains,
                    max_results=3,
                    search_depth="basic"
                )
                
                for result in response.get('results', []):
                    if extracted_count >= target_count:
                        break
                    
                    url = result.get("url", "")
                    text_content = result.get("raw_content", "")
                    title = result.get("title", "")
                    
                    if not text_content or url in self.seen_urls:
                        continue
                    
                    # Apply minimum length filter (100 chars)
                    if len(text_content.strip()) < 100:
                        continue
                    
                    domain = self._extract_domain(url)
                    
                    item = {
                        'url': url,
                        'text': text_content.strip(),
                        'language': language,
                        'domain': domain,
                        'source': source,
                        'length': len(text_content.strip()),
                        'word_count': len(text_content.strip().split())
                    }
                    
                    self.collected_items.append(item)
                    self.seen_urls.add(url)
                    extracted_count += 1
                    
                    logger.info(f"✓ {source} {language}: {title[:40]}... ({len(text_content)} chars)")
                
                # Small delay between searches
                time.sleep(random.uniform(1, 2))
                
            except Exception as e:
                logger.error(f"Error searching {source} '{search_term}': {str(e)}")
                continue
        
        logger.info(f"{source} {language}: extracted {extracted_count}/{target_count} items")
        return extracted_count
    
    def _get_min_length_for_source(self, source: str) -> int:
        """Get minimum content length based on source type"""
        return 100  # Other sources minimum 100 chars
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain name from URL"""
        try:
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def create_dataset(self, total_items: int = 300):
        """Main method to create the unified dataset"""
        
        logger.info(f"Creating dataset with Wikipedia + dynamic content ({total_items} items)")
        logger.info("="*60)
        
        # Get target distribution
        distribution = self.get_target_distribution(total_items)
        
        logger.info("TARGET DISTRIBUTION:")
        for (source, language), count in sorted(distribution.items()):
            logger.info(f"  {source.capitalize()} ({language}): {count} items")
        
        logger.info("\nStarting extraction...")
        logger.info("="*60)
        
        # Extract content according to distribution
        total_extracted = 0
        
        for (source, language), target_count in distribution.items():
            if target_count == 0:
                continue
            
            logger.info(f"\n--- {source.upper()} in {language.upper()} (target: {target_count}) ---")
            
            if source == 'wikipedia':
                extracted = self.extract_wikipedia_content(language, target_count)
            else:
                extracted = self.extract_dynamic_content(source, language, target_count)
            
            total_extracted += extracted
            
            logger.info(f"Progress: {len(self.collected_items)}/{total_items} total items collected")
            
            # Add delay between different source-language combinations
            time.sleep(random.uniform(2, 4))
        
        # Save to CSV
        self._save_csv()
        
        # Print summary
        self._print_summary(distribution)
    
    def _save_csv(self):
        """Save collected items to CSV file"""
        
        if not self.collected_items:
            logger.warning("No items to save!")
            return
        
        logger.info(f"\nSaving {len(self.collected_items)} items to {self.output_file}")
        
        fieldnames = ['url', 'raw_markdown', 'language', 'domain', 'source', 'length', 'word_count']
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in self.collected_items:
                writer.writerow(item)
        
        # Calculate file size
        file_size_mb = os.path.getsize(self.output_file) / (1024 * 1024)
        logger.info(f"✓ Saved to {self.output_file} ({file_size_mb:.1f} MB)")
    
    def _print_summary(self, target_distribution):
        """Print comprehensive summary"""
        
        logger.info("\n" + "="*60)
        logger.info("DATASET CREATION COMPLETED!")
        logger.info("="*60)
        
        logger.info(f"Total items collected: {len(self.collected_items)}")
        logger.info(f"Output file: {self.output_file}")
        
        if not self.collected_items:
            return
        
        # Calculate actual distribution
        actual_dist = {}
        for item in self.collected_items:
            key = (item['source'], item['language'])
            actual_dist[key] = actual_dist.get(key, 0) + 1
        
        # Calculate statistics
        total_chars = sum(item['length'] for item in self.collected_items)
        total_words = sum(item['word_count'] for item in self.collected_items)
        avg_length = total_chars // len(self.collected_items)
        avg_words = total_words // len(self.collected_items)
        
        logger.info(f"Average content length: {avg_length:,} characters")
        logger.info(f"Average word count: {avg_words:,} words")
        logger.info(f"Total content: {total_chars:,} characters")
        
        logger.info("\nACTUAL vs TARGET DISTRIBUTION:")
        for (source, language), target in sorted(target_distribution.items()):
            actual = actual_dist.get((source, language), 0)
            logger.info(f"  {source.capitalize()} ({language}): {actual}/{target}")
        
        # Summary by source
        logger.info("\nBY SOURCE:")
        source_totals = {}
        for item in self.collected_items:
            source = item['source']
            source_totals[source] = source_totals.get(source, 0) + 1
        
        for source, count in sorted(source_totals.items()):
            percentage = (count / len(self.collected_items)) * 100
            logger.info(f"  {source.capitalize()}: {count} items ({percentage:.1f}%)")
        
        # Summary by language  
        logger.info("\nBY LANGUAGE:")
        lang_totals = {}
        for item in self.collected_items:
            language = item['language']
            lang_totals[language] = lang_totals.get(language, 0) + 1
        
        for language, count in sorted(lang_totals.items()):
            percentage = (count / len(self.collected_items)) * 100
            logger.info(f"  {language.capitalize()}: {count} items ({percentage:.1f}%)")
        
        # Top domains
        logger.info("\nTOP DOMAINS:")
        domain_counts = {}
        for item in self.collected_items:
            domain = item['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        top_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for domain, count in top_domains:
            logger.info(f"  {domain}: {count} items")
        
        logger.info(f"\n✓ Dataset saved to: {self.output_file}")

def main():
    """Main function to run the dataset creator"""
    creator = UnifiedDatasetCreator()
    creator.create_dataset(500)  # Target 500 total items

if __name__ == "__main__":
    main()