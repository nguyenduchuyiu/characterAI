from pathlib import Path
import re
import json
import logging
from typing import List, Dict, Union
import requests
from bs4 import BeautifulSoup
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, raw_data_path: Path):
        """
        Initialize preprocessor with path to raw data
        
        Args:
            raw_data_path (Path): Path to directory containing raw data files
        """
        self.raw_data_path = raw_data_path
        self.clean_patterns = {
            'whitespace': re.compile(r'\s+'),
            'chapter': re.compile(r'Chapter \d+[:\s]*(.*?)\n'),
            'quotes': re.compile(r'["""]'),
            'headers': re.compile(r'^.*?Contents.*?\n', re.MULTILINE | re.DOTALL)
        }

    def process_novel(self, file_path: str) -> List[Dict[str, Union[str, int]]]:
        """
        Process a novel file into chapters with metadata
        
        Args:
            file_path (str): Path to novel file
            
        Returns:
            List[Dict]: List of chapters with text and metadata
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Remove headers/table of contents if present
            text = self.clean_patterns['headers'].sub('', text)
            
            # Extract chapters with titles
            chapter_matches = self.clean_patterns['chapter'].finditer(text)
            chapters = []
            
            for i, match in enumerate(chapter_matches, 1):
                chapter_title = match.group(1).strip()
                start_pos = match.end()
                
                # Find the start of next chapter or end of text
                next_match = text.find('Chapter', start_pos)
                if next_match == -1:
                    next_match = len(text)
                
                chapter_text = text[start_pos:next_match].strip()
                
                # Clean the text
                chapter_text = self._clean_text(chapter_text)
                
                chapters.append({
                    'chapter_number': i,
                    'title': chapter_title,
                    'text': chapter_text,
                    'length': len(chapter_text)
                })
            
            logger.info(f"Processed {len(chapters)} chapters from {file_path}")
            return chapters
            
        except Exception as e:
            logger.error(f"Error processing novel file {file_path}: {str(e)}")
            raise

    def process_wiki(self, url: str) -> Dict:
        """
        Process wiki data into structured format
        
        Args:
            url (str): URL to wiki page
            
        Returns:
            Dict: Structured wiki data
        """
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content
            content = soup.find(id='mw-content-text')
            
            # Process sections
            sections = {}
            current_section = None
            
            for element in content.find_all(['h2', 'p']):
                if element.name == 'h2':
                    current_section = element.get_text().strip()
                    sections[current_section] = []
                elif current_section and element.name == 'p':
                    text = self._clean_text(element.get_text())
                    if text:  # Only add non-empty paragraphs
                        sections[current_section].append(text)
            
            # Structure the data
            wiki_data = {
                'title': soup.find(id='firstHeading').get_text(),
                'url': url,
                'sections': sections,
                'metadata': {
                    'last_modified': soup.find(id='footer-info-lastmod').get_text(),
                    'source': 'wiki'
                }
            }
            
            logger.info(f"Processed wiki page: {wiki_data['title']}")
            return wiki_data
            
        except Exception as e:
            logger.error(f"Error processing wiki page {url}: {str(e)}")
            raise

    def process_profile(self, file_path: str) -> Dict:
        """
        Process character profile data from JSON format
        
        Args:
            file_path (str): Path to profile JSON file
            
        Returns:
            Dict: Structured profile data
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                profile_data = json.load(f)
            
            # Validate required fields
            required_fields = ['name', 'description', 'traits', 'relationships']
            for field in required_fields:
                if field not in profile_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Clean text fields
            profile_data['description'] = self._clean_text(profile_data['description'])
            
            if 'quotes' in profile_data:
                profile_data['quotes'] = [
                    self._clean_text(quote) for quote in profile_data['quotes']
                ]
            
            # Add metadata
            profile_data['metadata'] = {
                'source': 'profile',
                'processed_date': datetime.now().isoformat()
            }
            
            logger.info(f"Processed profile for: {profile_data['name']}")
            return profile_data
            
        except Exception as e:
            logger.error(f"Error processing profile {file_path}: {str(e)}")
            raise

    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing extra whitespace and normalizing quotes
        
        Args:
            text (str): Text to clean
            
        Returns:
            str: Cleaned text
        """
        # Normalize whitespace
        text = self.clean_patterns['whitespace'].sub(' ', text)
        
        # Normalize quotes
        text = self.clean_patterns['quotes'].sub('"', text)
        
        # Remove leading/trailing whitespace
        return text.strip()
