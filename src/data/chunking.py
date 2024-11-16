import google.generativeai as genai
from typing import Optional, List, Dict, Any
import os
import logging
import json


class DocumentChunker:
    """Class to chunk documents using Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Configure API key
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable")
        
        genai.configure(api_key=api_key)
        
        try:
            self.model = genai.GenerativeModel('models/gemini-1.5-flash-001')
            self.logger.info("Successfully initialized Gemini model")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini model: {str(e)}")
            raise

    def create_chunks(self, chapter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use Gemini to divide chapter into logical chunks."""
        
        prompt = f"""
        Analyze the following chapter and divide it into logical chunks. Return ONLY a JSON array 
        with no additional text or markdown formatting. Each chunk object should have exactly two fields:
        "text" and "summary".

        Guidelines for chunks:
        - Each chunk should be a coherent scene or section
        - Maintain narrative flow
        - Not exceed 2000 characters
        - Not break mid-dialogue or mid-action

        Chapter {chapter['chapter_number']}: {chapter['title']}

        Text:
        {chapter['text']}

        Respond with ONLY a JSON array in this exact format:
        [
            {{"text": "chunk text here", "summary": "brief summary here"}},
            {{"text": "next chunk text", "summary": "next summary"}}
        ]
        """
        
        try:
            response = self.model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from API")
            
            # Clean up the response text
            json_str = response.text.strip()
            
            # Remove any markdown formatting or extra text
            if '```' in json_str:
                # Extract content between first ``` and last ```
                parts = json_str.split('```')
                for part in parts:
                    if '[' in part and ']' in part:
                        json_str = part.strip()
                        break
            
            # Remove any "json" or language identifier
            json_str = json_str.replace('json\n', '')
            
            # Ensure the string starts with [ and ends with ]
            json_str = json_str[json_str.find('['):json_str.rfind(']')+1]
            
            # Parse JSON with error handling
            try:
                chunks_data = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing error: {str(e)}")
                self.logger.error(f"Problematic JSON string: {json_str}")
                return []
            
            # Format chunks with metadata
            chunks = []
            start_pos = 0
            
            for i, chunk_data in enumerate(chunks_data):
                # Validate chunk data
                if not isinstance(chunk_data, dict) or 'text' not in chunk_data or 'summary' not in chunk_data:
                    self.logger.error(f"Invalid chunk format: {chunk_data}")
                    continue
                    
                chunk_text = chunk_data['text'].strip()
                end_pos = start_pos + len(chunk_text)
                
                chunk = {
                    'text': chunk_text,
                    'summary': chunk_data['summary'].strip(),
                    'metadata': {
                        'chapter_number': chapter['chapter_number'],
                        'chapter_title': chapter['title'],
                        'chunk_start': start_pos,
                        'chunk_end': end_pos,
                        'chunk_index': i,
                        'is_first_chunk': i == 0,
                        'is_last_chunk': i == len(chunks_data) - 1
                    }
                }
                chunks.append(chunk)
                start_pos = end_pos
            
            if not chunks:
                raise ValueError("No valid chunks were created")
                
            self.logger.info(f"Successfully created {len(chunks)} chunks for Chapter {chapter['chapter_number']}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Failed to create chunks: {str(e)}")
            return []

    def chunk_chapters(self, chapters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple chapters into chunks.
        
        Args:
            chapters: List of chapter dictionaries
            
        Returns:
            List of all chunks across all chapters
        """
        all_chunks = []
        for chapter in chapters:
            self.logger.info(f"Processing Chapter {chapter['chapter_number']}: {chapter['title']}")
            chunks = self.create_chunks(chapter)
            all_chunks.extend(chunks)
        return all_chunks

# %%
