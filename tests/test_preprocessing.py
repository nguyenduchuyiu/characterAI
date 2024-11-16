from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import logging
sys.path.append('..')
sys.path.append(str(Path(__file__).parent.parent))


from src.data.preprocessing import DataPreprocessor
from src.data.chunking import DocumentChunker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPreprocessing:
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.sample_path = Path('../data/raw/novels/Harry Potter and the Chamber of Secrets.txt')
        cls.preprocessor = DataPreprocessor(Path('../data/raw'))
        cls.chunker = DocumentChunker(chunk_size=1000, overlap=200)

    def test_novel_processing(self):
        """Test novel preprocessing"""
        try:
            # Process novel
            chapters = self.preprocessor.process_novel(self.sample_path)
            
            # Basic assertions
            assert len(chapters) > 0, "No chapters were processed"
            assert all(isinstance(chapter, dict) for chapter in chapters)
            assert all('text' in chapter for chapter in chapters)
            
            # Display sample
            self.display_sample_chapter(chapters)
            
            # Process chunks
            total_chunks, context_types = self.analyze_chunks_in_batches(chapters)
            
            # Display results
            self.display_results(total_chunks, context_types)
            
            # Visualize results
            self.plot_context_distribution(context_types, total_chunks)
            
        except Exception as e:
            logger.error(f"Error in test_novel_processing: {str(e)}")
            raise

    @staticmethod
    def display_sample_chapter(chapters, chapter_idx=0, sample_length=500):
        """Display sample from chapter"""
        if chapters and len(chapters) > chapter_idx:
            chapter = chapters[chapter_idx]
            logger.info(f"\nChapter {chapter['chapter_number']}: {chapter['title']}")
            logger.info(f"Length: {chapter['length']} characters")
            logger.info(f"\nSample text:\n{chapter['text'][:sample_length]}...")
        else:
            logger.warning("No chapters found or invalid chapter index")

    def analyze_chunks_in_batches(self, chapters, batch_size=5):
        """Process chapters in batches to manage memory"""
        total_chunks = 0
        context_types = {}
        
        for i in tqdm(range(0, len(chapters), batch_size), desc="Processing batches"):
            batch = chapters[i:i + batch_size]
            batch_chunks = self.chunker.create_chunks_from_chapters(batch)
            
            total_chunks += len(batch_chunks)
            
            # Update context type counts
            for chunk in batch_chunks:
                ctx_type = chunk['metadata']['context_type']
                context_types[ctx_type] = context_types.get(ctx_type, 0) + 1
            
            # Display sample from first batch
            if i == 0:
                logger.info("\nSample chunks from first batch:")
                for j, chunk in enumerate(batch_chunks[:3]):
                    self.display_chunk_info(chunk, j)
            
            del batch_chunks
            
        return total_chunks, context_types

    @staticmethod
    def display_chunk_info(chunk, index):
        """Display chunk information"""
        logger.info(f"\nChunk {index}:")
        logger.info(f"Chapter: {chunk['metadata']['chapter_number']} - {chunk['metadata']['chapter_title']}")
        logger.info(f"Context Type: {chunk['metadata']['context_type']}")
        logger.info(f"Dialogue Count: {chunk['metadata']['dialogue_count']}")
        logger.info(f"Sample text:\n{chunk['text'][:200]}...")

    @staticmethod
    def display_results(total_chunks, context_types):
        """Display processing results"""
        logger.info(f"\nTotal chunks created: {total_chunks}")
        logger.info("\nContext Type Distribution:")
        for ctx_type, count in context_types.items():
            percentage = (count/total_chunks*100)
            logger.info(f"{ctx_type}: {count} chunks ({percentage:.1f}%)")

    def plot_context_distribution(self, context_types, total_chunks):
        """Create visualization of context type distribution"""
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        types = list(context_types.keys())
        counts = list(context_types.values())
        percentages = [count/total_chunks*100 for count in counts]
        
        # Create bar plot
        sns.barplot(x=types, y=percentages)
        plt.title('Distribution of Context Types')
        plt.xlabel('Context Type')
        plt.ylabel('Percentage of Chunks')
        
        # Add value labels
        for i, p in enumerate(percentages):
            plt.text(i, p, f'{p:.1f}%', ha='center')
        
        plt.tight_layout()
        plt.savefig('../data/figures/context_distribution.png')
        plt.close()
        
if __name__ == "__main__":
    test = TestPreprocessing()
    test.test_novel_processing()
