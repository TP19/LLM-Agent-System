"""
Collection Router - Routes queries to appropriate collections
"""
from typing import List, Dict
import yaml
from pathlib import Path

class CollectionRouter:
    """Routes queries to the right collections based on task type"""
    
    def __init__(self, config_path: str = "config/rag_config.yaml"):
        """Load routing configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.collection_routes = self.config.get('collections', {})
    
    def get_collections_for_task(self, task_type: str) -> Dict[str, List[str]]:
        """
        Get collections to query for a given task
        
        Args:
            task_type: One of 'document_summarization', 'debugging', 'general_qa'
            
        Returns:
            Dict with 'private' and 'public' collection lists
        """
        route = self.collection_routes.get(
            task_type,
            {'private': ['documents'], 'public': []}  # Default
        )
        return route
    
    def get_all_collections(self, task_type: str) -> List[str]:
        """Get flat list of all collections for a task"""
        route = self.get_collections_for_task(task_type)
        return route.get('private', []) + route.get('public', [])