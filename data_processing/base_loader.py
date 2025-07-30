from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Abstract base class for all data loaders."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        
    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file and return standardized DataFrame."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate loaded data structure and content."""
        pass
    
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required columns for this data type."""
        pass
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply common preprocessing steps."""
        return data
    
    def quality_filter(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply quality filtering based on configuration."""
        return data
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Return loaded and processed data."""
        return self.data
    
    def has_data(self) -> bool:
        """Check if data has been loaded."""
        return self.data is not None and not self.data.empty


class MODISDataLoader(BaseDataLoader):
    """Base class for MODIS data loaders."""
    
    def __init__(self, config: Dict[str, Any], product_name: str):
        super().__init__(config)
        self.product_name = product_name
        
    def get_common_modis_columns(self) -> List[str]:
        """Return common MODIS columns."""
        return ['date', 'lat', 'lon', 'albedo', 'quality_flag']
    
    def validate_coordinates(self, data: pd.DataFrame) -> bool:
        """Validate coordinate ranges."""
        if 'lat' in data.columns and 'lon' in data.columns:
            lat_valid = data['lat'].between(-90, 90).all()
            lon_valid = data['lon'].between(-180, 180).all()
            return lat_valid and lon_valid
        return False
    
    def validate_albedo_range(self, data: pd.DataFrame) -> bool:
        """Validate albedo values are in reasonable range."""
        if 'albedo' in data.columns:
            return data['albedo'].between(0, 1).all()
        return False


class AWSDataLoader(BaseDataLoader):
    """Base class for AWS data loaders."""
    
    def get_required_columns(self) -> List[str]:
        """Return required AWS columns."""
        return ['date', 'albedo', 'station_id']
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate AWS data structure."""
        required_cols = self.get_required_columns()
        has_required = all(col in data.columns for col in required_cols)
        
        if not has_required:
            logger.error(f"Missing required columns: {required_cols}")
            return False
            
        if 'albedo' in data.columns:
            if not data['albedo'].between(0, 1).all():
                logger.error("Albedo values outside valid range [0, 1]")
                return False
                
        return True