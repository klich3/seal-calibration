"""SEAL calibration file parser utilities"""
import re
from typing import List, Tuple


class SEALCalibrationParser:
    """Utility functions for parsing SEAL format"""
    
    @staticmethod
    def parse_float_line(line: str, count: int = None) -> List[float]:
        """
        Parse a line containing space-separated floats
        
        Args:
            line: Input line
            count: Expected number of values (None = all)
            
        Returns:
            List of float values
        """
        values = list(map(float, line.split()))
        if count is not None:
            return values[:count]
        return values
    
    @staticmethod
    def replace_leading_floats(
        line: str, 
        new_values: List[float], 
        max_replace: int = None
    ) -> str:
        """
        Replace leading float values in a line preserving layout
        
        Args:
            line: Input line
            new_values: New values to insert
            max_replace: Maximum number of values to replace
            
        Returns:
            Modified line
        """
        pattern = re.compile(r"[-+]?\d+(?:\.\d+)?")
        matches = list(pattern.finditer(line))
        
        n_to_replace = (
            len(new_values) if max_replace is None 
            else min(len(new_values), max_replace)
        )
        
        if n_to_replace == 0 or not matches:
            return line
        
        out = []
        last_idx = 0
        
        for i, m in enumerate(matches):
            if i >= n_to_replace:
                break
            out.append(line[last_idx:m.start()])
            out.append("{:.6f}".format(float(new_values[i])))
            last_idx = m.end()
        
        out.append(line[last_idx:])
        return "".join(out)
    
    @staticmethod
    def extract_metadata_field(line: str, field: str) -> str:
        """
        Extract a metadata field from the metadata line
        
        Args:
            line: Metadata line (***DevID:xxx***...)
            field: Field name (e.g., 'DevID', 'CalibrateDate')
            
        Returns:
            Field value or empty string if not found
        """
        pattern = f"{field}:([^\\*]+)"
        match = re.search(pattern, line)
        return match.group(1) if match else ""
