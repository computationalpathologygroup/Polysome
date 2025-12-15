import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


class OutputFormatter:
    """
    Utility class for converting JSONL files to various output formats.
    
    Supports conversion from JSONL to Excel, JSON array, and Parquet formats.
    All methods work by reading a completed JSONL file and converting it to the target format.
    """

    @staticmethod
    def jsonl_to_excel(
        jsonl_path: Path, 
        output_path: Path, 
        sheet_name: str = "Sheet1",
        index: bool = False,
        **options
    ) -> None:
        """
        Convert JSONL file to Excel format.
        
        Args:
            jsonl_path: Path to source JSONL file
            output_path: Path for output Excel file
            sheet_name: Name of Excel sheet
            index: Whether to include pandas index in output
            **options: Additional options passed to pandas.to_excel()
        """
        logger.info(f"Converting JSONL to Excel: {jsonl_path} -> {output_path}")
        
        try:
            # Read JSONL into DataFrame
            df = OutputFormatter._read_jsonl_to_dataframe(jsonl_path)
            
            if df.empty:
                logger.warning(f"No data found in {jsonl_path}, creating empty Excel file")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to Excel
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(
                    writer, 
                    sheet_name=sheet_name, 
                    index=index,
                    **options
                )
            
            logger.info(f"Successfully converted to Excel: {output_path} ({len(df)} rows)")
            
        except ImportError as e:
            logger.error(f"Excel export requires 'openpyxl'. Install with: pip install openpyxl")
            raise ImportError("Missing dependency for Excel export: openpyxl") from e
        except Exception as e:
            logger.error(f"Failed to convert JSONL to Excel: {e}")
            raise

    @staticmethod
    def jsonl_to_json(
        jsonl_path: Path,
        output_path: Path,
        indent: Optional[int] = 2,
        orient: str = "records",
        **options
    ) -> None:
        """
        Convert JSONL file to JSON array format.
        
        Args:
            jsonl_path: Path to source JSONL file
            output_path: Path for output JSON file
            indent: JSON indentation (None for compact)
            orient: Pandas orient parameter ('records', 'index', etc.)
            **options: Additional options passed to DataFrame.to_json()
        """
        logger.info(f"Converting JSONL to JSON: {jsonl_path} -> {output_path}")
        
        try:
            # Read JSONL into DataFrame
            df = OutputFormatter._read_jsonl_to_dataframe(jsonl_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if df.empty:
                # Write empty array for empty data
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=indent)
                logger.warning(f"No data found in {jsonl_path}, created empty JSON array")
            else:
                # Convert to JSON
                df.to_json(
                    output_path,
                    orient=orient,
                    indent=indent,
                    force_ascii=False,
                    **options
                )
            
            logger.info(f"Successfully converted to JSON: {output_path} ({len(df)} rows)")
            
        except Exception as e:
            logger.error(f"Failed to convert JSONL to JSON: {e}")
            raise

    @staticmethod
    def jsonl_to_parquet(
        jsonl_path: Path,
        output_path: Path,
        compression: str = "snappy",
        index: bool = False,
        **options
    ) -> None:
        """
        Convert JSONL file to Parquet format.
        
        Args:
            jsonl_path: Path to source JSONL file
            output_path: Path for output Parquet file
            compression: Compression algorithm ('snappy', 'gzip', 'brotli', None)
            index: Whether to include pandas index in output
            **options: Additional options passed to DataFrame.to_parquet()
        """
        logger.info(f"Converting JSONL to Parquet: {jsonl_path} -> {output_path}")
        
        try:
            # Read JSONL into DataFrame
            df = OutputFormatter._read_jsonl_to_dataframe(jsonl_path)
            
            if df.empty:
                logger.warning(f"No data found in {jsonl_path}, creating empty Parquet file")
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to Parquet
            df.to_parquet(
                output_path,
                compression=compression,
                index=index,
                **options
            )
            
            logger.info(f"Successfully converted to Parquet: {output_path} ({len(df)} rows)")
            
        except ImportError as e:
            logger.error(f"Parquet export requires 'pyarrow'. Install with: pip install pyarrow")
            raise ImportError("Missing dependency for Parquet export: pyarrow") from e
        except Exception as e:
            logger.error(f"Failed to convert JSONL to Parquet: {e}")
            raise

    @staticmethod
    def convert_jsonl_to_formats(
        jsonl_path: Path,
        output_base_path: Path,
        formats: List[str],
        format_options: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Path]:
        """
        Convert JSONL file to multiple output formats.
        
        Args:
            jsonl_path: Path to source JSONL file
            output_base_path: Base path for output files (without extension)
            formats: List of formats to generate ('excel', 'json', 'parquet')
            format_options: Format-specific options dictionary
            
        Returns:
            Dictionary mapping format names to output file paths
        """
        if format_options is None:
            format_options = {}
            
        logger.info(f"Converting JSONL to multiple formats: {formats}")
        
        generated_files = {}
        
        for format_name in formats:
            format_name = format_name.lower()
            options = format_options.get(format_name, {})
            
            try:
                if format_name == "excel":
                    output_path = output_base_path.with_suffix(".xlsx")
                    OutputFormatter.jsonl_to_excel(jsonl_path, output_path, **options)
                    generated_files["excel"] = output_path
                    
                elif format_name == "json":
                    output_path = output_base_path.with_suffix(".json")
                    OutputFormatter.jsonl_to_json(jsonl_path, output_path, **options)
                    generated_files["json"] = output_path
                    
                elif format_name == "parquet":
                    output_path = output_base_path.with_suffix(".parquet")
                    OutputFormatter.jsonl_to_parquet(jsonl_path, output_path, **options)
                    generated_files["parquet"] = output_path
                    
                else:
                    logger.warning(f"Unsupported output format: {format_name}")
                    
            except Exception as e:
                logger.error(f"Failed to generate {format_name} format: {e}")
                # Continue with other formats even if one fails
                
        logger.info(f"Successfully generated {len(generated_files)} additional formats")
        return generated_files

    @staticmethod
    def _read_jsonl_to_dataframe(jsonl_path: Path) -> pd.DataFrame:
        """
        Read JSONL file into a pandas DataFrame.
        
        Args:
            jsonl_path: Path to JSONL file
            
        Returns:
            DataFrame containing the JSONL data
        """
        if not jsonl_path.exists():
            logger.warning(f"JSONL file not found: {jsonl_path}")
            return pd.DataFrame()
            
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num} in {jsonl_path}: {e}")
                            
        except Exception as e:
            logger.error(f"Error reading JSONL file {jsonl_path}: {e}")
            raise
            
        if not data:
            logger.info(f"No valid data found in {jsonl_path}")
            return pd.DataFrame()
            
        return pd.DataFrame(data)