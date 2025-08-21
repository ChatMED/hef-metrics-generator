"""
Query logger for hef_metrics_generator.

This module tracks all queries sent to external tools (e.g., PubMed, ArXiv, DuckDuckGo)
and saves them to timestamped log files for auditing, debugging, and reproducibility.
"""

import os
import datetime
from typing import List


class QueryLogger:
    """
    A lightweight logger that stores queries in memory and saves them
    to a timestamped log file under `query_logs/`.
    """

    def __init__(self, log_dir: str = "query_logs") -> None:
        """
        Initialize the query logger.

        Args:
            log_dir (str): Directory where query logs are saved.
                           Defaults to "query_logs".
        """
        self.queries: List[str] = []
        self.log_dir = log_dir

    def log(self, tool_name: str, query: str) -> None:
        """
        Log a single query with the tool name.

        Args:
            tool_name (str): Name of the tool making the query.
            query (str): The query string sent to the tool.
        """
        entry = f"{tool_name}: {query.strip()}"
        self.queries.append(entry)

    def save(self) -> None:
        """
        Save all accumulated queries to a timestamped file in `self.log_dir`.
        Clears the in-memory query list after saving.
        """
        if not self.queries:
            return

        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = os.path.join(self.log_dir, f"{timestamp}.txt")

        with open(filename, "w", encoding="utf-8") as f:
            for q in self.queries:
                f.write(q + "\n")

        self.queries.clear()


query_logger = QueryLogger()
