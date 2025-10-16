import os
import json
import hashlib
import tempfile
from copy import deepcopy
from utils import strip_comments

class ExecutionCache:
    """
    Cache for storing and retrieving execution logs of code snippets.
    This helps avoid re-executing the same code multiple times during testing.
    """
    
    def __init__(self, cache_file_path=None):
        """
        Initialize the execution cache.
        
        Args:
            cache_file_path (str, optional): Path to the cache file. If None, defaults to 
                                            'execution_cache.json' in the same directory.
        """
        if cache_file_path is None:
            # Default to a file in the same directory as this module
            self.cache_file_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "results",
                "op_cog",
                "execution_cache.json"
            )
        else:
            self.cache_file_path = cache_file_path

        print(cache_file_path)
            
        self.cache = self._load_cache()
    
    def _load_cache(self):
        """Load the cache from disk if it exists, otherwise return an empty cache."""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load cache file: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save the cache atomically (write-then-replace)."""
        dir_path = os.path.dirname(self.cache_file_path)
        os.makedirs(dir_path, exist_ok=True)

        tmp_file = None
        try:
            # 1. write to a temp file in the same directory
            with tempfile.NamedTemporaryFile(
                mode="w",
                delete=False,
                dir=dir_path,
                prefix="cache_",
                suffix=".tmp",
            ) as tmp_file:
                json.dump(self.cache, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())       # ensure bytes hit disk

            # 2. atomically replace the old cache
            os.replace(tmp_file.name, self.cache_file_path)
        except IOError as e:
            print(f"Warning: Failed to save cache file atomically: {e}")
            if tmp_file and os.path.exists(tmp_file.name):
                try:
                    os.remove(tmp_file.name)
                except Exception:
                    pass
    
    @staticmethod
    def _generate_key(code_string: str) -> str:
        """
        Generate a cache key for a snippet.
        – strip comments (semantic-neutral)        → reuse cache when only
                                                   comments change
        – keep leading indentation (semantic)      → different semantics ⇒
                                                   different key
        – remove trailing spaces                   → whitespace-insensitive
        """
        # 1. drop comments
        code_no_comments = strip_comments(code_string)

        # 2. remove leading / trailing blank lines
        code_core = code_no_comments.strip("\n")

        # 3. remove only trailing spaces but keep indentation
        normalized_lines = [ln.rstrip() for ln in code_core.split("\n")]
        normalized_code  = "\n".join(normalized_lines)

        # 4. hash
        return hashlib.md5(normalized_code.encode("utf-8")).hexdigest()
    
    def get(self, code_string):
        """
        Get the execution logs for a code snippet from the cache.
        
        Args:
            code_string (str): The code snippet to look up.
            
        Returns:
            list or None: The execution logs if found, None otherwise.
        """
        key = self._generate_key(code_string)
        if key in self.cache:
            print(f"Cache hit for code snippet (key: {key[:8]}...)")
            return deepcopy(self.cache[key])
        
        print(f"Cache miss for code snippet (key: {key[:8]}...)")
        return None
    
    def put(self, code_string, logs):
        """
        Store the execution logs for a code snippet in the cache.
        
        Args:
            code_string (str): The code snippet.
            logs (list): The execution logs.
        """
        key = self._generate_key(code_string)
        self.cache[key] = deepcopy(logs)
        print(f"Cached execution logs for code snippet (key: {key[:8]}...)")
        self._save_cache()
    
    def clear(self):
        """Clear the entire cache."""
        self.cache = {}
        if os.path.exists(self.cache_file_path):
            try:
                os.remove(self.cache_file_path)
                print(f"Cache file deleted: {self.cache_file_path}")
            except OSError as e:
                print(f"Warning: Failed to delete cache file: {e}")
        else:
            print(f"Cache file does not exist: {self.cache_file_path}")
