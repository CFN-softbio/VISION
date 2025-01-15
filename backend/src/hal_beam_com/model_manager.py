from typing import Dict, List, Tuple, Collection

from src.hal_beam_com.utils import load_model, load_whisper_model


class ModelManager:
    """
    Singleton class to manage loading and caching of models.
    Handles both text models (LLMs) and audio models (Whisper variants).
    """
    _text_models: Dict[str, any] = {}
    _whisper_models: Dict[Tuple[str, bool], Tuple[any, any]] = {}

    @staticmethod
    def get_model(base_model: str) -> any:
        """
        Get a text model, loading it if not already cached.

        Args:
            base_model (str): Name of the model to load

        Returns:
            The loaded model
        """
        print(f"Accessing model {base_model}. Cache contains: {list(ModelManager._text_models.keys())}")
        if base_model not in ModelManager._text_models:
            print(f"Loading model {base_model} as it's not in cache")
            ModelManager._text_models[base_model] = load_model(base_model)
        return ModelManager._text_models[base_model]

    @staticmethod
    def get_whisper_model(model_name: str, finetuned: bool = False) -> Tuple[any, any]:
        """
        Get a Whisper model and its processor, loading if not already cached.

        Args:
            model_name (str): Name of the Whisper model to load
            finetuned (bool): Whether to load the finetuned version

        Returns:
            tuple: (model, processor)
        """
        key = (model_name, finetuned)
        print(f"Accessing whisper model {model_name} (finetuned={finetuned}). Cache contains: {list(ModelManager._whisper_models.keys())}")
        if key not in ModelManager._whisper_models:
            print(f"Loading whisper model {model_name} as it's not in cache")
            ModelManager._whisper_models[key] = load_whisper_model(model_name, finetuned)
        return ModelManager._whisper_models[key]

    @staticmethod
    def load_models(text_models: Collection[str], whisper_models: List[Tuple[str, bool]]) -> None:
        """
        Preload multiple models at once.

        Args:
            text_models: List of text model names to load
            whisper_models: List of (model_name, finetuned) tuples for Whisper models
        """
        # TODO: Needs to do an actual request to be properly cached, exclude this from DB tracking

        print(f"Preloading {len(text_models)} text models and {len(whisper_models)} whisper models")
        for model_name in text_models:
            ModelManager.get_model(model_name)

        for model_name, finetuned in whisper_models:
            ModelManager.get_whisper_model(model_name, finetuned)
        print("Model preloading complete")

    @staticmethod
    def clear_cache() -> None:
        """Clear all cached models"""
        ModelManager._text_models.clear()
        ModelManager._whisper_models.clear()
        print("Model cache cleared")
