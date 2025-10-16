import pytest
from src.hal_beam_com.cogs.op_cog import invoke
from src.hal_beam_com.utils import CogType


def test_op_cog_basic_query():
    """
    Test that op_cog can generate a non-empty response with a local model.
    This is a basic smoke test to verify the prompt and model interaction work.
    """
    # Create a simple test query
    data = [{
        'beamline': '11BM',
        'text_input': 'move sample x to 5',
        'include_context_functions': True,
        'only_text_input': 1,
        'operator_cog_history': "",
        'operator_cog_db_history': "",
        'user_id': "test",
    }]
    
    # Use a local model
    base_model = 'qwen2.5-coder'
    
    # Invoke the cog
    result = invoke(
        data,
        base_model=base_model,
        finetuned=False,
        system_prompt_path=None
    )
    
    # Verify we got a result
    assert result is not None, "Result should not be None"
    assert len(result) > 0, "Result should not be empty"
    
    # Extract the generated code
    generated_code = result[0][f'{CogType.OP.value}_cog_output']
    
    # Verify the generated code is not empty
    assert generated_code is not None, "Generated code should not be None"
    assert isinstance(generated_code, str), "Generated code should be a string"
    assert len(generated_code.strip()) > 0, "Generated code should not be empty or whitespace only"
    
    print(f"\nGenerated code:\n{generated_code}")
    print(f"\nTest passed! Generated {len(generated_code)} characters of code.")

if __name__ == '__main__':
    # Allow running the test directly
    test_op_cog_basic_query()