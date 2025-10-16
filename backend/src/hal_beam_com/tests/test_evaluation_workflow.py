import pytest
from src.hal_beam_com.workflows.evaluation_workflow import run as evaluation_workflow


def test_evaluation_workflow_basic():
    """
    Test that evaluation_workflow can generate a non-empty evaluation with a local model.
    This is a basic smoke test to verify the prompt and model interaction work.
    """
    # Create sample simulation data
    data = [{
        'sim_id': 'test-sim-123',
        'beamline': '11BM',
        'original_query': 'move sample x to 5 and measure for 10 seconds',
        'generated_code': '''sam.x.move(5)
count([det], num=1, delay=10)''',
        'pv_events': [
            {
                'type': 'pv',
                'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr',
                'value': 5.0,
                'timestamp': 1234567890.0,
                'elapsed_time': 0.523,
                'delta_time': 0.523
            },
            {
                'type': 'pv',
                'pvname': 'XF:11BMB-ES{Det:PIL2M}:cam1:AcquireTime',
                'value': 10.0,
                'timestamp': 1234567890.6,
                'elapsed_time': 1.123,
                'delta_time': 0.600
            },
            {
                'type': 'pv',
                'pvname': 'XF:11BMB-ES{Det:PIL2M}:cam1:Acquire',
                'value': 1,
                'timestamp': 1234567891.0,
                'elapsed_time': 1.523,
                'delta_time': 0.400
            }
        ]
    }]
    
    # Invoke the evaluation workflow
    result = evaluation_workflow(data)
    
    # Verify we got a result
    assert result is not None, "Result should not be None"
    assert len(result) > 0, "Result should not be empty"
    
    # Extract the evaluation
    evaluation = result[0].get('evaluation')
    status = result[0].get('status')
    
    # Verify the evaluation is not empty
    assert evaluation is not None, "Evaluation should not be None"
    assert isinstance(evaluation, str), "Evaluation should be a string"
    assert len(evaluation.strip()) > 0, "Evaluation should not be empty or whitespace only"
    assert status == 'success', f"Status should be 'success', got '{status}'"
    
    print(f"\nEvaluation result:\n{evaluation}")
    print(f"\nTest passed! Generated {len(evaluation)} characters of evaluation.")


def test_evaluation_workflow_no_pv_changes():
    """
    Test evaluation when no PV changes occurred during simulation.
    """
    data = [{
        'sim_id': 'test-sim-456',
        'beamline': '11BM',
        'original_query': 'print hello world',
        'generated_code': 'print("hello world")',
        'pv_events': []  # No PV events
    }]
    
    result = evaluation_workflow(data)
    
    assert result is not None
    assert result[0].get('evaluation') is not None
    assert result[0].get('status') == 'success'
    
    print(f"\nNo PV changes evaluation:\n{result[0]['evaluation']}")


def test_evaluation_workflow_complex_scenario():
    """
    Test evaluation with a more complex scenario involving multiple PV changes.
    """
    data = [{
        'sim_id': 'test-sim-789',
        'beamline': '11BM',
        'original_query': 'scan sample x from 0 to 10 in steps of 2, measuring at each point',
        'generated_code': '''for x_pos in range(0, 11, 2):
    sam.x.move(x_pos)
    count([det], num=1)''',
        'pv_events': [
            {'type': 'pv', 'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr', 'value': 0.0, 
             'timestamp': 1000.0, 'elapsed_time': 0.1, 'delta_time': 0.1},
            {'type': 'pv', 'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr', 'value': 2.0,
             'timestamp': 1001.0, 'elapsed_time': 1.1, 'delta_time': 1.0},
            {'type': 'pv', 'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr', 'value': 4.0,
             'timestamp': 1002.0, 'elapsed_time': 2.1, 'delta_time': 1.0},
            {'type': 'pv', 'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr', 'value': 6.0,
             'timestamp': 1003.0, 'elapsed_time': 3.1, 'delta_time': 1.0},
            {'type': 'pv', 'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr', 'value': 8.0,
             'timestamp': 1004.0, 'elapsed_time': 4.1, 'delta_time': 1.0},
            {'type': 'pv', 'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr', 'value': 10.0,
             'timestamp': 1005.0, 'elapsed_time': 5.1, 'delta_time': 1.0},
        ]
    }]
    
    result = evaluation_workflow(data)
    
    assert result is not None
    assert result[0].get('evaluation') is not None
    assert result[0].get('status') == 'success'
    
    evaluation = result[0]['evaluation']
    print(f"\nComplex scenario evaluation:\n{evaluation}")


def test_evaluation_workflow_mismatched_behavior():
    """
    Test evaluation where generated code doesn't match the user's intent.
    """
    data = [{
        'sim_id': 'test-sim-999',
        'beamline': '11BM',
        'original_query': 'move sample x to 10',  # User asked for 10
        'generated_code': 'sam.x.move(5)',  # But code moves to 5
        'pv_events': [
            {
                'type': 'pv',
                'pvname': 'XF:11BMB-ES{Chm:Smpl-Ax:X}Mtr',
                'value': 5.0,  # Moved to 5, not 10
                'timestamp': 1234567890.0,
                'elapsed_time': 0.523,
                'delta_time': 0.523
            }
        ]
    }]
    
    result = evaluation_workflow(data)
    
    assert result is not None
    evaluation = result[0].get('evaluation')
    assert evaluation is not None
    
    # The evaluation should ideally identify the mismatch
    print(f"\nMismatched behavior evaluation:\n{evaluation}")


if __name__ == '__main__':
    # Allow running the tests directly
    print("=" * 80)
    print("Running basic evaluation workflow test...")
    print("=" * 80)
    test_evaluation_workflow_basic()
    
    print("\n" + "=" * 80)
    print("Running no PV changes test...")
    print("=" * 80)
    test_evaluation_workflow_no_pv_changes()
    
    print("\n" + "=" * 80)
    print("Running complex scenario test...")
    print("=" * 80)
    test_evaluation_workflow_complex_scenario()
    
    print("\n" + "=" * 80)
    print("Running mismatched behavior test...")
    print("=" * 80)
    test_evaluation_workflow_mismatched_behavior()
    
    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
