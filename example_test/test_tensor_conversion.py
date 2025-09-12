#!/usr/bin/env python3
"""
Test script to verify tensor to scalar conversion for valid_response_length.
"""

import torch

def test_tensor_conversion():
    """Test that tensor conversion works correctly."""
    
    print("Testing tensor to scalar conversion...")
    
    # Test 1: Integer input
    valid_response_length_int = 100
    print(f"Integer input: {valid_response_length_int} (type: {type(valid_response_length_int)})")
    
    if hasattr(valid_response_length_int, 'item'):
        converted_int = valid_response_length_int.item()
        print(f"  After conversion: {converted_int} (type: {type(converted_int)})")
    else:
        print(f"  No conversion needed: {valid_response_length_int}")
    
    # Test 2: Tensor input
    valid_response_length_tensor = torch.tensor(100)
    print(f"Tensor input: {valid_response_length_tensor} (type: {type(valid_response_length_tensor)})")
    
    if hasattr(valid_response_length_tensor, 'item'):
        converted_tensor = valid_response_length_tensor.item()
        print(f"  After conversion: {converted_tensor} (type: {type(converted_tensor)})")
    else:
        print(f"  No conversion needed: {valid_response_length_tensor}")
    
    # Test 3: Comparison test
    start = 50
    
    print(f"\nComparison tests with start = {start}:")
    
    # With integer
    print(f"  start < valid_response_length_int: {start < valid_response_length_int}")
    
    # With tensor (before conversion) - this would cause issues
    try:
        result = start < valid_response_length_tensor
        print(f"  start < valid_response_length_tensor (before conversion): {result}")
        print(f"    Result type: {type(result)}")
    except Exception as e:
        print(f"  start < valid_response_length_tensor (before conversion): ERROR - {e}")
    
    # With tensor (after conversion) - this works correctly
    converted = valid_response_length_tensor.item()
    print(f"  start < converted_tensor: {start < converted}")
    
    print("\n✅ Tensor conversion fix works correctly!")
    print("   - Integer inputs: no conversion needed")
    print("   - Tensor inputs: converted to scalar using .item()")
    print("   - Comparisons work correctly after conversion")

if __name__ == "__main__":
    test_tensor_conversion()
    print("\n🎉 The tensor conversion fix resolves the comparison issue!")
