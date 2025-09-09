#!/usr/bin/env python3
"""
Explanation of the _handle_pending_state flow and when the engine is called.
"""

def explain_pending_state_flow():
    """
    Explain the flow of _handle_pending_state and when the engine is called.
    """
    print("🔄 SGLangRollout State Flow Explanation")
    print("=" * 60)
    
    print("\n1. 📋 Initial State:")
    print("-" * 30)
    print("When a request starts, it begins in PENDING state")
    print("_req.state = AsyncRolloutRequestStateEnum.PENDING")
    
    print("\n2. 🔄 Main Loop Flow:")
    print("-" * 30)
    print("while current_turns < max_assistant_turns:")
    print("  if _req.state == PENDING:")
    print("    await self._handle_pending_state(_req)  # ← FIRST STEP")
    print("    _req.state = RUNNING")
    print("  elif _req.state == RUNNING:")
    print("    output = await self._handle_engine_call(_req, ...)  # ← ENGINE CALL")
    print("    # Process output and determine next state")
    print("  elif _req.state == TOOL_CALLING:")
    print("    # Handle tool calls")
    print("    _req.state = RUNNING")
    print("  elif _req.state == INTERACTING:")
    print("    # Handle interactions")
    print("    _req.state = RUNNING")
    
    print("\n3. 🎯 _handle_pending_state() - What it does:")
    print("-" * 50)
    print("✅ Does NOT call the engine")
    print("✅ Sets up tools (if any)")
    print("✅ Sets up interactions (if any)")
    print("✅ Changes state from PENDING → RUNNING")
    
    print("\n4. 🚀 _handle_engine_call() - When engine is called:")
    print("-" * 50)
    print("✅ Called in RUNNING state")
    print("✅ This is where the actual LLM generation happens")
    print("✅ Returns generated text and metadata")
    
    print("\n5. 📊 State Transitions:")
    print("-" * 30)
    print("PENDING → RUNNING (after _handle_pending_state)")
    print("RUNNING → TOOL_CALLING (if tool calls detected)")
    print("RUNNING → INTERACTING (if interaction needed)")
    print("TOOL_CALLING → RUNNING (after tool execution)")
    print("INTERACTING → RUNNING (after interaction)")
    
    print("\n6. 🔍 Key Insight:")
    print("-" * 20)
    print("_handle_pending_state() is the FIRST step but does NOT call the engine")
    print("The engine is called in the RUNNING state via _handle_engine_call()")
    print("This allows for proper setup before any generation happens")


def show_code_flow():
    """
    Show the actual code flow with line numbers.
    """
    print("\n7. 📝 Code Flow with Line Numbers:")
    print("-" * 40)
    
    print("Line 841: while current_turns < self.config.multi_turn.max_assistant_turns:")
    print("Line 842:   if _req.state == AsyncRolloutRequestStateEnum.PENDING:")
    print("Line 843:     await self._handle_pending_state(_req)  # ← FIRST STEP")
    print("Line 844:     _req.state = AsyncRolloutRequestStateEnum.RUNNING")
    print("Line 867:   elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:")
    print("Line 893:     output = await self._handle_engine_call(_req, ...)  # ← ENGINE CALL")
    
    print("\n_handle_pending_state() (Line 1040):")
    print("  - Sets up tools (Lines 1041-1050)")
    print("  - Sets up interactions (Lines 1051-1062)")
    print("  - Does NOT call engine")
    
    print("\n_handle_engine_call() (Line 1014):")
    print("  - Calls self._handle_engine_generate() (Line 1018)")
    print("  - Decodes output (Line 1019)")
    print("  - Returns generated text")


def show_timing():
    """
    Show the timing of when things happen.
    """
    print("\n8. ⏰ Timing Sequence:")
    print("-" * 30)
    
    print("Step 1: PENDING state")
    print("  - _handle_pending_state() runs")
    print("  - Sets up tools and interactions")
    print("  - Changes state to RUNNING")
    print("  - NO engine call yet")
    
    print("\nStep 2: RUNNING state")
    print("  - _handle_engine_call() runs")
    print("  - Engine generates text")
    print("  - Text is decoded")
    print("  - State may change based on output")
    
    print("\nStep 3: Subsequent states")
    print("  - TOOL_CALLING: Execute tools")
    print("  - INTERACTING: Handle interactions")
    print("  - Back to RUNNING for next generation")


def main():
    """Main explanation function."""
    explain_pending_state_flow()
    show_code_flow()
    show_timing()
    
    print("\n🎯 Answer to your question:")
    print("=" * 40)
    print("✅ _handle_pending_state() runs at the FIRST step")
    print("❌ It does NOT call the engine")
    print("✅ The engine is called later in RUNNING state")
    print("✅ _handle_pending_state() only sets up tools and interactions")


if __name__ == "__main__":
    main()
