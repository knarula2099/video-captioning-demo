#!/usr/bin/env python3
"""
Test script to verify Claude API client and key work correctly.
"""

import os
import sys
from typing import Optional

def test_claude_api(api_key: Optional[str] = None):
    """Test Claude API connection and functionality."""
    
    print("🧪 Testing Claude API Connection")
    print("=" * 50)
    
    # Step 1: Check if anthropic library is installed
    try:
        import anthropic
        print("✅ Anthropic library imported successfully")
        print(f"   Version: {anthropic.__version__ if hasattr(anthropic, '__version__') else 'Unknown'}")
    except ImportError as e:
        print(f"❌ Failed to import anthropic library: {e}")
        print("   Install with: pip install anthropic")
        return False
    
    # Step 2: Get API key
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("❌ No API key provided")
        print("   Set ANTHROPIC_API_KEY environment variable or pass as argument")
        return False
    
    print("✅ API key found")
    print(f"   Key preview: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    
    # Step 3: Initialize client
    try:
        client = anthropic.Anthropic(api_key=api_key)
        print("✅ Claude client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Claude client: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    
    # Step 4: Test simple text message
    print("\n🔍 Testing simple text message...")
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",  # Using a stable model
            max_tokens=50,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": "Say 'Hello, I am Claude and I'm working correctly!' in exactly those words."
                }
            ]
        )
        
        response_text = message.content[0].text.strip()
        print(f"✅ Text message successful")
        print(f"   Response: {response_text}")
        
        # Check if response is what we expected
        expected = "Hello, I am Claude and I'm working correctly!"
        if expected.lower() in response_text.lower():
            print("✅ Response content matches expected output")
        else:
            print("⚠️  Response content differs from expected, but API is working")
            
    except Exception as e:
        print(f"❌ Text message failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False
    
    # Step 5: Test model availability
    print("\n🔍 Testing different Claude models...")
    models_to_test = [
        "claude-3-5-sonnet-20241022",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307"
    ]
    
    working_models = []
    for model in models_to_test:
        try:
            test_message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            working_models.append(model)
            print(f"✅ {model} - Working")
        except Exception as e:
            print(f"❌ {model} - Error: {str(e)[:100]}...")
    
    # Step 6: Test Claude 4 models if available
    print("\n🔍 Testing Claude 4 models...")
    claude4_models = [
        "claude-4-opus-20250514",
        "claude-sonnet-4-20250514"
    ]
    
    for model in claude4_models:
        try:
            test_message = client.messages.create(
                model=model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}]
            )
            working_models.append(model)
            print(f"✅ {model} - Working")
        except Exception as e:
            print(f"❌ {model} - Error: {str(e)[:100]}...")
    
    # Step 7: Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"✅ API Key: Valid")
    print(f"✅ Client: Initialized successfully")
    print(f"✅ Basic functionality: Working")
    print(f"✅ Working models: {len(working_models)}")
    
    if working_models:
        print("   Available models:")
        for model in working_models:
            print(f"   - {model}")
    
    print("\n🎉 Claude API is working correctly!")
    return True

def test_with_image():
    """Test Claude with image if available."""
    print("\n🖼️  Testing image capabilities...")
    
    # You can add image testing here if needed
    # This would require a test image file
    print("   (Image testing skipped - add test image if needed)")

if __name__ == "__main__":
    # Get API key from command line argument or environment
    api_key = None
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    print("Claude API Test Script")
    print("=" * 50)
    
    if not api_key:
        print("Usage: python test_claude.py [API_KEY]")
        print("Or set ANTHROPIC_API_KEY environment variable")
        print()
    
    success = test_claude_api(api_key)
    
    if success:
        print("\n✅ All tests passed! Your Claude setup is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)