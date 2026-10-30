#!/usr/bin/env python3
"""
🧠 ULTIMATE MENTAL HEALTH ANALYTICS PLATFORM v3.0
One-Click Launcher with Automatic Setup
"""

import subprocess
import sys
import os
import time
import webbrowser
import platform
import json

def print_header():
    """Display beautiful header"""
    print("\n" + "="*80)
    print("🧠 ULTIMATE MENTAL HEALTH ANALYTICS PLATFORM v3.0")
    print("AI-Powered Wellness Intelligence")
    print("="*80 + "\n")

def check_python_version():
    """Verify Python 3.9+"""
    version = sys.version_info
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)

def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'
        ], check=True, cwd=os.path.dirname(__file__))
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)

def train_model():
    """Train ML models if needed"""
    if not os.path.exists('best_model.pkl'):
        print("\n🧠 Training ML models (first run only)...")
        print("   This may take 1-2 minutes...")
        try:
            subprocess.run([
                sys.executable, 'train_advanced_model.py'
            ], check=True, cwd=os.path.dirname(__file__))
            print("✅ Models trained and saved")
        except subprocess.CalledProcessError as e:
            print(f"❌ Model training failed: {e}")
            sys.exit(1)
    else:
        print("✅ Using pre-trained model")

def start_server():
    """Start FastAPI server"""
    cwd = os.path.dirname(__file__)
    
    print("\n🚀 Starting server...")
    print("   URL: http://127.0.0.1:8000")
    print("   Dashboard: http://127.0.0.1:8000/docs (API docs)")
    
    cmd = [sys.executable, '-m', 'uvicorn', 'app:app', '--host', '127.0.0.1', '--port', '8000']
    
    try:
        # Wait for server to start
        time.sleep(2)
        
        # Open browser
        print("\n🌐 Opening browser...")
        webbrowser.open('http://127.0.0.1:8000')
        print("✅ Browser opened")
        
        # Start server
        print("\n" + "="*80)
        print("✅ SERVER IS RUNNING!")
        print("="*80)
        print("\n📊 Available endpoints:")
        print("   • GET  http://127.0.0.1:8000/")
        print("   • GET  http://127.0.0.1:8000/api/health")
        print("   • POST http://127.0.0.1:8000/api/predict")
        print("   • GET  http://127.0.0.1:8000/api/stats")
        print("   • GET  http://127.0.0.1:8000/api/dashboard")
        print("   • GET  http://127.0.0.1:8000/api/sample-profiles")
        print("\n📖 API Documentation: http://127.0.0.1:8000/docs")
        print("\n🛑 Press Ctrl+C to stop the server\n")
        
        subprocess.run(cmd, cwd=cwd)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    print_header()
    
    print("⏳ Initializing Ultimate Mental Health Analytics Platform...")
    print("   1. Checking Python version...")
    check_python_version()
    
    print("   2. Installing dependencies...")
    install_dependencies()
    
    print("   3. Preparing ML models...")
    train_model()
    
    print("   4. Starting server...")
    start_server()

if __name__ == '__main__':
    main()
