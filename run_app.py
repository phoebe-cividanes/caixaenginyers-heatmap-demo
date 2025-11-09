#!/usr/bin/env python3
"""
Quick launcher for the Caixa Enginyers Streamlit App
"""

import subprocess
import sys
from pathlib import Path

def main():
    app_path = Path(__file__).parent / "app data" / "streamlit_app_scored.py"
    
    if not app_path.exists():
        print(f"❌ App not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Launching Caixa Enginyers Branch Location Optimizer...")
    print(f"📂 App: {app_path}")
    print("\n🌐 The app will open in your browser at http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server\n")
    
    try:
        subprocess.run([
            "streamlit", "run", str(app_path),
            "--server.headless", "true",
            "--server.fileWatcherType", "none",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 App stopped")
    except FileNotFoundError:
        print("\n❌ Streamlit not found. Install with:")
        print("   pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
