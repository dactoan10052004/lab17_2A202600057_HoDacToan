from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_agent.seed_semantic import main


if __name__ == "__main__":
    main()
