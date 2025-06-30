"""
conftest.py
pytest에서 테스트 경로 문제를 해결하기 위한 설정 파일입니다.

- src 디렉토리를 sys.path에 추가하여, 테스트 코드에서 src 내부 모듈을 import할 수 있도록 합니다.
- pytest가 자동으로 로드하는 파일로, 별도의 import 없이도 적용됩니다.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))