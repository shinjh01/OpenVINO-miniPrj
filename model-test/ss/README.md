

pytest 테스트 실행법 
1. model-test/ss에서 pytest 실행 - 전체 테스트 
2. pytest ./tests/pipelines/test_pix2pix_twinconv.py 등과 같이  특정파일 지정하여 실행

pytest 커버리지 확인법
pytest --cov=src --cov-report=term-missing --cov-report=html --cov-branch --cov-config=.coveragerc