from unittest.mock import patch, MagicMock

# main.py의 main 함수를 임포트
from src import main as main_module


def test_main_launch_called(monkeypatch):
    """
    main.py의 main 함수가 Gradio demo의 launch를 호출하는지 테스트합니다.

    입력:
        - 없음 (main 함수 직접 호출)
    출력:
        - demo.queue().launch(debug=True)가 호출됨
    예시:
        >>> test_main_launch_called(monkeypatch)
        (에러 없이 통과)
    설명:
        GradioHelper.make_demo와 Inference.run을 mock 처리하여 실제 서버 실행 없이 launch 호출 여부만 검증합니다.
    """
    # GradioHelper.make_demo와 Inference.run을 mock
    mock_demo = MagicMock()
    mock_queue = MagicMock()
    mock_demo.queue.return_value = mock_queue
    mock_queue.launch.return_value = None

    with patch('src.main.GradioHelper') as MockGradioHelper, \
         patch('src.main.Inference') as MockInference:
        MockGradioHelper.return_value.make_demo.return_value = mock_demo
        MockInference.return_value.run = MagicMock()

        # 예외 없이 launch가 호출되는지 확인
        main_module.main()
        mock_demo.queue.assert_called()
        mock_queue.launch.assert_called_with(debug=True)


def test_main_launch_share_on_exception(monkeypatch):
    """
    main.py의 main 함수에서 launch가 예외 발생 시 share=True로 재시도하는지 테스트합니다.

    입력:
        - 없음 (main 함수 직접 호출)
    출력:
        - 첫 launch(debug=True)에서 예외 발생 시, launch(debug=True, share=True)로 재시도
    예시:
        >>> test_main_launch_share_on_exception(monkeypatch)
        (에러 없이 통과)
    설명:
        queue().launch의 첫 호출에서 예외를 발생시키고, 두 번째 호출에서 정상 동작하는지 검증합니다.
    """
    mock_demo = MagicMock()
    mock_queue = MagicMock()
    # 첫 번째 launch는 예외, 두 번째는 정상
    mock_queue.launch.side_effect = [Exception("fail"), None]
    mock_demo.queue.return_value = mock_queue

    with patch('src.main.GradioHelper') as MockGradioHelper, \
         patch('src.main.Inference') as MockInference:
        MockGradioHelper.return_value.make_demo.return_value = mock_demo
        MockInference.return_value.run = MagicMock()

        main_module.main()
        assert mock_queue.launch.call_count == 2
        mock_queue.launch.assert_called_with(debug=True, share=True)
