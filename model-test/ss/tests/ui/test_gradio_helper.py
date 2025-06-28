import pytest
from unittest.mock import Mock, patch
from PIL import Image
from ui.gradio_helper import GradioHelper


class TestGradioHelper:
    """
    UI 이므로 나머지 것들은 통합테스트로 대체
    """
    def setup_method(self):
        """
        각 테스트 전에 GradioHelper 인스턴스를 생성합니다.
        """
        self.helper = GradioHelper()

    def test_update_canvas_with_eraser(self):
        """
        지우개 모드일 때 캔버스 업데이트가 올바르게 동작하는지 테스트합니다.
        brush_radius와 brush_color가 지우개에 맞게 설정되어야 합니다.
        """
        result = self.helper.update_canvas(use_line=False, use_eraser=True)

        assert result["brush_radius"] == 20
        assert result["brush_color"] == "#ffffff"

    def test_update_canvas_with_line(self):
        """
        선 그리기 모드일 때 캔버스 업데이트가 올바르게 동작하는지 테스트합니다.
        brush_radius와 brush_color가 선 그리기에 맞게 설정되어야 합니다.
        """
        result = self.helper.update_canvas(use_line=True, use_eraser=False)

        assert result["brush_radius"] == 4
        assert result["brush_color"] == "#000000"

    def test_update_canvas_both_false(self):
        """
        use_line과 use_eraser가 모두 False일 때 예외가 발생하는지 테스트합니다.
        """
        with pytest.raises(UnboundLocalError):
            self.helper.update_canvas(use_line=False, use_eraser=False)

    @patch("PIL.Image.open")
    def test_upload_sketch(self, mock_image_open):
        """
        파일 업로드 시 스케치 이미지를 올바르게 처리하는지 테스트합니다.
        PIL.Image.open과 convert 메서드를 mock하여 동작을 검증합니다.
        """
        # Mock file object
        mock_file = Mock()
        mock_file.name = "dummy.png"
        # Mock PIL Image and its convert method
        mock_img = Mock(spec=Image.Image)
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value = mock_img

        result = self.helper.upload_sketch(mock_file)
        mock_image_open.assert_called_once_with("dummy.png")
        mock_img.convert.assert_called_once_with("L")
        assert result["value"] == mock_img
        assert result["source"] == "upload"
        assert result["interactive"] is True
