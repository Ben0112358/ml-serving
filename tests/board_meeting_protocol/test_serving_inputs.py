import base64

import pytest
from fastapi import HTTPException


@pytest.mark.parametrize(
    "filename", ["minutes.docx", "minutes.pdf", "minutes.odt"]
)
def test_unreadable_document_is_rejected_as_bad_request(
    ml_serving_env, filename
):
    from ml_serving.board_meeting_protocol.serving import FilePart, _text_of

    part = FilePart(
        filename=filename,
        content_base64=base64.b64encode(b"not a document").decode("ascii"),
    )
    with pytest.raises(HTTPException) as caught:
        _text_of(part)
    assert caught.value.status_code == 400
