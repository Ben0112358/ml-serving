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


def _document_part(filename, add_members=None):
    import io
    import zipfile

    from docx import Document
    from odf.opendocument import OpenDocumentText
    from odf.text import P

    from ml_serving.board_meeting_protocol.serving import FilePart

    buffer = io.BytesIO()
    if filename.endswith(".docx"):
        document = Document()
        document.add_paragraph("Budget approved")
    else:
        document = OpenDocumentText()
        document.text.addElement(P(text="Budget approved"))
    document.save(buffer)
    if add_members is not None:
        with zipfile.ZipFile(buffer, "a", zipfile.ZIP_DEFLATED) as archive:
            add_members(archive)
    return FilePart(
        filename=filename,
        content_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def _add_large_member(archive):
    from ml_serving.board_meeting_protocol.serving import (
        MAX_UNCOMPRESSED_BYTES,
    )

    chunk = b"\0" * 1_000_000
    with archive.open("padding.bin", "w") as member:
        remaining = MAX_UNCOMPRESSED_BYTES
        while remaining:
            size = min(remaining, len(chunk))
            member.write(chunk[:size])
            remaining -= size


def _add_many_members(archive):
    from ml_serving.board_meeting_protocol.serving import MAX_ARCHIVE_MEMBERS

    for index in range(MAX_ARCHIVE_MEMBERS):
        archive.writestr(f"padding/{index}.xml", b"")


@pytest.mark.parametrize("filename", ["minutes.docx", "minutes.odt"])
@pytest.mark.parametrize("add_members", [_add_large_member, _add_many_members])
def test_oversized_archive_is_rejected_as_bad_request(
    ml_serving_env, filename, add_members
):
    from ml_serving.board_meeting_protocol.serving import _text_of

    part = _document_part(filename, add_members)
    with pytest.raises(HTTPException) as caught:
        _text_of(part)
    assert caught.value.status_code == 400
    assert caught.value.detail == f"Could not read {filename}"


def test_oversized_archive_template_is_rejected_as_bad_request(
    ml_serving_env,
):
    from ml_serving.board_meeting_protocol.serving import (
        RowIn,
        _style_template,
    )

    row = RowIn(middle=_document_part("finished.docx", _add_many_members))
    with pytest.raises(HTTPException) as caught:
        _style_template([row], "new.docx")
    assert caught.value.status_code == 400


@pytest.mark.parametrize("filename", ["minutes.docx", "minutes.odt"])
def test_regular_archive_document_is_read(ml_serving_env, filename):
    from ml_serving.board_meeting_protocol.serving import _text_of

    assert "Budget approved" in _text_of(_document_part(filename))
