"""
Security Tests for FRA Diagnostics Platform
SIH 2025 PS 25190

Tests cover:
- File upload validation
- Path traversal prevention
- Magic byte verification
- XXE attack prevention
- File size limits
- Binary file rejection
"""

import pytest
import tempfile
from pathlib import Path
import sys
import io

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock Streamlit UploadedFile for testing
class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""
    
    def __init__(self, name: str, content: bytes, size: int = None):
        self.name = name
        self._content = content
        self.size = size if size is not None else len(content)
        self._position = 0
    
    def getvalue(self):
        """Return file content."""
        return self._content
    
    def seek(self, position):
        """Seek to position in file."""
        self._position = position
    
    def read(self):
        """Read from current position."""
        return self._content[self._position:]


class TestFileUploadValidation:
    """Test file upload validation in app.py."""
    
    def test_none_file_rejected(self):
        """Test that None file is rejected."""
        from app import validate_uploaded_file
        
        is_valid, error = validate_uploaded_file(None)
        
        assert not is_valid
        assert "No file uploaded" in error
    
    def test_empty_file_rejected(self):
        """Test that empty file is rejected."""
        from app import validate_uploaded_file
        
        empty_file = MockUploadedFile("test.csv", b"", size=0)
        is_valid, error = validate_uploaded_file(empty_file)
        
        assert not is_valid
        assert "empty" in error.lower()
    
    def test_oversized_file_rejected(self):
        """Test that oversized files are rejected."""
        from app import validate_uploaded_file, MAX_FILE_SIZE_BYTES
        
        # Create file larger than limit
        large_content = b"X" * (MAX_FILE_SIZE_BYTES + 1)
        large_file = MockUploadedFile("huge.csv", large_content)
        
        is_valid, error = validate_uploaded_file(large_file)
        
        assert not is_valid
        assert "exceeds maximum" in error
    
    def test_invalid_extension_rejected(self):
        """Test that invalid file extensions are rejected."""
        from app import validate_uploaded_file
        
        invalid_file = MockUploadedFile("test.exe", b"data")
        is_valid, error = validate_uploaded_file(invalid_file)
        
        assert not is_valid
        assert "not supported" in error
    
    def test_valid_csv_accepted(self):
        """Test that valid CSV file is accepted."""
        from app import validate_uploaded_file
        
        csv_content = b"frequency,magnitude,phase\n20,0.5,-10\n"
        csv_file = MockUploadedFile("test.csv", csv_content)
        
        is_valid, error = validate_uploaded_file(csv_file)
        
        assert is_valid
        assert error == ""
    
    def test_valid_xml_accepted(self):
        """Test that valid XML file is accepted."""
        from app import validate_uploaded_file
        
        xml_content = b"<?xml version='1.0'?><root><data>test</data></root>"
        xml_file = MockUploadedFile("test.xml", xml_content)
        
        is_valid, error = validate_uploaded_file(xml_file)
        
        assert is_valid
        assert error == ""


class TestMagicByteValidation:
    """Test magic byte validation for file type verification."""
    
    def test_binary_file_rejected(self):
        """Test that binary files are rejected."""
        from app import validate_uploaded_file
        
        # Binary file with NULL bytes
        binary_content = b"\x00\x01\x02\x03\x04\x05"
        binary_file = MockUploadedFile("test.csv", binary_content)
        
        is_valid, error = validate_uploaded_file(binary_file)
        
        assert not is_valid
        assert "binary" in error.lower()
    
    def test_executable_disguised_as_csv_rejected(self):
        """Test that executable disguised as CSV is rejected."""
        from app import validate_uploaded_file
        
        # ELF executable header
        elf_content = b"\x7fELF\x02\x01\x01\x00" + b"X" * 100
        elf_file = MockUploadedFile("malicious.csv", elf_content)
        
        is_valid, error = validate_uploaded_file(elf_file)
        
        assert not is_valid
        assert "binary" in error.lower()
    
    def test_xml_with_invalid_header_rejected(self):
        """Test that XML without proper header is rejected."""
        from app import validate_uploaded_file
        
        invalid_xml = b"This is not XML at all"
        xml_file = MockUploadedFile("test.xml", invalid_xml)
        
        is_valid, error = validate_uploaded_file(xml_file)
        
        assert not is_valid
        assert "Invalid XML" in error


class TestXXEPrevention:
    """Test XXE (XML External Entity) attack prevention."""
    
    def test_xxe_with_doctype_rejected(self):
        """Test that XML with DOCTYPE is rejected."""
        from app import validate_uploaded_file
        
        xxe_xml = b"""<?xml version="1.0"?>
<!DOCTYPE foo [
  <!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<root><data>&xxe;</data></root>
"""
        xml_file = MockUploadedFile("malicious.xml", xxe_xml)
        
        is_valid, error = validate_uploaded_file(xml_file)
        
        assert not is_valid
        assert "dangerous content" in error.lower()
    
    def test_xxe_with_entity_rejected(self):
        """Test that XML with ENTITY is rejected."""
        from app import validate_uploaded_file
        
        xxe_xml = b"""<?xml version="1.0"?>
<!DOCTYPE data [
  <!ENTITY file SYSTEM "file:///etc/shadow">
]>
<root>&file;</root>
"""
        xml_file = MockUploadedFile("attack.xml", xxe_xml)
        
        is_valid, error = validate_uploaded_file(xml_file)
        
        assert not is_valid
        assert "dangerous content" in error.lower()
    
    def test_xxe_with_system_rejected(self):
        """Test that XML with SYSTEM keyword is rejected."""
        from app import validate_uploaded_file
        
        xxe_xml = b'<?xml version="1.0"?>\n<!DOCTYPE x [<!ENTITY y SYSTEM "file:///dev/random">]>\n<x>&y;</x>'
        xml_file = MockUploadedFile("malicious.xml", xxe_xml)
        
        is_valid, error = validate_uploaded_file(xml_file)
        
        assert not is_valid
    
    def test_safe_xml_accepted(self):
        """Test that safe XML without dangerous patterns is accepted."""
        from app import validate_uploaded_file
        
        safe_xml = b"""<?xml version="1.0"?>
<root>
  <measurement>
    <frequency>20</frequency>
    <magnitude>0.5</magnitude>
  </measurement>
</root>
"""
        xml_file = MockUploadedFile("safe.xml", safe_xml)
        
        is_valid, error = validate_uploaded_file(xml_file)
        
        assert is_valid


class TestEncodingValidation:
    """Test file encoding validation."""
    
    def test_utf8_file_accepted(self):
        """Test that UTF-8 encoded file is accepted."""
        from app import validate_uploaded_file
        
        utf8_content = "frequency,magnitude,phase\n20,0.5,-10\n".encode('utf-8')
        utf8_file = MockUploadedFile("test.csv", utf8_content)
        
        is_valid, error = validate_uploaded_file(utf8_file)
        
        assert is_valid
    
    def test_iso88591_file_accepted(self):
        """Test that ISO-8859-1 encoded file is accepted."""
        from app import validate_uploaded_file
        
        iso_content = "frequency,magnitude,phase\n20,0.5,-10\n".encode('iso-8859-1')
        iso_file = MockUploadedFile("test.csv", iso_content)
        
        is_valid, error = validate_uploaded_file(iso_file)
        
        assert is_valid
    
    def test_invalid_encoding_rejected(self):
        """Test that file with invalid encoding is rejected."""
        from app import validate_uploaded_file
        
        # Invalid UTF-8 sequence
        invalid_content = b"\xff\xfe\x00\x01invalid\x80\x81"
        invalid_file = MockUploadedFile("test.csv", invalid_content)
        
        is_valid, error = validate_uploaded_file(invalid_file)
        
        assert not is_valid
        assert "encoding not supported" in error.lower()


class TestFileExtensionValidation:
    """Test file extension validation."""
    
    def test_csv_extension_accepted(self):
        """Test .csv extension is accepted."""
        from app import validate_uploaded_file
        
        csv_file = MockUploadedFile("data.csv", b"frequency,mag,phase\n20,0,-10\n")
        is_valid, _ = validate_uploaded_file(csv_file)
        assert is_valid
    
    def test_xml_extension_accepted(self):
        """Test .xml extension is accepted."""
        from app import validate_uploaded_file
        
        xml_file = MockUploadedFile("data.xml", b"<?xml version='1.0'?><root></root>")
        is_valid, _ = validate_uploaded_file(xml_file)
        assert is_valid
    
    def test_txt_extension_accepted(self):
        """Test .txt extension is accepted."""
        from app import validate_uploaded_file
        
        txt_file = MockUploadedFile("data.txt", b"frequency\tmag\tphase\n20\t0\t-10\n")
        is_valid, _ = validate_uploaded_file(txt_file)
        assert is_valid
    
    def test_dat_extension_accepted(self):
        """Test .dat extension is accepted."""
        from app import validate_uploaded_file
        
        dat_file = MockUploadedFile("data.dat", b"frequency,mag,phase\n20,0,-10\n")
        is_valid, _ = validate_uploaded_file(dat_file)
        assert is_valid
    
    def test_case_insensitive_extension(self):
        """Test that extension check is case-insensitive."""
        from app import validate_uploaded_file
        
        # Test various cases
        for ext in ['.CSV', '.Xml', '.TxT', '.DaT']:
            test_file = MockUploadedFile(f"data{ext}", b"data")
            is_valid, _ = validate_uploaded_file(test_file)
            assert is_valid, f"Extension {ext} should be accepted"
    
    def test_dangerous_extensions_rejected(self):
        """Test that dangerous extensions are rejected."""
        from app import validate_uploaded_file
        
        dangerous_extensions = ['.exe', '.sh', '.bat', '.py', '.js', '.php', '.asp']
        
        for ext in dangerous_extensions:
            dangerous_file = MockUploadedFile(f"malicious{ext}", b"data")
            is_valid, error = validate_uploaded_file(dangerous_file)
            assert not is_valid, f"Extension {ext} should be rejected"
            assert "not supported" in error


class TestFileSizeLimits:
    """Test file size limit enforcement."""
    
    def test_file_at_limit_accepted(self):
        """Test that file exactly at size limit is accepted."""
        from app import validate_uploaded_file, MAX_FILE_SIZE_BYTES
        
        # File exactly at limit
        content = b"X" * MAX_FILE_SIZE_BYTES
        file = MockUploadedFile("large.csv", content)
        
        # Size check happens first, content check might fail
        # but we're testing size validation
        is_valid, error = validate_uploaded_file(file)
        # Either accepted or rejected for content (binary), not size
        assert "exceeds maximum" not in error
    
    def test_file_just_over_limit_rejected(self):
        """Test that file just over limit is rejected."""
        from app import validate_uploaded_file, MAX_FILE_SIZE_BYTES
        
        # File 1 byte over limit
        content = b"X" * (MAX_FILE_SIZE_BYTES + 1)
        file = MockUploadedFile("toolarge.csv", content)
        
        is_valid, error = validate_uploaded_file(file)
        
        assert not is_valid
        assert "exceeds maximum" in error


class TestPathTraversalPrevention:
    """Test path traversal prevention in simulator export."""
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        from simulator import TransformerSimulator
        
        sim = TransformerSimulator()
        
        # Test normal filename
        safe = sim._sanitize_filename("normal_file_name.csv")
        assert safe == "normal_file_name.csv"
    
    def test_sanitize_filename_removes_path_separators(self):
        """Test that path separators are removed."""
        from simulator import TransformerSimulator
        
        sim = TransformerSimulator()
        
        # Test with path separators
        safe = sim._sanitize_filename("../../../etc/passwd")
        assert "/" not in safe
        assert "\\" not in safe
        assert ".." not in safe
    
    def test_sanitize_filename_prevents_parent_directory(self):
        """Test that parent directory references are blocked."""
        from simulator import TransformerSimulator
        
        sim = TransformerSimulator()
        
        # Test dot and double-dot
        with pytest.raises(ValueError):
            sim._sanitize_filename(".")
        
        with pytest.raises(ValueError):
            sim._sanitize_filename("..")
    
    def test_sanitize_filename_removes_dangerous_chars(self):
        """Test that dangerous characters are removed."""
        from simulator import TransformerSimulator
        
        sim = TransformerSimulator()
        
        dangerous = "file<name>with:bad|chars?.txt"
        safe = sim._sanitize_filename(dangerous)
        
        # Should not contain dangerous characters
        assert "<" not in safe
        assert ">" not in safe
        assert ":" not in safe
        assert "|" not in safe
        assert "?" not in safe
    
    def test_sanitize_filename_prevents_empty(self):
        """Test that empty filename after sanitization raises error."""
        from simulator import TransformerSimulator
        
        sim = TransformerSimulator()
        
        # String that becomes empty after sanitization
        with pytest.raises(ValueError, match="empty"):
            sim._sanitize_filename("<<<>>>")
    
    def test_sanitize_filename_removes_leading_dots(self):
        """Test that leading dots are removed (prevents hidden files)."""
        from simulator import TransformerSimulator
        
        sim = TransformerSimulator()
        
        safe = sim._sanitize_filename("...hidden_file.txt")
        assert not safe.startswith(".")


class TestSecurityBestPractices:
    """Test general security best practices."""
    
    def test_no_eval_or_exec_in_parser(self):
        """Test that parser doesn't use eval or exec."""
        from parser import UniversalFRAParser
        import inspect
        
        parser = UniversalFRAParser()
        source = inspect.getsource(UniversalFRAParser)
        
        # Should not contain eval or exec
        assert "eval(" not in source
        assert "exec(" not in source
    
    def test_no_shell_commands_in_simulator(self):
        """Test that simulator doesn't execute shell commands."""
        from simulator import TransformerSimulator
        import inspect
        
        sim = TransformerSimulator()
        source = inspect.getsource(TransformerSimulator)
        
        # Should not contain os.system or subprocess
        assert "os.system" not in source
        assert "subprocess.call" not in source
        assert "subprocess.run" not in source or "# subprocess.run" in source  # Allow in comments


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
