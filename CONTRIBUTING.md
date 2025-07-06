# Contributing to DocuLLaMA

Thank you for your interest in contributing to DocuLLaMA! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository**
   ```bash
   git clone https://github.com/snedea/docullama.git
   cd docullama
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Run tests**
   ```bash
   pytest
   ```

## 🛠️ Development Guidelines

### Code Style

We use the following tools for code quality:

- **Black** for code formatting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing

Run these before submitting:
```bash
black .
flake8 .
mypy .
pytest
```

### Commit Messages

Use conventional commit format:
```
feat: add new document processing feature
fix: resolve Qdrant connection issue
docs: update API documentation
test: add unit tests for RAG engine
```

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `test/description` - Test improvements

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_rag_engine.py

# Run tests with specific marker
pytest -m "not slow"
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Include both unit and integration tests
- Mock external dependencies (Azure OpenAI, Qdrant)

Example test structure:
```python
import pytest
from unittest.mock import Mock, patch
from app import create_app

class TestRAGEngine:
    def test_search_documents(self):
        # Test implementation
        pass
    
    @pytest.mark.asyncio
    async def test_async_processing(self):
        # Async test implementation
        pass
```

## 📋 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   pytest
   black .
   flake8 .
   mypy .
   ```

4. **Submit pull request**
   - Use a descriptive title
   - Include a detailed description
   - Reference any related issues
   - Ensure CI checks pass

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests
- [ ] Updated existing tests

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🏗️ Architecture Guidelines

### Adding New Features

1. **Document processing features** → `document_processor.py`
2. **RAG capabilities** → `rag_engine.py`
3. **Knowledge graph features** → `knowledge_graph.py`
4. **Research intelligence** → `research_intelligence.py`
5. **API endpoints** → `app.py`
6. **Configuration** → `config.py`
7. **Monitoring** → `monitoring.py`

### Database Changes

- Update Qdrant collection schemas carefully
- Provide migration scripts if needed
- Test with both empty and populated collections

### API Changes

- Maintain backward compatibility
- Update OpenAPI documentation
- Add comprehensive tests
- Consider rate limiting and authentication

## 🐛 Bug Reports

When reporting bugs, include:

1. **Environment information**
   - Python version
   - Operating system
   - Azure Container Apps version
   - Dependency versions

2. **Steps to reproduce**
   - Detailed steps
   - Expected behavior
   - Actual behavior

3. **Logs and error messages**
   - Complete error traces
   - Relevant log entries
   - Screenshots if applicable

4. **Configuration**
   - Environment variables (redact secrets)
   - Container app settings
   - Qdrant configuration

## 💡 Feature Requests

For new features:

1. **Check existing issues** first
2. **Describe the use case** clearly
3. **Explain the benefit** to users
4. **Consider implementation** complexity
5. **Provide examples** if possible

## 🔒 Security

### Reporting Security Issues

**Do not create public issues for security vulnerabilities.**

Instead:
1. Email security concerns to the maintainers
2. Include detailed description
3. Provide steps to reproduce
4. Allow time for response and fix

### Security Guidelines

- Never commit secrets or API keys
- Use Azure Key Vault for sensitive data
- Validate all user inputs
- Implement proper authentication
- Follow OWASP guidelines

## 📖 Documentation

### Code Documentation

- Use clear docstrings for all functions/classes
- Include type hints
- Document complex algorithms
- Provide usage examples

Example:
```python
async def process_document(
    self,
    file: UploadFile,
    collection_name: str,
    additional_metadata: Optional[Dict[str, Any]] = None
) -> ProcessingResult:
    """
    Process uploaded document with AutoLLaMA features.
    
    Args:
        file: Uploaded file object
        collection_name: Target collection for storage
        additional_metadata: Optional metadata to include
        
    Returns:
        ProcessingResult with status and chunk information
        
    Raises:
        ValueError: If file format is unsupported
        ProcessingError: If document processing fails
    """
```

### API Documentation

- Update OpenAPI specs for new endpoints
- Include request/response examples
- Document error codes and messages
- Provide curl examples

## 🌍 Internationalization

Currently, DocuLLaMA supports English. For international contributions:

- Use English for code comments and documentation
- Consider locale-specific document processing
- Test with non-ASCII content
- Plan for future i18n support

## 📞 Getting Help

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - Questions and community support
- **Documentation** - Check `/docs` endpoint when running

## 🏆 Recognition

Contributors will be:
- Added to the contributors list
- Mentioned in release notes for significant contributions
- Invited to join the maintainer team for outstanding contributions

Thank you for contributing to DocuLLaMA! 🚀