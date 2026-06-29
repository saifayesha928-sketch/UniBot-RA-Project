"""Document parsers.

This package ships the generic document parsers (Docling / ADE / cached) used
by :mod:`unibot.extract.documents` to turn PDF/DOCX assets into structured
``ParsedDocument`` objects. Source-specific HTML page parsers are not part of
the serving distribution — partners deliver already-extracted records
(``data/records.jsonl``) and do not run the HTML collection step.
"""
