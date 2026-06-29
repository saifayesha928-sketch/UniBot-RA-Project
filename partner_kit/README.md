# Partner Kit

These are the record formats the ingester reads. Copy these dataclasses into
your own data tooling so the files you produce (`data/sources.json` and
`data/records.jsonl`) match what `scripts/ingest_records.py` expects.

## What is here

```
partner_kit/
  __init__.py          schema version pin
  contracts/
    __init__.py        public exports
    enums.py           allowed values for the typed fields
    records.py         SourceRegistryEntry, ExtractedRecord, DocumentAssetRecord
```

`SourceRegistryEntry` is one entry in `sources.json`. `ExtractedRecord` is one
line in `records.jsonl`. `DocumentAssetRecord` is a helper for records that
point at a PDF or DOCX file, with a `to_extracted_record()` method.

## Quick start

```python
from dataclasses import asdict
import json

from partner_kit.contracts import ExtractedRecord, SourceRegistryEntry

source = SourceRegistryEntry(
    source_url="https://your.edu/admissions/",
    canonical_url="https://your.edu/admissions",
    source_class="admissions_cycle",
    crawl_method="html_static",
    legal_status="allowed",
    default_authority_tier=1,
    refresh_policy="weekly",
)

record = ExtractedRecord(
    record_id="admissions_cycle:application-deadline",
    record_type="admissions_cycle",
    source_url=source.source_url,
    source_section_id="calendar-table",
    source_section_label="Admissions Calendar",
    source_locator="body",
    source_authority_tier=1,
    conflict_scope_id="admissions_cycle:application-deadline",
    dedupe_key="admissions_cycle:application-deadline",
    cycle_label="2026-2027",
    year_confidence="high",
    record_payload={
        "milestone_name": "Application Deadline",
        "date_text": "3rd March 2026",
        "normalized_dates": ["2026-03-03"],
    },
)

# write one record per line to records.jsonl
print(json.dumps(asdict(record), default=str))
```

See the top level `README.md` for how to load these files and run the backend.
