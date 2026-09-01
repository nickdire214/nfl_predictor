"""Read-only local dashboard for reviewing weekly prediction logs.

This package NEVER writes to data/, never triggers a model run, and never
edits an override file. It opens parquet logs and renders them.
"""
