"""Shared helpers. Import the submodule you need (`utils.naming`, `utils.ratings`, ...).

This file deliberately imports nothing: any import here runs whenever any
`utils` submodule is imported, so `score` would load the LLM stack just to read
a rating label.
"""
