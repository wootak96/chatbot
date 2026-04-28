"""Each node returns a partial RAGState dict, plus a `progress_message`
(consumed by the SSE layer and stripped before merging back into state).
"""
PROGRESS_KEY = "progress_message"
