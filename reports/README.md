# Background experiment tools

This directory retains executable analysis, experiment launchers, exact
compatibility metadata, and technical validation records needed to reproduce
the background experiments. It is not a destination for report drafts,
presentations, generated PDFs, or saved result-table summaries.

Use the generators' output arguments to save new deliverables outside this
repository. Some historical generators still default to a PDF in this
directory; those outputs are ignored by Git. Do not force-add them.

The source-reference JSON is used by the matched sphere launchers and learned
flow diagnostics. The compatibility JSON and preparation release records are
provenance inputs or technical audit records, not disposable report output.

Historical generated deliverables and one-off scripts were archived before
removal from the release-cleanup branch. The archive is local and is not a
dependency of fitting, training, or the test suite. Removing files from the
branch does not erase earlier Git history.
