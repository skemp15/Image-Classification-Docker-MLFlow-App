"""
Run EDGAR Script
"""

__date__ = "2023-03-05"
__author__ = "SamKemp"



# %% --------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import edgar_downloader as ed
import edgar_cleaner as ec
import edgar_sentiment_wordcount as esw
import ref_data as rf

# %% --------------------------------------------------------------------------
# Get ticker
# -----------------------------------------------------------------------------
ticker = ed.getsp100()

# %% --------------------------------------------------------------------------
# Run downloader to download EDGAR files
# -----------------------------------------------------------------------------
dest_folder = 'html_files'
ed.download_files_10k(ticker, dest_folder)

# %% --------------------------------------------------------------------------
# Clean files and convert to .txt format
# -----------------------------------------------------------------------------
input_folder = 'html_files'
dest_folder = 'txt_files'
ec.write_clean_html_text_files(input_folder, dest_folder)

# %% --------------------------------------------------------------------------
# Calculate counts of sentiment words
# -----------------------------------------------------------------------------
input_file = 'txt_files'
output_file = 'word_count.csv'
esw.write_document_sentiments(input_file,output_file)

# %% --------------------------------------------------------------------------
# Add yahoo data to word_counts
# -----------------------------------------------------------------------------
esw.add_returns_to_df(output_file)

# %% --------------------------------------------------------------------------
# Plot results
# -----------------------------------------------------------------------------
esw.plot_data('count_returns.csv', 'Negative', '1daily_return')

# %%
