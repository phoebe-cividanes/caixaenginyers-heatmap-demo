# How to Run the App with Custom Data Path

## Option 1: Using run_app.py (Recommended)

### With default data path:
```bash
python run_app.py
```
This will look for `data/municipalities_scored_clean.csv`

### With custom data path:
```bash
python run_app.py --data-path /path/to/your/data.csv
```

Example:
```bash
python run_app.py --data-path data/out_with_scores.csv
```

## Option 2: Direct Streamlit Command

### With default data path:
```bash
streamlit run "app data/streamlit_app_scored.py"
```

### With custom data path:
```bash
streamlit run "app data/streamlit_app_scored.py" -- --data-path /path/to/your/data.csv
```

Example:
```bash
streamlit run "app data/streamlit_app_scored.py" -- --data-path data/out_with_scores.csv
```

**Note:** The `--` is important! It separates Streamlit's arguments from your app's arguments.

## Option 3: With uv

### With default data path:
```bash
uv run python run_app.py
```

### With custom data path:
```bash
uv run python run_app.py --data-path data/out_with_scores.csv
```

## Expected CSV Format

Your CSV file should contain these columns:
- `municipio` - Municipality name
- `provincia` - Province name
- `poblacion_total` - Total population
- `num_bancos` - Number of banks
- `normalized_bank_count` - Normalized bank count [0-1]
- `economic_score` - Raw economic score
- `social_score` - Raw social score
- `economic_score_normalized` - Normalized economic score [0-100]
- `social_score_normalized` - Normalized social score [0-100]
- `total_score_alpha_0`, `total_score_alpha_25`, `total_score_alpha_50`, `total_score_alpha_75`, `total_score_alpha_100` - Total scores for different alpha values

## Examples

### After running your pipeline:
```bash
# Generate scores
uv run python scripts/apply_scoring_pipeline.py \
  --data-path data/merged_es_imputed.csv \
  --out-path data/out_with_scores.csv

# Clean scores
uv run python scripts/clean_and_normalize_scores.py \
  --input-path data/out_with_scores.csv \
  --output-path data/clean_scores.csv

# Run app with cleaned data
python run_app.py --data-path data/clean_scores.csv
```

### Quick test with custom output location:
```bash
python run_app.py --data-path ../out/test.csv
```

## Troubleshooting

### "Scored data not found" error
Make sure the file path is correct and the file exists:
```bash
ls -l data/out_with_scores.csv
```

### "Column not found" error
Make sure you've run the clean_and_normalize_scores.py script to add normalized columns:
```bash
uv run python scripts/clean_and_normalize_scores.py \
  --input-path data/out_with_scores.csv \
  --output-path data/clean_scores.csv
```

### Relative vs Absolute Paths
- Relative paths are relative to where you run the command
- Use absolute paths if you're unsure: `/Users/marc/marc/HackUAB/caixaenginyers-heatmap-demo/data/out_with_scores.csv`
