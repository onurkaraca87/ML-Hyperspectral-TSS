## Step 1 – Band Inspector
```bash
python CatBoost/01_band_inspector.py \
  --input path/to/prisma.tif \
  --output configs/band_indices_prisma.json



## Step 2 – Apply CatBoost Model
```bash
python CatBoost/02_predict_catboost.py \
  --input path/to/prisma.tif \
  --model path/to/model.pkl \
  --band-indices configs/band_indices_prisma.json \
  --out-dir outputs
