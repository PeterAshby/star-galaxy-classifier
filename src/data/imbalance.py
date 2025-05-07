from scripts.run_inference import load_model
from pathlib import Path

model = load_model('LR')
print(model)
