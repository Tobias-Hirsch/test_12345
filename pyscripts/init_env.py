import os
import matplotlib
from dotenv import load_dotenv

# Kommentar
load_dotenv()

# Kommentar
os.environ['MILVUS_HOST'] = '127.0.0.1'
#'10.32.7.109'
os.environ['MILVUS_PORT'] = '19530'

# Kommentar
os.environ['MPLBACKEND'] = 'Agg'
matplotlib.use('Agg', force=True)

# Kommentar
matplotlib.interactive(False)

# Kommentar
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

print("Environment initialized successfully")
print(f"Milvus Host: {os.getenv('MILVUS_HOST')}")
print(f"Milvus Port: {os.getenv('MILVUS_PORT')}") 