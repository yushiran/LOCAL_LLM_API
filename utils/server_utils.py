import socket
import GPUtil
import logging
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def get_local_ip():
    """Get the local IP address of the machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"
    


def find_available_gpu():
    """Find the GPU with the most available memory using multiple methods"""
    # Method 1: Try using GPUtil library
    try:
        logger.info("Detecting available GPUs using GPUtil...")
        available_gpus = GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.5, maxMemory=0.5)
        if available_gpus:
            selected_gpu = available_gpus[0]
            gpu_info = GPUtil.getGPUs()[selected_gpu]
            logger.info(f"Selected GPU {selected_gpu} with {gpu_info.memoryFree}MB free memory (GPUtil method)")
            return selected_gpu
        
        # If no GPU meets the strict criteria, find the one with most available memory
        gpus = GPUtil.getGPUs()
        if gpus:
            best_gpu = max(gpus, key=lambda gpu: gpu.memoryFree)
            logger.info(f"Selected GPU {best_gpu.id} with {best_gpu.memoryFree}MB free memory (GPUtil method)")
            return best_gpu.id
    except Exception as e:
        logger.warning(f"Error finding available GPU using GPUtil: {e}")
    
    # Method 2: Try using nvidia-smi directly
    try:
        import subprocess
        import csv
        from io import StringIO
        
        logger.info("Detecting available GPUs using nvidia-smi...")
        
        # Run nvidia-smi to get GPU info
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse the output
        best_gpu = None
        max_free_mem = 0
        
        logger.info("Available GPUs:")
        reader = csv.reader(StringIO(result.stdout))
        for row in reader:
            if len(row) == 3:
                idx, used, total = map(int, map(str.strip, row))
                free = total - used
                logger.info(f"GPU {idx}: {free} MB free of {total} MB")
                
                if free > max_free_mem:
                    max_free_mem = free
                    best_gpu = idx
        
        if best_gpu is not None:
            logger.info(f"Selected GPU {best_gpu} with {max_free_mem}MB free memory (nvidia-smi method)")
            return best_gpu
    except Exception as e:
        logger.warning(f"Error finding available GPU using nvidia-smi: {e}")
    
    logger.warning("No available GPU found, will use CPU")
    return None
