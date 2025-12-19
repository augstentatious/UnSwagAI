import jax
import sys
import time
import shutil

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

def _print_centered(text):
    columns = shutil.get_terminal_size().columns
    print(text.center(columns))

def boot_sequence():
    """
    The UnSwag Startup Animation.
    """
    # 1. The Logo
    logo = f"""
{RED}
██╗   ██╗███╗   ██╗███████╗██╗    ██╗ █████╗  ██████╗ 
██║   ██║████╗  ██║██╔════╝██║    ██║██╔══██╗██╔════╝ 
██║   ██║██╔██╗ ██║███████╗██║ █╗ ██║███████║██║  ███╗
██║   ██║██║╚██╗██║╚════██║██║███╗██║██╔══██║██║   ██║
╚██████╔╝██║ ╚████║███████║╚███╔███╔╝██║  ██║╚██████╔╝
 ╚═════╝ ╚═╝  ╚═══╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝ ╚═════╝ {RESET}
    """
    print(logo)
    time.sleep(0.2)
    
    # 2. Hardware Recon
    sys.stdout.write(f"{BOLD}[SYSTEM]    {RESET}Scanning silicon... ")
    sys.stdout.flush()
    time.sleep(0.4)
    
    try:
        devices = jax.local_devices()
        device_type = devices[0].platform.upper()
        count = len(devices)
        status = f"{GREEN}ONLINE{RESET}"
        hw_msg = f"{count}x {device_type}"
    except:
        device_type = "CPU"
        status = f"{YELLOW}FALLBACK{RESET}"
        hw_msg = "Standard Execution"

    print(f"[{status}]")
    print(f"{BOLD}[HARDWARE]  {RESET}{hw_msg} detected.")
    
    # 3. The Promise
    print(f"{BOLD}[KERNEL]    {RESET}Loading 1-bit Isomorphism...", end="")
    sys.stdout.flush()
    # Fake loading bar for the "feel"
    for _ in range(3):
        time.sleep(0.1)
        sys.stdout.write(".")
        sys.stdout.flush()
    print(f" {GREEN}FUSED{RESET}")
    
    # 4. The Flex
    print(f"{BOLD}[MEMORY]    {RESET}Swag Removal Target: {CYAN}96.875%{RESET}")
    print(f"{BOLD}[STATUS]    {RESET}The Memory Wall is now optional.\n")
    
    # 5. Divider
    print(f"{RED}{'—' * 40}{RESET}")

def monitor(iterable, desc="Training"):
    """
    A minimal, UnSwag-styled progress bar.
    Replaces tqdm with something more 'industrial'.
    """
    total = len(iterable)
    
    # The 'Lion' cursor
    cursor = "🦁" 
    
    print(f"\n{BOLD}{desc} initialized.{RESET}")
    print(f"Offloading gradients to the ether...")
    
    start_time = time.time()
    
    for i, item in enumerate(iterable):
        yield item
        
        # Update every few steps to save I/O
        if i % 10 == 0 or i == total - 1:
            percent = (i + 1) / total
            bar_len = 30
            filled_len = int(bar_len * percent)
            
            # The Bar
            bar = f"{RED}█{RESET}" * filled_len + f"{CYAN}-{RESET}" * (bar_len - filled_len)
            
            # Rate
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            
            sys.stdout.write(f"\r{cursor} [{bar}] {percent:.0%} | {rate:.2f} it/s | {CYAN}1-Bit Mode{RESET}")
            sys.stdout.flush()
            
    print(f"\n{GREEN}Cycle Complete.{RESET}\n")
