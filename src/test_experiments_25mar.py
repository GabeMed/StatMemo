import time
import importlib.util
import argparse
import sys
from pathlib import Path


SRC_DIR  = Path(__file__).resolve().parent          
ROOT_DIR = SRC_DIR.parent                           
sys.path.append(str(ROOT_DIR))                      


from src.statisticaly_pure_memoizer import StatisticalyPureMemoizer

MEMO = StatisticalyPureMemoizer(  
    number_of_executions=10,
    max_error=0.5,
    confidence_level=0.95
)



# Lista de argumentos e nº de rodadas
EXPERIMENTS = [
    {
        "name": "genetic_algorithm",
        "file": ROOT_DIR / "experiments" / "genetic_algorithm_traveling_sales_person.py",
        "func_name": "ga_tsp",
        "args": [],
        "rounds": 12,
    },
    {
        "name": "belief_propagation",
        "file": ROOT_DIR / "experiments" / "belief_propagation" / "belief_propagation.py",
        "func_name": "belief_propagation",
        "args": [500,500,500,500,500],
        "rounds": 12,
    },
    
]


def load_function(file_path: Path, func_name: str):
    """Import <file_path> dynamically and return the callable <func_name>."""
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)



# Aplica o memoizador à função alvo
def run_experiment(exp: dict):
    mod = load_function(exp["file"], exp["func_name"])
    func = MEMO(mod)  

    arg_list = exp["args"] if exp["args"] else [None]

    try:
        nice_path = exp["file"].relative_to(ROOT_DIR)
    except ValueError:
        nice_path = exp["file"]

    print(f"\n### {exp['name']} | {nice_path} ###")
    header = f"{'round':>5}  {'arg':>12}  {'elapsed (s)':>12}"
    print(header)
    print("-" * len(header))

    
    for rnd in range(1, exp["rounds"] + 1):
        for arg in arg_list:
            t0 = time.perf_counter()

            # Aceita chamada sem argumentos
            if arg is None:
                func()              
            elif isinstance(arg, tuple):
                func(*arg)          
            else:
                func(arg)           

            # Mede tempo e imprime tabela
            elapsed = time.perf_counter() - t0
            print(f"{rnd:5d}  {str(arg):>12}  {elapsed:12.4f}")
    print("completed")



# CLI: permite filtrar quais experimentos rodar com --only
def main():
    parser = argparse.ArgumentParser(
        description="Run experiments with StatisticalyPureMemoizer."
    )
    parser.add_argument(
        "--only", metavar="NAME", nargs="*",
        help="Run only the experiments whose names match."
    )
    args = parser.parse_args()

    chosen = [
        exp for exp in EXPERIMENTS
        if not args.only or exp["name"] in args.only
    ]

    for exp in chosen:
        run_experiment(exp)


if __name__ == "__main__":
    main()
