"""Freeze the existing context preparation and Raven-recipe training adapter."""
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
PREP = PARENT / 'cypher-context-35-2026-11-17'
BASE = PARENT / 'cypher-3.0.0-2026-11-17'
SOURCE = HERE / 'source'


def main():
    SOURCE.mkdir(exist_ok=True)
    for name in ('context.py', 'context_planes.h'):
        shutil.copy2(PREP / name, SOURCE / name)
    context = (SOURCE / 'context.py').read_text()
    context = context.replace("ROOT = HERE.parents[1]", "ROOT = HERE.parents[2]")
    context = context.replace("BASE = HERE.parent / 'cypher-3.0.0-2026-11-17'", "BASE = HERE.parents[1] / 'cypher-3.0.0-2026-11-17'")
    context = context.replace("DATASET = HERE.parent / 'davi-normal-results-2026-11-17'", "DATASET = HERE.parents[1] / 'raven-2.3.9-overnight-2026-11-16'")
    (SOURCE / 'context.py').write_text(context)
    network = (BASE / 'source/network.py').read_text()
    assert network.count('in_channels=31') == 1
    (SOURCE / 'network.py').write_text(network.replace('in_channels=31', 'in_channels=35'))
    prep = (PREP / 'prepare.py').read_text()
    prep = prep.replace("def main():\n", "def main():\n    global HERE\n    HERE = Path(__file__).resolve().parent.parent\n")
    prep = prep.replace("source = json.loads((DATASET / 'source-manifest.json').read_text())", "source = {'outputs': json.loads((BASE.parent / 'davi-normal-results-2026-11-17/source-manifest.json').read_text())['source_files']}")
    (SOURCE / 'prepare.py').write_text(prep)
    training = (BASE / 'source/training.py').read_text()
    training = training[training.index('def train('):]
    training = training.replace('artifacts/Davi_3.0.0_Cypher', 'artifacts/context-35')
    end = training.index('    executable = HERE.parent')
    training = training[:end] + '''    save_json("artifacts/context-35/manifest.json", {
        "at": now(), "status": "trained weights only; no compatible executable packaged",
        "input_planes": 35, "architecture_sha256": sha(HERE / "source/network.py"),
        "selection": "Lowest combined validation loss across 15 epochs",
        "files": {p.name: sha(p) for p in artifact.iterdir() if p.is_file()},
        "runtime": "Requires a separately built 35-plane runtime; old Raven/Cypher EXE incompatible"})
    event("complete", artifact=str(artifact), best_validation_loss=best)
'''
    (SOURCE / 'training.py').write_text(training)


if __name__ == '__main__':
    main()
