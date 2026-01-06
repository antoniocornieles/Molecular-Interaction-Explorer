#!/usr/bin/env python3
# run_kemuri_demo.py

import os
import webbrowser
import requests
import numpy as np
from Bio import Entrez, SeqIO
from scipy.integrate import odeint
import plotly.graph_objects as go
import py3Dmol
from openai import OpenAI

client = OpenAI()

# ============================
# Demo Accessions
# ============================
DEMO_ACCESSIONS = {
    "H1N1 (2009)": "NC_026434.1",
    "H3N2": "KJ609208.1",
    "H5N1": "EF541467.1"
}

# ============================
# Configuration
# ============================
Entrez.email = "antonio.e.cornieles@gmail.com"
OpenAI.api_key = os.environ.get("YOUR_KEY_HERE")  # REQUIRED

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Sequence Fetch
# ============================
def fetch_sequence(accession):
    print(f"[+] Fetching nucleotide sequence {accession}")

    handle = Entrez.efetch(
        db="nucleotide",
        id=accession,
        rettype="fasta",
        retmode="text"
    )

    record = SeqIO.read(handle, "fasta")
    fasta_path = os.path.join(OUTPUT_DIR, "sequence.fasta")
    SeqIO.write(record, fasta_path, "fasta")

    print(f"[✓] Sequence saved → {fasta_path}")
    print(f"    Length: {len(record.seq)} bp")

    return str(record.seq)

# ============================
# PDB Download
# ============================
def download_pdb(pdb_id="3TI6"):
    print(f"[+] Downloading PDB {pdb_id}")

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)
    r.raise_for_status()

    pdb_path = os.path.join(OUTPUT_DIR, f"{pdb_id}.pdb")
    with open(pdb_path, "w") as f:
        f.write(r.text)

    print(f"[✓] PDB saved → {pdb_path}")
    return pdb_path

# ============================
# 3D Visualization
# ============================
def visualize_3d(pdb_path):
    print("[+] Generating 3D structure visualization")

    with open(pdb_path) as f:
        pdb_data = f.read()

    view = py3Dmol.view(width=900, height=650)
    view.addModel(pdb_data, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()

    html_path = pdb_path.replace(".pdb", "_3d.html")
    view.write_html(html_path)

    print(f"[✓] 3D HTML saved → {html_path}")
    webbrowser.open(f"file://{os.path.abspath(html_path)}")

# ============================
# Helix Visualization
# ============================
def build_helix(seq, radius=2.0, bases_per_turn=10, pitch=1.5):
    x, y, z, colors = [], [], [], []
    base_colors = {'A':'red','T':'blue','U':'blue','G':'green','C':'orange','N':'gray'}

    for i, b in enumerate(seq[:300]):
        angle = 2 * np.pi * (i / bases_per_turn)
        x.append(radius * np.cos(angle))
        y.append(radius * np.sin(angle))
        z.append(pitch * i / bases_per_turn)
        colors.append(base_colors.get(b.upper(), "gray"))

    return x, y, z, colors

def plot_helix(seq):
    print("[+] Plotting nucleotide helix")

    x, y, z, colors = build_helix(seq)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='lines',
        line=dict(width=3, color='black'),
        name="Backbone"
    ))

    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(size=4, color=colors),
        name="Bases"
    ))

    fig.update_layout(
        title="Nucleotide Helix (First 300 bases)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False)
        )
    )

    fig.show()

# ============================
# AI Explanation
# ============================
def run_ai_summary(data_blob, title):
    prompt = f"""
You are a computational biology assistant.

Title: {title}

Data:
{data_blob}

Explain the cause-and-effect mechanism clearly and scientifically.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content


# ============================
# Reaction Simulation
# ============================
def simulate_reaction_with_ai():
    print("[+] Simulating enzyme inhibition reaction")

    E0, S0, I0 = 1.0, 10.0, 0.5
    Km, Ki, Vmax = 2.0, 0.2, 1.0

    def dPdt(P, t):
        S = S0 - P
        Km_app = Km * (1 + I0 / Ki)
        return Vmax * S / (Km_app + S)

    def dPdt_no_inhib(P, t):
        S = S0 - P
        return Vmax * S / (Km + S)

    t = np.linspace(0, 10, 200)
    P = odeint(dPdt, 0, t).flatten()
    P_no = odeint(dPdt_no_inhib, 0, t).flatten()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=P_no, name="No inhibitor"))
    fig.add_trace(go.Scatter(x=t, y=P, name="With inhibitor"))

    fig.update_layout(
        title="Enzyme Inhibition Dynamics",
        xaxis_title="Time",
        yaxis_title="Product Formed"
    )
    fig.show()

    data_blob = f"""
Time (first 10): {t[:10]}
No inhibitor (first 10): {P_no[:10]}
With inhibitor (first 10): {P[:10]}
"""

    ai_summary = run_ai_summary(data_blob, "Competitive Enzyme Inhibition")
    print("\n--- AI ANALYSIS ---\n")
    print(ai_summary)

# ============================
# Main
# ============================
def main():
    print("=== Antonio Cornieles | Kemuri BioSim Demo ===\n")

    seq = fetch_sequence(DEMO_ACCESSIONS["H1N1 (2009)"])
    pdb_path = download_pdb("3TI6")

    visualize_3d(pdb_path)
    plot_helix(seq)
    simulate_reaction_with_ai()

    print("\n[✓] Demo complete.")

if __name__ == "__main__":
    main()
