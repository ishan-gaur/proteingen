# MSA Acquisition

Tools and strategies for obtaining multiple sequence alignments of homologous proteins.

!!! note "Coming soon"
    This module will provide convenience wrappers or links to established tools. For now, we describe the landscape.

## Sequence-based homolog search

Standard tools for finding proteins with similar sequences:

| Tool | Description | Link |
|------|-------------|------|
| **MMseqs2** | Fast sequence search — orders of magnitude faster than BLAST | [github.com/soedinglab/MMseqs2](https://github.com/soedinglab/MMseqs2) |
| **jackhmmer** | Iterative HMM search against UniRef/UniProt | Part of [HMMER](http://hmmer.org/) |
| **ColabFold MSA server** | Free web API for fast MSA generation | [colabfold.com](https://colabfold.com/) |

**When to use sequence-based:** Default choice. Works well when your protein has identifiable homologs in UniRef. Most protein families have thousands of homologs.

## Structure-based homolog search

For proteins with few sequence homologs (e.g. *de novo* designs, orphan proteins), search by structural similarity:

| Tool | Description | Link |
|------|-------------|------|
| **Foldseek** | Fast structural search against AlphaFold DB or PDB | [github.com/steineggerlab/foldseek](https://github.com/steineggerlab/foldseek) |

**When to use structure-based:** When sequence search returns too few hits (< 100), or when you want to include remote homologs that share fold but not sequence. Combine the structural hits with sequence-based hits for a richer MSA.

## From MSA to dataset

Once you have your MSA (as a FASTA file), proceed to [MSA → Dataset](msa-to-dataset.md) to prepare it for training.

<!-- TODO: add convenience functions or CLI for running MMseqs2/Foldseek from protstar -->
