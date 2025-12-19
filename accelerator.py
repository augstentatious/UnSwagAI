refactor: Rename to UnSwag; enforce 1-bit structural constraints

Formalizes the transition from 'Sophia-Pallas' to 'UnSwag'. This rebrand aligns the repository with the core 'Sophia Protocol' hypothesis: that rigorous engineering constraints (specifically, 1-bit activation packing on TPUs) are isomorphic to robust alignment constraints.

Changes include:
- Renamed package root to `unswag/` to enforce the "no-bloat" philosophy—liquidating the "toxic asset" of 16-bit activation debt.
- Updated README with technical specifications for the Pallas SRAM-to-Register bitmask kernel.
- Added 'Proof of Convergence' logs demonstrating successful gradient flow under extreme quantization constraints.