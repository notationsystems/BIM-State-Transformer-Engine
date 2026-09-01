# SP1 bounded beam guest v1

## Claim

The first real GAT proof program is deliberately small. It proves that SP1
v6.5.0 executed this checked fixed-point calculation for one ledger-bound
posterior beam slice:

```text
nominal_milli_N_mm   = fy_milli_MPa * Zx_mm3
available_milli_N_mm = round_ties_even(nominal * 900000 / 1000000)
verdict              = PASS iff available >= factored_demand
```

The request binds the input to the material-evidence digest and source digest,
the engineering model and validation-profile digests, and the canonical public
statement for the exact evidence transition and ledger head. The guest commits:

```text
domain
public-statement digest
fixed-computation digest
nominal capacity
available capacity
demand
PASS/FAIL
```

Python and Rust use the same domain-separated big-endian byte encoding. A
checked known vector in both test suites detects arithmetic, serialization,
rounding, or digest drift before proof generation.

## Numeric profile

`beam-milli-mpa-mm3-milli-nmm-v1` uses:

| Quantity | Integer representation |
|---|---:|
| yield strength | `MPa * 1000`, nearest ties to even |
| plastic section modulus | `m3 * 1e9` = `mm3`, nearest ties to even |
| factored demand | `N*m * 1e6` = milli-`N*mm`, nearest ties to even |
| resistance factor | parts per million; v1 requires exactly `900000` |

Inputs are checked `u64` values, intermediate capacity products are checked
`u128` values, and division uses integer nearest-ties-to-even rounding. The
guest rejects a changed numeric-profile digest or resistance factor.

## End-to-end run

SP1 supports its native toolchain on Linux and macOS. After installing the
pinned v6.5.0 toolchain, build and run:

```bash
cargo test --locked --manifest-path sp1/beam/Cargo.toml -p gat-sp1-beam-lib
cargo build --locked --release --manifest-path sp1/beam/Cargo.toml -p gat-sp1-beam
python -m gat.demo.beam_sp1 out/beam-sp1 \
  --executable sp1/beam/target/release/gat-sp1-beam
```

The dedicated Linux CI lane performs this complete sequence with the CPU
prover. It writes the proof, backend-verification receipt, and GAT computation
proof manifest, then invokes the SP1 verifier again through GAT's fail-closed
backend adapter. Success requires both manifest/ledger binding and actual SP1
verification.

## Artifacts and commitments

| File | Meaning |
|---|---|
| `beam_sp1_request.json` | Strict guest input, expected public values, and ledger public-statement digest |
| `beam.sp1-proof` | Binary SP1 core proof generated for the pinned guest ELF |
| `beam_sp1_receipt.json` | Program, verifying-key, proof, statement, and result commitments returned after backend verification |
| `beam_sp1_manifest.json` | Backend-neutral GAT manifest binding the proof to the exact ledger transition and later fixed-computation assessment |

The program digest is SHA-256 of the guest ELF. The verifying-key digest is
SHA-256 of SP1's 32-byte verifying-key commitment. The proof-artifact digest is
SHA-256 of the exact saved proof bytes.

## Explicit non-claims

The guest proves deterministic computation integrity only. It does **not**
prove:

- that the certificate or measurement is truthful;
- that the Gaussian conditioning was correct;
- that AISC F2-1 is applicable to the real member;
- that the probabilistic `SATISFIED / VIOLATED / UNRESOLVED` assessment is
  correct;
- that the member or building is safe;
- that a professional approved fabrication, installation, or another physical
  action.

Those claims remain separate evidence, state-transition, validation, decision,
and authority records. This narrow boundary is what makes the proof auditable.
