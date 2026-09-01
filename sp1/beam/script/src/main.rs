use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use gat_sp1_beam_lib::{
    evaluate, BeamClaimInput, BeamClaimOutput, GuestInput, PublicValues, NUMERIC_PROFILE_DIGEST,
    NUMERIC_PROFILE_ID, PROOF_TYPE, RECEIPT_FORMAT, REQUEST_FORMAT, SCHEMA_VERSION,
    SP1_CIRCUIT_VERSION as PINNED_SP1_CIRCUIT_VERSION, SP1_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sp1_sdk::{
    blocking::{ProveRequest, Prover, ProverClient},
    include_elf, Elf, HashableKey, ProvingKey, SP1Proof, SP1ProofWithPublicValues, SP1Stdin,
    SP1_CIRCUIT_VERSION as SDK_SP1_CIRCUIT_VERSION,
};

const ELF: Elf = include_elf!("gat-sp1-beam-program");

#[derive(Debug, Parser)]
#[command(about = "Execute, prove, or verify GAT's bounded SP1 beam claim")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Execute {
        #[arg(long)]
        request: PathBuf,
    },
    Prove {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    Verify {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    format: String,
    schema_version: u64,
    sp1_version: String,
    sp1_circuit_version: String,
    transition_event_seq: u64,
    public_statement_digest: String,
    numeric_contract: NumericContract,
    claim: Claim,
    expected_public_values_hex: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NumericContract {
    profile_id: String,
    profile_digest: String,
    arithmetic: String,
    rounding: String,
    overflow: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Claim {
    input: JsonClaimInput,
    output: JsonClaimOutput,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonClaimInput {
    yield_strength_milli_mpa: u64,
    plastic_section_modulus_mm3: u64,
    factored_demand_milli_n_mm: u64,
    resistance_factor_ppm: u64,
    numeric_profile_digest: String,
    model_contract_digest: String,
    validation_profile_digest: String,
    evidence_digest: String,
    evidence_source_digest: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct JsonClaimOutput {
    nominal_milli_n_mm: u128,
    available_milli_n_mm: u128,
    verdict: String,
    computation_digest: String,
}

#[derive(Debug, Serialize)]
struct Receipt {
    format: &'static str,
    schema_version: u64,
    sp1_version: &'static str,
    sp1_circuit_version: &'static str,
    proof_type: &'static str,
    program_digest: String,
    verifying_key_digest: String,
    proof_artifact_digest: String,
    public_values_hex: String,
    public_statement_digest: String,
    computation_result_digest: String,
    proof_verified: bool,
    cycles: Option<u64>,
}

struct CheckedRequest {
    guest: GuestInput,
    output: BeamClaimOutput,
    expected_public_values: Vec<u8>,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Execute { request } => execute(&request),
        Command::Prove {
            request,
            proof,
            receipt,
        } => prove(&request, &proof, &receipt),
        Command::Verify {
            request,
            proof,
            receipt,
        } => verify(&request, &proof, &receipt),
    }
}

fn execute(request_path: &Path) -> Result<()> {
    let request = read_request(request_path)?;
    let mut stdin = SP1Stdin::new();
    stdin.write(&request.guest);
    let client = ProverClient::from_env();
    let (public_values, report) = client
        .execute(ELF, stdin)
        .run()
        .context("SP1 execution failed")?;
    if public_values.as_slice() != request.expected_public_values.as_slice() {
        bail!("guest public values differ from the checked request");
    }
    println!(
        "SP1 beam guest execution passed ({} cycles)",
        report.total_instruction_count()
    );
    Ok(())
}

fn prove(request_path: &Path, proof_path: &Path, receipt_path: &Path) -> Result<()> {
    let request = read_request(request_path)?;
    let client = ProverClient::from_env();
    let proving_key = client.setup(ELF).context("SP1 setup failed")?;
    let mut stdin = SP1Stdin::new();
    stdin.write(&request.guest);
    let proof = client
        .prove(&proving_key, stdin)
        .run()
        .context("SP1 proof generation failed")?;
    validate_public_values(&proof, &request.expected_public_values)?;
    client
        .verify(&proof, proving_key.verifying_key(), None)
        .context("SP1 proof verification failed")?;
    proof.save(proof_path).context("could not save SP1 proof")?;
    write_receipt(
        receipt_path,
        proof_path,
        &proof,
        &proving_key.verifying_key().bytes32_raw(),
        &request,
    )
}

fn verify(request_path: &Path, proof_path: &Path, receipt_path: &Path) -> Result<()> {
    let request = read_request(request_path)?;
    let proof = SP1ProofWithPublicValues::load(proof_path).context("could not load SP1 proof")?;
    validate_public_values(&proof, &request.expected_public_values)?;
    let client = ProverClient::from_env();
    let proving_key = client.setup(ELF).context("SP1 setup failed")?;
    client
        .verify(&proof, proving_key.verifying_key(), None)
        .context("SP1 proof verification failed")?;
    write_receipt(
        receipt_path,
        proof_path,
        &proof,
        &proving_key.verifying_key().bytes32_raw(),
        &request,
    )
}

fn read_request(path: &Path) -> Result<CheckedRequest> {
    let text = fs::read_to_string(path).context("could not read beam proof request")?;
    let request: Request = serde_json::from_str(&text).context("invalid beam proof request")?;
    if request.format != REQUEST_FORMAT || request.schema_version != SCHEMA_VERSION {
        bail!("unsupported beam proof request format or schema");
    }
    if request.sp1_version != SP1_VERSION
        || request.sp1_circuit_version != PINNED_SP1_CIRCUIT_VERSION
        || SDK_SP1_CIRCUIT_VERSION != PINNED_SP1_CIRCUIT_VERSION
        || request.transition_event_seq == 0
    {
        bail!("request is not bound to the pinned SP1 version and transition");
    }
    let profile_digest =
        decode_digest(&request.numeric_contract.profile_digest, "numeric profile")?;
    if request.numeric_contract.profile_id != NUMERIC_PROFILE_ID
        || profile_digest != NUMERIC_PROFILE_DIGEST
        || request.numeric_contract.arithmetic != "checked-integer"
        || request.numeric_contract.rounding != "nearest-ties-to-even"
        || request.numeric_contract.overflow != "checked"
    {
        bail!("request numeric contract differs from the v1 guest contract");
    }
    let input = BeamClaimInput {
        yield_strength_milli_mpa: request.claim.input.yield_strength_milli_mpa,
        plastic_section_modulus_mm3: request.claim.input.plastic_section_modulus_mm3,
        factored_demand_milli_n_mm: request.claim.input.factored_demand_milli_n_mm,
        resistance_factor_ppm: request.claim.input.resistance_factor_ppm,
        numeric_profile_digest: decode_digest(
            &request.claim.input.numeric_profile_digest,
            "claim profile",
        )?,
        model_contract_digest: decode_digest(
            &request.claim.input.model_contract_digest,
            "model contract",
        )?,
        validation_profile_digest: decode_digest(
            &request.claim.input.validation_profile_digest,
            "validation profile",
        )?,
        evidence_digest: decode_digest(&request.claim.input.evidence_digest, "evidence")?,
        evidence_source_digest: decode_digest(
            &request.claim.input.evidence_source_digest,
            "evidence source",
        )?,
    };
    let output = evaluate(&input).map_err(anyhow::Error::msg)?;
    if request.claim.output.nominal_milli_n_mm != output.nominal_milli_n_mm
        || request.claim.output.available_milli_n_mm != output.available_milli_n_mm
        || request.claim.output.verdict != output.verdict
        || request.claim.output.computation_digest != hex::encode(output.computation_digest)
    {
        bail!("request claim output differs from checked evaluation");
    }
    let statement = decode_digest(&request.public_statement_digest, "public statement")?;
    let expected = PublicValues::from_claim(statement, &input, &output).encode();
    if request.expected_public_values_hex != hex::encode(&expected) {
        bail!("request public values differ from checked evaluation");
    }
    Ok(CheckedRequest {
        guest: GuestInput {
            public_statement_digest: statement,
            claim: input,
        },
        output,
        expected_public_values: expected,
    })
}

fn validate_public_values(proof: &SP1ProofWithPublicValues, expected: &[u8]) -> Result<()> {
    if proof.sp1_version != SDK_SP1_CIRCUIT_VERSION {
        bail!("proof was generated for a different SP1 circuit version");
    }
    if !matches!(&proof.proof, SP1Proof::Core(_)) {
        bail!("proof is not the declared SP1 core proof type");
    }
    if proof.public_values.as_slice() != expected {
        bail!("proof public values differ from the checked request");
    }
    Ok(())
}

fn write_receipt(
    path: &Path,
    proof_path: &Path,
    proof: &SP1ProofWithPublicValues,
    verifying_key_commitment: &[u8; 32],
    request: &CheckedRequest,
) -> Result<()> {
    let proof_bytes = fs::read(proof_path).context("could not read saved SP1 proof")?;
    let public_values = proof.public_values.as_slice();
    let receipt = Receipt {
        format: RECEIPT_FORMAT,
        schema_version: SCHEMA_VERSION,
        sp1_version: SP1_VERSION,
        sp1_circuit_version: PINNED_SP1_CIRCUIT_VERSION,
        proof_type: PROOF_TYPE,
        program_digest: sha256(&*ELF),
        verifying_key_digest: sha256(verifying_key_commitment),
        proof_artifact_digest: sha256(&proof_bytes),
        public_values_hex: hex::encode(public_values),
        public_statement_digest: hex::encode(request.guest.public_statement_digest),
        computation_result_digest: hex::encode(request.output.computation_digest),
        proof_verified: true,
        cycles: None,
    };
    let text = serde_json::to_string_pretty(&receipt)? + "\n";
    fs::write(path, text).context("could not write SP1 receipt")
}

fn decode_digest(value: &str, label: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(value).with_context(|| format!("{label} is not hexadecimal"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("{label} is not 32 bytes"))
}

fn sha256(value: &[u8]) -> String {
    hex::encode(Sha256::digest(value))
}
